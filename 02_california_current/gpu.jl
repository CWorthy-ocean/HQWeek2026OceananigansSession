# # California Current at 1/36° on an NVIDIA GPU (CUDA), physics only
#
# The configuration of ../CaliforniaCurrent.jl/california_current_jra55_2010_2011.jl (2010–2011 at 1/36°:
# 666 × 648 × 42, ≈ 2.1–2.7 km in x, 3.1 km in y, Δt = 5 min; 2010–2011 took 12.9 h on one H100), reading the workshop
# data instead of downloading: GLORYS from the 1/4° files (`GLORYSQuarterDegree`, not daily 1/12° GLORYS) and JRA55 cut
# to the region (`RegionalJRA55`). Unlike metal.jl it runs in Float64 and needs none of the Metal workarounds.
# Run with
#
#     julia --project --threads=4 02_california_current/gpu.jl
#
# Rerunning picks up from the latest checkpoint in `output_directory`.

# Before `using NumericalEarth`: it reads the data directory when it loads
ENV["NUMERICALEARTH_DATA_DIRECTORY"] = "data"

using Dates: Date, DateTime, Day, Millisecond, year
using Printf
using CUDA
using NumericalEarth
using Oceananigans
using Oceananigans.Units # just defines stuff like day = days = 24*60*60
using HQWeek2026OceananigansSession # the regional datasets and the boundary, drag and sponge functions

#####
##### Warning: the GLORYS files cover -134:-115.5E and 30:48N for 2010–2012; a different region or period needs
##### different data (usually `GLORYSDaily()` with a `region`, downloaded with CopernicusMarine)
#####

#####
##### Settings
#####

arch       = GPU()             # CUDA
resolution = 1/36              # degrees
longitude  = (-134, -115.5)    # °E
latitude   = (30, 48)          # °N
Nz         = 42
depth      = 5500              # m
Δt         = 5minutes          # advective CFL 0.41 at 2.85 m s⁻¹ on the smallest Δx (2.1 km, 48°N)
initial_Δt = 2minutes          # the run starts at this and ramps up to Δt
first_date = DateTime(2010, 1, 1)
last_date  = DateTime(2011, 12, 31)

output_directory = joinpath(@__DIR__, "output", "gpu_36th")

#####
##### Defining the grid
#####

Nx = round(Int, (longitude[2] - longitude[1]) / resolution)
Ny = round(Int, (latitude[2] - latitude[1]) / resolution)

# Exponential levels; scale = 700 m gives Δz ≈ 0.44 m at the surface, ≈ 20 m at 100 m, ≈ 940 m at the bottom
z = ExponentialDiscretization(Nz, -depth, 0; scale = 700, mutable = true)

underlying_grid = LatitudeLongitudeGrid(arch; size = (Nx, Ny, Nz), halo = (7, 7, 7),
                                        longitude, latitude, z,
                                        topology = (Bounded, Bounded, Bounded))

bottom_height = regrid_bathymetry(underlying_grid; minimum_depth = 10, major_basins = 1, interpolation_passes = 5)
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height);
                            active_cells_map = true)

@info "Grid" grid

#####
##### Getting all the boundary data - usually GLORYSDaily not GLORYSQuarterDegree but I downsampled to reduce storage
#####

glorys_dates  = first_date:Day(1):last_date  # daily means
glorys_kwargs = (dataset = GLORYSQuarterDegree(), dates = glorys_dates)

Tᵉ = FieldTimeSeries(Metadata(:temperature;  glorys_kwargs...), grid; time_indices_in_memory = 3)
Sᵉ = FieldTimeSeries(Metadata(:salinity;     glorys_kwargs...), grid; time_indices_in_memory = 3)
uᵉ = FieldTimeSeries(Metadata(:u_velocity;   glorys_kwargs...), grid; time_indices_in_memory = 3)
vᵉ = FieldTimeSeries(Metadata(:v_velocity;   glorys_kwargs...), grid; time_indices_in_memory = 3)
ηᵉ = FieldTimeSeries(Metadata(:free_surface; glorys_kwargs...), grid; time_indices_in_memory = 3)

#####
##### Atmosphere, radiation and rivers - usually `JRA55PrescribedAtmosphere(arch; dataset = MultiYearJRA55(), region, ...)`
##### reading the global JRA55 files, but this is JRA55 already cut to the region (`RegionalJRA55`, data/JRA55_region)
#####

jra55_kwargs = (dataset = RegionalJRA55(), start_date = first_date, end_date = last_date)

atmosphere = JRA55PrescribedAtmosphere(arch; jra55_kwargs..., time_indices_in_memory = 24)
radiation  = JRA55PrescribedRadiation(arch;  jra55_kwargs..., time_indices_in_memory = 24)

# Rivers: mouths searched within 5 grid cells and spread over 1.2° (`maximum_search_radius` is converted from cells to
# degrees as if the grid were global, hence the rescaling)
search_radius = 5 * resolution / ((360 / Nx + 180 / Ny) / 2)
land = JRA55PrescribedLand(grid; jra55_kwargs..., spread_radius = 1.2, maximum_search_radius = search_radius)

#####
##### Bottom drag: quadratic, with a log-layer drag coefficient for the bottom cell (src/boundary_functions.jl);
##### κ is the von Kármán constant and z₀ = 1 cm the roughness length
#####

drag_parameters = (κ = 0.41, z₀ = 0.01)

#####
##### Open boundaries (west, south and north; the east edge is the coast), as CWorthy's UCLA ROMS
#####
##### T, S and the tangential velocities are carried through the boundary by the normal velocity, relaxing towards
##### GLORYS on inflow (`TracerReservoir`); the normal velocities radiate with no outflow nudging and relax towards
##### GLORYS on inflow over Δx / ubind; the depth-integrated transports use Flather's condition with GLORYS transport
##### and sea surface height outside.
#####

using Oceananigans.BoundaryConditions: ObliqueRadiation, TracerReservoir

# The exterior values (`x_boundary`, `y_boundary`), Flather (`x_flather`, `y_flather`) and drag functions are in
# src/boundary_functions.jl; without a `time` parameter the Flather functions read the clock's time
Δy = grid.radius * deg2rad(resolution)              # ≈ 3.1 km, normal to the south and north edges
Δx = Δy * cosd((latitude[1] + latitude[2]) / 2)     # ≈ 2.4 km at 39°N, normal to the west edge
ubind = 0.1                                         # m s⁻¹, ROMS' "binding velocity"

x_reservoir = TracerReservoir(inflow_length_scale = Δx, outflow_length_scale = Δx)
y_reservoir = TracerReservoir(inflow_length_scale = Δy, outflow_length_scale = Δy)
x_radiation = ObliqueRadiation(inflow_timescale = Δx / ubind, outflow_timescale = Inf)
y_radiation = ObliqueRadiation(inflow_timescale = Δy / ubind, outflow_timescale = Inf)

u_bcs = FieldBoundaryConditions(
    west  = NormalFlowBoundaryCondition(x_boundary; discrete_form = true, scheme = x_radiation, parameters = (ψ = uᵉ, i = 1)),
    south = ValueBoundaryCondition(y_boundary;      discrete_form = true, scheme = y_reservoir, parameters = (ψ = uᵉ, j = 1)),
    north = ValueBoundaryCondition(y_boundary;      discrete_form = true, scheme = y_reservoir, parameters = (ψ = uᵉ, j = Ny)),
    bottom = IMEXFluxBoundaryCondition(0, u_bottom_drag; discrete_form = true, parameters = drag_parameters),
    immersed = ImmersedBoundaryCondition(bottom = IMEXFluxBoundaryCondition(0, u_immersed_drag; discrete_form = true, parameters = drag_parameters)))

v_bcs = FieldBoundaryConditions(
    west  = ValueBoundaryCondition(x_boundary;      discrete_form = true, scheme = x_reservoir, parameters = (ψ = vᵉ, i = 1)),
    south = NormalFlowBoundaryCondition(y_boundary; discrete_form = true, scheme = y_radiation, parameters = (ψ = vᵉ, j = 1)),
    north = NormalFlowBoundaryCondition(y_boundary; discrete_form = true, scheme = y_radiation, parameters = (ψ = vᵉ, j = Ny)),
    bottom = IMEXFluxBoundaryCondition(0, v_bottom_drag; discrete_form = true, parameters = drag_parameters),
    immersed = ImmersedBoundaryCondition(bottom = IMEXFluxBoundaryCondition(0, v_immersed_drag; discrete_form = true, parameters = drag_parameters)))

T_bcs = FieldBoundaryConditions(
    west  = ValueBoundaryCondition(x_boundary; discrete_form = true, scheme = x_reservoir, parameters = (ψ = Tᵉ, i = 1)),
    south = ValueBoundaryCondition(y_boundary; discrete_form = true, scheme = y_reservoir, parameters = (ψ = Tᵉ, j = 1)),
    north = ValueBoundaryCondition(y_boundary; discrete_form = true, scheme = y_reservoir, parameters = (ψ = Tᵉ, j = Ny)))

S_bcs = FieldBoundaryConditions(
    west  = ValueBoundaryCondition(x_boundary; discrete_form = true, scheme = x_reservoir, parameters = (ψ = Sᵉ, i = 1)),
    south = ValueBoundaryCondition(y_boundary; discrete_form = true, scheme = y_reservoir, parameters = (ψ = Sᵉ, j = 1)),
    north = ValueBoundaryCondition(y_boundary; discrete_form = true, scheme = y_reservoir, parameters = (ψ = Sᵉ, j = Ny)))

U_bcs = FieldBoundaryConditions(
    west  = GravityWaveRadiationBoundaryCondition(x_flather; discrete_form = true, parameters = (u = uᵉ, η = ηᵉ, i = 1, face = 1)))

V_bcs = FieldBoundaryConditions(
    south = GravityWaveRadiationBoundaryCondition(y_flather; discrete_form = true, parameters = (v = vᵉ, η = ηᵉ, j = 1, face = 1)),
    north = GravityWaveRadiationBoundaryCondition(y_flather; discrete_form = true, parameters = (v = vᵉ, η = ηᵉ, j = Ny, face = Ny + 1)))

boundary_conditions = (u = u_bcs, v = v_bcs, T = T_bcs, S = S_bcs, U = U_bcs, V = V_bcs)

#####
##### Sponge: viscosity and diffusivity rising linearly to Δx/10 m² s⁻¹ (≈ 240 m² s⁻¹) over the 16 cells (0.44°)
##### inside each open boundary (`sponge_coefficient`, src/boundary_functions.jl)
#####

sponge = HorizontalScalarDiffusivity(ν = sponge_coefficient, κ = sponge_coefficient, discrete_form = true,
                                     parameters = (west = longitude[1], south = latitude[1], north = latitude[2],
                                                   Δ = resolution, width = 16, ν = Δx / 10))

#####
##### Vertical mixing: CATKE
#####

using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEMixingLength

catke = CATKEVerticalDiffusivity(; mixing_length = CATKEMixingLength(Cᵇ = 0.01),
                                   maximum_viscosity = 3, maximum_tracer_diffusivity = 3,
                                   maximum_tke_diffusivity = 3, negative_tke_damping_time_scale = 10)

#####
##### Surface salinity restoring to the monthly WOA climatology, piston velocity 1/6 m day⁻¹ over the top cell
#####

using NumericalEarth.DataWrangling: SurfaceFluxRestoring
using WorldOceanAtlasTools # loads the NumericalEarth WOA extension

# Global WOA (not cut to the region, as ../CaliforniaCurrent.jl does): a region would overwrite the global inpainted
# files in data/WOA/monthly that the 03_global scripts read
piston_velocity = 1 / 6   # m day⁻¹
S_restoring = DatasetRestoring(Metadata(:salinity; dataset = WOAMonthly()), arch;
                               rate = piston_velocity / (minimum_zspacing(underlying_grid) * days),   # Δz of the top cell
                               time_indices_in_memory = 12)

#####
##### Ocean
#####

using Oceananigans.Advection: EnstrophyConserving

ocean = ocean_simulation(grid; Δt,
                         momentum_advection = WENOVectorInvariant(vorticity_order = 9, time_discretization = AdaptiveVerticallyImplicitDiscretization(cfl = 0.5)),
                         tracer_advection = WENO(order = 7, time_discretization = AdaptiveVerticallyImplicitDiscretization(cfl = 0.5)),
                         coriolis = HydrostaticSphericalCoriolis(scheme = EnstrophyConserving()),
                         closure = (catke, sponge),
                         free_surface = SplitExplicitFreeSurface(grid; cfl = 0.7, fixed_Δt = Δt),
                         boundary_conditions,
                         additional_surface_fluxes = (; S = SurfaceFluxRestoring(S_restoring)),
                         river_routing = land.river_routing)

# Initial state: GLORYS on `first_date`
set!(ocean.model; T = Tᵉ[1], S = Sᵉ[1], u = uᵉ[1], v = vᵉ[1], η = ηᵉ[1])

#####
##### Coupled model and simulation
#####

coupled = OceanOnlyModel(ocean; atmosphere, radiation, land)
# Stops 15 minutes before a 24-hour job limit and checkpoints then (`checkpoint_at_end` below)
simulation = Simulation(coupled; Δt = initial_Δt, stop_time = (last_date + Day(1) - first_date) / Millisecond(1000),
                        wall_time_limit = 23.75hours)

# Ramp the time step up from `initial_Δt`, doubling every 3 simulated days until it reaches `Δt` (≈ 10 days),
# while the interpolated GLORYS initial state adjusts. A run picked up from a checkpoint is past the ramp.
ramp_time_step!(sim) = sim.Δt = min(Δt, initial_Δt * 2^(sim.model.clock.time / 3days))
add_callback!(simulation, ramp_time_step!, IterationInterval(1))

wall_time = Ref(time_ns())
function progress(sim)
    t = sim.model.clock.time
    o = sim.model.ocean.model
    wall = 1e-9 * (time_ns() - wall_time[])
    speed = 100 * sim.Δt / wall   # model seconds per wall second over the last 100 iterations
    @info @sprintf("%s (%.1f%% of the run), iteration %d, Δt = %s, %.0f model days per wall day, about %s left; max|u| = %.2f m/s, SST ∈ (%.1f, %.1f) °C",
                   Date(first_date + Millisecond(round(Int, 1000t))), 100t / sim.stop_time, sim.model.clock.iteration,
                   prettytime(sim.Δt), speed, prettytime((sim.stop_time - t) / speed),
                   maximum(abs, o.velocities.u), extrema(view(o.tracers.T, :, :, Nz))...)
    wall_time[] = time_ns()
end
add_callback!(simulation, progress, IterationInterval(100))

#####
##### Output - daily means of surface temperature and speed (≈ 5 GB for 2010–2011), and hourly snapshots through
##### June 2011 (≈ 5 GB) for inertial oscillations, the diurnal cycle and upwelling fronts and filaments
#####

o = ocean.model
u, v = o.velocities.u, o.velocities.v
surface_speed = @at (Center, Center, Center) sqrt(u^2 + v^2)

# One file per year (`surface_part1.jld2` is 2010, `_part2.jld2` 2011), split at 01:00 on 1 January so that
# 31 December's daily mean, written at 00:00 on 1 January, stays in its year's file
new_years = [(DateTime(y, 1, 1) - first_date) / Millisecond(1000) + 1hour for y in year(first_date)+1:year(last_date)]
simulation.output_writers[:surface] = JLD2Writer(o, (; T = o.tracers.T, speed = surface_speed); dir = output_directory,
                                                 filename = "surface", indices = (:, :, Nz),
                                                 schedule = AveragedTimeInterval(1day), including = [],
                                                 file_splitting = SpecifiedTimes(new_years))

june = (DateTime(2011, 6, 1) - first_date) / Millisecond(1000)
simulation.output_writers[:june_hourly] = JLD2Writer(o, (; T = o.tracers.T, speed = surface_speed); dir = output_directory,
                                                     filename = "surface_hourly_june_2011", indices = (:, :, Nz),
                                                     schedule = SpecifiedTimes([june + n * hour for n in 0:30*24]),
                                                     including = [])

# Checkpoints to carry on in a new job: keep the latest (≈ 1.5 GB) and delete the older ones
simulation.output_writers[:checkpointer] = Checkpointer(coupled; dir = output_directory, prefix = "checkpoint",
                                                        schedule = TimeInterval(30days), cleanup = true)

pickup = isdir(output_directory) && any(startswith("checkpoint"), readdir(output_directory))
run!(simulation; pickup, checkpoint_at_end = true)

# If the run stopped on a NaN, report where it is in each field (cell centres; k = Nz is the surface)
for (name, field) in pairs((; T = o.tracers.T, S = o.tracers.S, u = o.velocities.u, v = o.velocities.v, w = o.velocities.w,
                              η = o.free_surface.displacement))
    nans = findall(isnan, Array(interior(field)))
    isempty(nans) && continue
    i, j, k = Tuple(first(nans))
    is, js, ks = (extrema(I[d] for I in nans) for d in 1:3)
    @info @sprintf("%s: %d NaNs in i ∈ %s, j ∈ %s, k ∈ %s; first at (i, j, k) = (%d, %d, %d) ≈ (%.2f°E, %.2f°N), bottom at %.0f m",
                   name, length(nans), is, js, ks, i, j, k, longitude[1] + (i - 1/2) * resolution,
                   latitude[1] + (j - 1/2) * resolution, CUDA.@allowscalar(grid.immersed_boundary.bottom_height[i, j, 1]))
end
