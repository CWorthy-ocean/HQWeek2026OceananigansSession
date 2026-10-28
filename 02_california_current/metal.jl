# Before `using NumericalEarth`: it reads the data directory when it loads
ENV["NUMERICALEARTH_DATA_DIRECTORY"] = "data"

using Dates
using Metal
using NumericalEarth
using Oceananigans
using Oceananigans.Units # just defines stuff like day = days = 24*60*60
using HQWeek2026OceananigansSession # hacks and shinanigans to get Metal to work for this setup
using OceanBioME

Oceananigans.defaults.FloatType = Float32   # Metal has no Float64

#####
##### Warning: I included GLORYS initial conditions for -134:-115.5E and 30:48N, and time series of the boundaries
##### if you want todo a different region we will need to download different data
##### I have included coarsened GLORYS data for the whole domain so you can use the hack below
##### But if you want full resolution for e.g. a really small domain I will bring on memory stick for NE interface
#####

#####
##### Settings - todo get rid of this
#####

arch       = GPU(Metal.MetalBackend())
resolution = 1/4               # degrees; for 1/2°, set Δt = 120minutes below
longitude  = (-134, -115.5)    # °E
latitude   = (30, 48)          # °N
Nz         = 42                
depth      = 5500              # m

start_date = DateTime(2010, 1, 1)
duration = 730days
end_date = start_date + Millisecond(round(Int, 1000duration))

#####
##### Defining the grid
#####
Nx = round(Int, (longitude[2] - longitude[1]) / resolution)
Ny = round(Int, (latitude[2] - latitude[1]) / resolution)
z = ExponentialDiscretization(Nz, -depth, 0; scale = 700, mutable = true)

underlying_grid = LatitudeLongitudeGrid(arch, Float32; 
                                        size = (Nx, Ny, Nz),
                                        halo = (7, 7, 7), 
                                        longitude, 
                                        latitude, 
                                        z)

bottom_height = regrid_bathymetry(underlying_grid; minimum_depth = 10, major_basins = 1, interpolation_passes = 5)
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height);
                            active_cells_map = true)

Δt         = 60minutes          # from the start (no ramp); tested stable at 1/4°. At 1/2°, 120minutes is stable
start_date = DateTime(2010, 1, 1)   # time 0 of the GLORYS boundary files; the clock starts on 1 January
run_length = 730days                # 2010 spin-up (both carbonate systems identical), then the release through 2011

#####
##### Getting all the boundary data - usually GLORYSDaily not GLORYSQuarterDegree but I downsampled to reduce storage
#####
glorys_dates  = start_date:Day(1):start_date + Day(round(Int, run_length / days) - 1)   # daily means
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

jra55_kwargs = (dataset = RegionalJRA55(), start_date, end_date)

atmosphere = JRA55PrescribedAtmosphere(arch; jra55_kwargs..., time_indices_in_memory = 24, prefetch = false)
radiation  = JRA55PrescribedRadiation(arch; jra55_kwargs..., time_indices_in_memory = 24)

# Rivers: mouths searched within 5 grid cells and spread over 1.2° (`maximum_search_radius` is converted from cells to
# degrees as if the grid were global, hence the rescaling)
search_radius = 5 * resolution / ((360 / Nx + 180 / Ny) / 2)
land = JRA55PrescribedLand(grid; jra55_kwargs..., spread_radius = 1.2, maximum_search_radius = search_radius)

#####
##### Bottom drag: quadratic, with a log-layer drag coefficient for the bottom cell (src/boundary_functions.jl);
##### κ is the von Kármán constant and z₀ = 1 cm the roughness length
#####

drag_parameters = (κ = 0.41f0, z₀ = 0.01f0)

#####
##### Open boundaries (west, south and north; the east edge is the coast)
#####
##### T, S and the tangential velocities are carried through the boundary by the normal velocity, relaxing towards
##### GLORYS on inflow (`TracerReservoir`); the normal velocities radiate and are nudged towards GLORYS (`ObliqueRadiation`);
##### the depth-integrated transports use Flather's condition with GLORYS transport and sea surface height outside.
#####

using Oceananigans.BoundaryConditions: ObliqueRadiation, TracerReservoir

# The exterior values (`x_boundary`, `y_boundary`), Flather (`x_flather`, `y_flather`) and drag functions are in
# src/boundary_functions.jl
# The barotropic substeps' clock time is Float64, which Metal kernels cannot use: the Flather functions
# (`x_flather`, `y_flather`, src/boundary_functions.jl) read the time from `barotropic_time` instead, set at the end of
# every step (see src/metal_hacks.jl)
barotropic_time = barotropic_time_array(arch)

Δy = grid.radius * deg2rad(resolution)              # grid spacing normal to the south and north edges
Δx = Δy * cosd((latitude[1] + latitude[2]) / 2)     # and normal to the west edge (at the middle latitude)

x_reservoir = TracerReservoir(inflow_length_scale = Δx, outflow_length_scale = Δx)
y_reservoir = TracerReservoir(inflow_length_scale = Δy, outflow_length_scale = Δy)
x_radiation = ObliqueRadiation(inflow_timescale = 0.5days, outflow_timescale = 2.5days)
y_radiation = ObliqueRadiation(inflow_timescale = 0.5days, outflow_timescale = 2.5days)

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
    west  = GravityWaveRadiationBoundaryCondition(x_flather; discrete_form = true, parameters = (u = uᵉ, η = ηᵉ, i = 1, face = 1, time = barotropic_time)))

V_bcs = FieldBoundaryConditions(
    south = GravityWaveRadiationBoundaryCondition(y_flather; discrete_form = true, parameters = (v = vᵉ, η = ηᵉ, j = 1, face = 1, time = barotropic_time)),
    north = GravityWaveRadiationBoundaryCondition(y_flather; discrete_form = true, parameters = (v = vᵉ, η = ηᵉ, j = Ny, face = Ny + 1, time = barotropic_time)))

boundary_conditions = (u = u_bcs, v = v_bcs, T = T_bcs, S = S_bcs, U = U_bcs, V = V_bcs)

#####
##### Sponge: viscosity and diffusivity rising linearly to Δx/10 m² s⁻¹ over the 1.6° inside each open boundary
##### (`sponge_coefficient`, src/boundary_functions.jl)
#####

sponge = HorizontalScalarDiffusivity(ν = sponge_coefficient, κ = sponge_coefficient, discrete_form = true,
                                     parameters = (west = Float32(longitude[1]), south = Float32(latitude[1]), north = Float32(latitude[2]),
                                                   Δ = Float32(resolution), width = Float32(round(1.6 / resolution)), ν = Float32(Δx / 10)))

#####
##### Vertical mixing: CATKE
#####

using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEMixingLength

catke = CATKEVerticalDiffusivity(; mixing_length = CATKEMixingLength(Cᵇ = 0.01),
                                   maximum_viscosity = 3, maximum_tracer_diffusivity = 3,
                                   maximum_tke_diffusivity = 3, negative_tke_damping_time_scale = 10)

#####
##### Surface salinity restoring to the annual WOA climatology, piston velocity 1/6 m day⁻¹ over the top cell
#####

using NumericalEarth.DataWrangling: SurfaceFluxRestoring, oceananigans_fieldnames
using WorldOceanAtlasTools # loads the NumericalEarth WOA extension

# `DatasetRestoring` restores towards a time series, so the annual climatology is a series of one time, on the model grid
using Oceananigans.OutputReaders: Clamp

S★ = FieldTimeSeries{Center, Center, Center}(grid, [0f0]; time_indexing = Clamp())
set!(S★[1], Metadatum(:salinity; dataset = WOAAnnual()))

Δz_top = minimum_zspacing(underlying_grid)
S_restoring = DatasetRestoring(S★, nothing, 1, oceananigans_fieldnames[:salinity], Float32(1 / 6 / (Δz_top * days)))

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

# Initial state: GLORYS on `start_date`
set!(ocean.model; T = Tᵉ[1], S = Sᵉ[1], u = uᵉ[1], v = vᵉ[1], η = ηᵉ[1])

#####
##### Coupled model and simulation
#####

coupled = OceanOnlyModel(ocean; atmosphere, radiation, land)
simulation = Simulation(coupled; Δt, stop_time = run_length)

# The Flather boundary data's time (see `barotropic_time` above)
add_callback!(simulation, sim -> set_barotropic_time!(barotropic_time, sim), IterationInterval(1))

wall_time = Ref(time_ns())
function progress(sim)
    o = sim.model.ocean.model
    elapsed = 1e-9 * (time_ns() - wall_time[])
    @info string(Date(start_date + Millisecond(round(Int, 1000 * sim.model.clock.time))), ", iteration ", iteration(sim),
                 ", Δt = ", prettytime(sim.Δt), ", max|u| = ", maximum(abs, o.velocities.u), " m s⁻¹, SST ∈ ",
                 extrema(view(o.tracers.T, :, :, Nz)), " °C, wall time ", prettytime(elapsed))
    wall_time[] = time_ns()
end
add_callback!(simulation, progress, IterationInterval(100))

#####
##### Output - daily surface snapshots (`TimeInterval`, see the note at the top of src/metal_hacks.jl)
#####

ocean = ocean.model
surface = (T = view(ocean.tracers.T, :, :, Nz), S = view(ocean.tracers.S, :, :, Nz),
           u = view(ocean.velocities.u, :, :, Nz), v = view(ocean.velocities.v, :, :, Nz),
           η = ocean.free_surface.displacement)
simulation.output_writers[:surface] = JLD2Writer(ocean, surface; dir = joinpath(@__DIR__, "output"), filename = "surface",
                                                 schedule = TimeInterval(1days), including = [], overwrite_files = true)

run!(simulation)
