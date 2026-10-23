# Before `using NumericalEarth`: it reads the data directory when it loads
ENV["NUMERICALEARTH_DATA_DIRECTORY"] = "data"
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"            # WOA downloads otherwise wait for a [y/n] on stdin

using Dates
using Metal
using NumericalEarth
using NumericalEarth.DataWrangling: SurfaceFluxRestoring
using Oceananigans
using Oceananigans.Units
using HQWeek2026OceananigansWorkshop # hacks and shinanigans to get Metal to work for this setup, and ORCA2Grid
using Printf
using WorldOceanAtlasTools # loads the NumericalEarth WOA extension

Oceananigans.defaults.FloatType = Float32   # Metal has no Float64

#####
##### Ocean-only (no sea ice: temperature is clipped at the freezing point) global simulation on ORCA2,
##### forced by repeat-year JRA55 (RYF 1990--1991, run 03_global/download_jra55.jl first)
##### Physics follows experiments/orca2.jl in NumericalEarth PR #153
#####

#####
##### Settings
#####

arch       = GPU(Metal.MetalBackend())
Nz         = 30
depth      = 5500              # m
Δt         = 90minutes
start_date = DateTime(1990, 1, 1)
run_length = 365days

#####
##### Grid - see src/orca2.jl
#####

z = ExponentialDiscretization(Nz, -depth, 0; scale = 1240)
grid = ORCA2Grid(arch; z, major_basins = 1)

#####
##### Closures
#####

# Biharmonic viscosity with a damping time of 40 days at the grid scale: ν = Az² / λ (m⁴ s⁻¹)
@inline νhb(i, j, k, grid, λ) = Oceananigans.Operators.Az(i, j, k, grid, Center(), Center(), Center())^2 / λ
νh = CenterField(grid)
set!(νh, KernelFunctionOperation{Center, Center, Center}(νhb, grid, Float32(40days)))
horizontal_viscosity = HorizontalScalarBiharmonicDiffusivity(ν = νh)

# Background diapycnal mixing from internal tides, Henyey et al. style latitude dependence (m² s⁻¹)
@inline henyey_diffusivity(λ, φ, z) = max(2e-6, 3e-5 * abs(sind(φ)))
κ_henyey = CenterField(grid)
set!(κ_henyey, henyey_diffusivity)
vertical_diffusivity = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); κ = κ_henyey, ν = 3e-5)

# The default isopycnal tensor and slope limiter are Float64 whatever the FloatType, which Metal rejects
using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity, SmallSlopeIsopycnalTensor, FluxTapering
eddy_closure = IsopycnalSkewSymmetricDiffusivity(κ_skew = 500, κ_symmetric = 500, # m² s⁻¹
                                                 isopycnal_tensor = SmallSlopeIsopycnalTensor(Float32),
                                                 slope_limiter = FluxTapering(1f-2))

vertical_mixing = CATKEVerticalDiffusivity(VerticallyImplicitTimeDiscretization();
                                           maximum_tracer_diffusivity = 1,
                                           maximum_tke_diffusivity = 1,
                                           maximum_viscosity = 1)

closure = (eddy_closure, vertical_mixing, horizontal_viscosity, vertical_diffusivity)

free_surface       = SplitExplicitFreeSurface(grid; substeps = 60)
momentum_advection = WENOVectorInvariant(order = 5)
tracer_advection   = WENO(order = 5)

#####
##### Surface salinity restoring to monthly WOA (there is no sea ice, so this stands in for brine rejection and melt)
#####

# As in GlobalOcean (OMIPSimulations.salinity_surface_restoring): the series is built on the architecture, not the grid,
# so it stays on WOA's own grid (which only reaches 1525 m) and is interpolated at the surface as the model runs.
# The rate is a piston velocity over the top cell: 0.254 m day⁻¹ / Δz_top (s⁻¹)
piston_velocity = 0.254 # m day⁻¹
Δz_top = Oceananigans.Operators.Δzᶜᶜᶜ(1, 1, Nz, Oceananigans.Architectures.on_architecture(CPU(), grid))
salinity = Metadata(:salinity; dataset = WOAMonthly())
S_restoring = DatasetRestoring(salinity, arch; rate = piston_velocity / (Δz_top * days),
                               time_indices_in_memory = length(salinity))

ocean = ocean_simulation(grid; Δt, momentum_advection, tracer_advection, free_surface, closure,
                         additional_surface_fluxes = (; S = SurfaceFluxRestoring(S_restoring)))

#####
##### Initial condition: annual WOA climatology
#####

set!(ocean.model, T = Metadatum(:temperature; dataset = WOAAnnual()),
                  S = Metadatum(:salinity;    dataset = WOAAnnual()))

#####
##### Atmosphere and radiation: repeat-year JRA55
#####

jra55_dir = joinpath("data", "JRA55")
atmosphere = JRA55PrescribedAtmosphere(arch; dataset = RepeatYearJRA55(), dir = jra55_dir)
radiation  = JRA55PrescribedRadiation(arch; dataset = RepeatYearJRA55(), dir = jra55_dir)

#####
##### Coupled model: ocean only
#####

coupled_model = OceanSeaIceModel(ocean, FreezingLimitedOceanTemperature(); atmosphere, radiation)
simulation = Simulation(coupled_model; Δt, stop_time = run_length)

wall_time = Ref(time_ns())

function progress(sim)
    u, v, w = sim.model.ocean.model.velocities
    T = sim.model.ocean.model.tracers.T
    step_time = 1e-9 * (time_ns() - wall_time[])

    @info @sprintf("time: %s, iter: %d, max|u|: (%.1e, %.1e, %.1e) m s⁻¹, extrema(T): (%.1f, %.1f) °C, wall time: %s",
                   prettytime(sim), iteration(sim), maximum(abs, u), maximum(abs, v), maximum(abs, w),
                   minimum(T), maximum(T), prettytime(step_time))

    wall_time[] = time_ns()
    return nothing
end

add_callback!(simulation, progress, TimeInterval(10days))

#####
##### Output - snapshots (`TimeInterval`), see the note at the top of src/metal_hacks.jl
#####

ocean_outputs = merge(ocean.model.tracers, ocean.model.velocities)

ocean.output_writers[:surface] = JLD2Writer(ocean.model, ocean_outputs;
                                            schedule = TimeInterval(5days),
                                            filename = "orca2_metal_surface",
                                            indices = (:, :, grid.Nz),
                                            overwrite_files = true)

run!(simulation)
