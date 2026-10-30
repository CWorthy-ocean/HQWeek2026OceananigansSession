# Before `using NumericalEarth`: it reads the data directory when it loads
ENV["NUMERICALEARTH_DATA_DIRECTORY"] = "data"
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"            # WOA downloads otherwise wait for a [y/n] on stdin

using Dates
using Metal
using NumericalEarth
using NumericalEarth.DataWrangling: SurfaceFluxRestoring
using Oceananigans
using Oceananigans.Units
using HQWeek2026OceananigansSession # hacks and shinanigans to get Metal to work for this setup, and ORCA2Grid
using Printf
using WorldOceanAtlasTools # loads the NumericalEarth WOA extension
using OceanBioME

const FT = Float32                     # Metal has no Float64
Oceananigans.defaults.FloatType = FT

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
##### Grid 
#####

z = ExponentialDiscretization(Nz, -depth, 0; scale = 1240)
grid = ORCA2Grid(arch; z, major_basins = 1)

#####
##### Closures
#####

@inline νhb(i, j, k, grid, λ) = Oceananigans.Operators.Az(i, j, k, grid, Center(), Center(), Center())^2 / λ
νh = CenterField(grid)
set!(νh, KernelFunctionOperation{Center, Center, Center}(νhb, grid, convert(FT, 40days)))
horizontal_viscosity = HorizontalScalarBiharmonicDiffusivity(ν = νh)

@inline henyey_diffusivity(λ, φ, z) = max(2e-6, 3e-5 * abs(sind(φ)))
κ_henyey = CenterField(grid)
set!(κ_henyey, henyey_diffusivity)
vertical_diffusivity = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); κ = κ_henyey, ν = 3e-5)

# The default isopycnal tensor and slope limiter are Float64 whatever the FloatType, which Metal rejects
using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity, SmallSlopeIsopycnalTensor, FluxTapering
eddy_closure = IsopycnalSkewSymmetricDiffusivity(κ_skew = 500, κ_symmetric = 500, # m² s⁻¹
                                                 isopycnal_tensor = SmallSlopeIsopycnalTensor(FT), # these are defaults but are not automatically converted for now
                                                 slope_limiter = FluxTapering(convert(FT, 1e-2)))

vertical_mixing = CATKEVerticalDiffusivity(VerticallyImplicitTimeDiscretization();
                                           maximum_tracer_diffusivity = 1,
                                           maximum_tke_diffusivity = 1,
                                           maximum_viscosity = 1)

closure = (eddy_closure, vertical_mixing, horizontal_viscosity, vertical_diffusivity)

using Oceananigans.TimeSteppers: AdaptiveVerticallyImplicitDiscretization
time_discretization = AdaptiveVerticallyImplicitDiscretization(cfl = 0.5)

free_surface       = SplitExplicitFreeSurface(grid; substeps = 200)
momentum_advection = WENOVectorInvariant(order = 5; time_discretization)
tracer_advection   = WENO(order = 5; time_discretization, boundary_scheme = UpwindBiased(order = 1))

#####
##### Surface salinity restoring to monthly WOA (there is no sea ice, so this stands in for brine rejection and melt)
#####

piston_velocity = 0.254 # m day⁻¹
Δz_top = Oceananigans.Operators.Δzᶜᶜᶜ(1, 1, Nz, Oceananigans.Architectures.on_architecture(CPU(), grid))
salinity = Metadata(:salinity; dataset = WOAMonthly())
S_restoring = DatasetRestoring(salinity, arch; rate = convert(FT, piston_velocity / (Δz_top * days)),
                               time_indices_in_memory = length(salinity))
woa_salinity_fts_to_teos10!(S_restoring.field_time_series) # Practical → Absolute Salinity, as the model's S

#####
##### Release sites (°E, °N): 1 Tmol each over the year, Gaussian (σ = 100 km, cut off at 3σ) in the surface cells, see
##### src/alkalinity_release.jl
#####

# Two sites, as each one adds a DIC/Alk copy and the model compiles much more slowly the more tracers it has (5 sites
# took over an hour to compile on an M5). More can be added from the commented ones
release_sites = (California = (-127.5, 40.5),   # eastern boundary upwelling
                 Labrador   = (-49.5, 55.5))    # deep water formation
              #  Kuroshio   = (134.5, 26.5),    # south-east of Japan, feeding the western boundary current
              #  Tasmania   = (141.5, -44.5),   # Southern Ocean: strong winds, deep mixing
              #  Arabian    = (59.5, 13.5))     # western Arabian Sea: monsoonal circulation

Ncopies = length(release_sites) + 1   # copy 1 is the control
DIC_names = ntuple(n -> Symbol(:DIC, n), Ncopies)
Alk_names = ntuple(n -> Symbol(:Alk, n), Ncopies)

forcing = NamedTuple(Alk_names[n + 1] => Forcing(alkalinity_release; discrete_form = true,
                                                  parameters = alkalinity_release_tendency(grid, site; duration = run_length))
                     for (n, site) in enumerate(release_sites))

#####
##### Biogeochemistry: MITgcmDIC
#####

# Surface PAR: 0.43 × the shortwave entering the ocean, which NumericalEarth copies in every step, attenuated by the
# chlorophyll of the ocean's two-colour shortwave penetration (NumericalEarth's default, 0.147 mg m⁻³ ≈ 0.0435 m⁻¹)
radiative_forcing = TwoColorRadiation(grid)
surface_PAR = PARFromShortwave(grid; photosynthetic_fraction_of_shortwave = convert(FT, 0.43))
light_attenuation = PrescribedAttenuationPAR(grid, surface_PAR;
                                             attenuation = PrescribedChlorophyllAttenuation(radiative_forcing),
                                             attenuation_discrete_form = true)

plankton = ImplicitProductivity(FT; maximum_community_productivity = 9.0 / 360days,
                                    light_half_saturation = 30.0,
                                    carbon_ratio = 117.0)

biogeochemistry = MITgcmDIC(grid; plankton, light_attenuation,
                            inorganic_carbon = CarbonateSystem(Ncopies),
                            open_bottom = false,
                            implicit_sinking = true)

iron_dates = NumericalEarth.DataWrangling.all_dates(BGCInitClimatology(), :soluble_iron_deposition)
iron_deposition = Field{Center, Center, Nothing}(grid)
for date in iron_dates # metal nonsense
    monthly = set!(Field{Center, Center, Nothing}(grid), Metadatum(:soluble_iron_deposition; dataset = BGCInitClimatology(), date))
    interior(iron_deposition) .+= interior(monthly) ./ length(iron_dates)
end

ocean = ocean_simulation(grid; Δt, momentum_advection, tracer_advection, free_surface, closure,
                         radiative_forcing, biogeochemistry, forcing,
                         additional_surface_fluxes = (; S = SurfaceFluxRestoring(S_restoring), Fe = iron_deposition))

#####
##### Initial state: the end of orca2_metal_bgc.jl's spin-up, every copy's DIC and Alk the spun-up ones
#####

restart = joinpath(@__DIR__, "output", "orca2_metal_bgc_restart.jld2")
load_restart_state!(ocean, restart; replicates = (DIC = DIC_names, Alk = Alk_names))

#####
##### Atmosphere and radiation: repeat-year JRA55
#####

jra55_dir = joinpath("data", "JRA55")

# Atmospheric CO₂ (ppm): the 1990 Mauna Loa monthly means, cycled every 365 days with the repeat-year JRA55
pCO₂ = MaunaLoaCO₂(arch; FT, start_date, end_date = start_date + Year(1), period = 365days)
atmosphere = JRA55PrescribedAtmosphere(arch; dataset = RepeatYearJRA55(), dir = jra55_dir, prefetch = false,
                                       tracers = (; pCO₂))
radiation  = JRA55PrescribedRadiation(arch; dataset = RepeatYearJRA55(), dir = jra55_dir)

#####
##### Coupled model: ocean only (`OceanOnlyModel`: temperature clipped at the freezing point, and the gas exchange of
##### src/ocean_only_gas_exchange.jl)
#####

coupled_model = OceanOnlyModel(ocean; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt, stop_time = run_length)

wall_time = Ref(time_ns())

function progress(sim)
    u, v, w = sim.model.ocean.model.velocities
    T = sim.model.ocean.model.tracers.T
    step_time = 1e-9 * (time_ns() - wall_time[])

    @info @sprintf("time: %s, iter: %d, max|u|: (%.1e, %.1e, %.1e) m s⁻¹, extrema(T): (%.1f, %.1f) °C, wall time: %s",
                   prettytime(sim), iteration(sim), maximum(abs, u), maximum(abs, v), maximum(abs, w),
                   minimum(T), maximum(T), prettytime(step_time))

    c = sim.model.ocean.model.tracers
    Nz = size(c.T, 3)
    @info @sprintf("    surface PO₄ ∈ (%.3f, %.3f), DIC1 ∈ (%.0f, %.0f), Alk1 ∈ (%.0f, %.0f) mmol m⁻³",
                   extrema(view(c.PO₄, :, :, Nz))..., extrema(view(c.DIC1, :, :, Nz))..., extrema(view(c.Alk1, :, :, Nz))...)
    for (n, name) in enumerate(keys(release_sites))
        @info @sprintf("    %-10s max ΔAlk = %.1f, min ΔDIC = %.2f mmol m⁻³", name,
                       maximum(c[Alk_names[n + 1]] - c.Alk1), minimum(c[DIC_names[n + 1]] - c.DIC1))
    end

    wall_time[] = time_ns()
    return nothing
end

add_callback!(simulation, progress, TimeInterval(10days))

#####
##### Output - snapshots (`TimeInterval`), AveragedTimeInterval doesn't work on Metal
#####

c = ocean.model.tracers
J = coupled_model.interfaces.net_fluxes.ocean # mmol m⁻² s⁻¹, positive out of the ocean

# Each release's perturbation: Alk(n+1) - Alk1 and DIC(n+1) - DIC1 (mmol m⁻³), and its extra air–sea CO₂ flux
# J.DIC(n+1) - J.DIC1 (mmol m⁻² s⁻¹, negative = extra uptake)
ΔAlk = NamedTuple(Symbol(:ΔAlk_, name) => c[Alk_names[n + 1]] - c.Alk1 for (n, name) in enumerate(keys(release_sites)))
ΔDIC = NamedTuple(Symbol(:ΔDIC_, name) => c[DIC_names[n + 1]] - c.DIC1 for (n, name) in enumerate(keys(release_sites)))
ΔCO₂_flux = NamedTuple(Symbol(:ΔCO₂_flux_, name) => J[DIC_names[n + 1]] - J.DIC1 for (n, name) in enumerate(keys(release_sites)))

output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)   # not in the repository, so missing in a fresh clone

surface = merge((T = c.T, S = c.S, u = ocean.model.velocities.u, v = ocean.model.velocities.v,
                 PO₄ = c.PO₄, DIC1 = c.DIC1, Alk1 = c.Alk1, O₂ = c.O₂), ΔAlk, ΔDIC)

ocean.output_writers[:surface] = JLD2Writer(ocean.model, surface;
                                            schedule = TimeInterval(1days),
                                            filename = joinpath(output_dir, "orca2_metal_oae_surface"),
                                            indices = (:, :, grid.Nz),
                                            overwrite_files = true)

ocean.output_writers[:fluxes] = JLD2Writer(ocean.model, merge((; DIC1_flux = J.DIC1), ΔCO₂_flux);
                                           schedule = TimeInterval(1days),
                                           filename = joinpath(output_dir, "orca2_metal_oae_fluxes"),
                                           overwrite_files = true)

# Full-depth perturbations, monthly snapshots (how the alkalinity spreads and is subducted)
ocean.output_writers[:interior] = JLD2Writer(ocean.model, merge(ΔAlk, ΔDIC);
                                             schedule = TimeInterval(365days / 12),
                                             filename = joinpath(output_dir, "orca2_metal_oae_interior"),
                                             overwrite_files = true)

run!(simulation)

