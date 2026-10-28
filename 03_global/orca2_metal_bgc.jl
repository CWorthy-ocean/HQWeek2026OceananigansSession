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
using OceanBioME

const FT = Float32                     # Metal has no Float64
Oceananigans.defaults.FloatType = FT

#####
##### orca2_metal.jl plus OceanBioME's `MITgcmDIC` biogeochemistry (PO₄, Fe, DOP, O₂ and the carbonate system DIC/Alk,
##### with implicit productivity and sinking), as 02_california_current/metal_alkalinity.jl with one carbonate system.
##### Air–sea CO₂ and O₂ exchange are NumericalEarth's (src/ocean_only_gas_exchange.jl for an ocean without sea ice),
##### with repeat-year JRA55 winds and the 1990 Mauna Loa CO₂ cycled every year
#####

#####
##### Settings
#####

arch       = GPU(Metal.MetalBackend())
Nz         = 30
depth      = 5500              # m
Δt         = 90minutes
start_date = DateTime(1990, 1, 1)
run_length = 730days

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
set!(νh, KernelFunctionOperation{Center, Center, Center}(νhb, grid, convert(FT, 40days)))
horizontal_viscosity = HorizontalScalarBiharmonicDiffusivity(ν = νh)

# Background diapycnal mixing from internal tides, Henyey et al. style latitude dependence (m² s⁻¹)
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
##### Biogeochemistry: MITgcmDIC
#####

# Surface PAR: 0.43 × the shortwave entering the ocean, which NumericalEarth copies in every step, attenuated by the
# chlorophyll of the ocean's two-colour shortwave penetration (NumericalEarth's default, 0.147 mg m⁻³ ≈ 0.0435 m⁻¹)
radiative_forcing = TwoColorRadiation(grid)
surface_PAR = PARFromShortwave(grid; photosynthetic_fraction_of_shortwave = convert(FT, 0.43))
light_attenuation = PrescribedAttenuationPAR(grid, surface_PAR;
                                             attenuation = PrescribedChlorophyllAttenuation(radiative_forcing),
                                             attenuation_discrete_form = true)

# MITgcm-style implicit productivity: 9 mmol P m⁻³ per 360 days at most, light half-saturation 30 W m⁻², C:P = 117
plankton = ImplicitProductivity(FT; maximum_community_productivity = 9.0 / 360days,
                                    light_half_saturation = 30.0,
                                    carbon_ratio = 117.0)

# POP sinks implicitly, redistributed down each column every step with an exponential profile; what reaches the floor
# is remineralised there (closed bottom)
biogeochemistry = MITgcmDIC(grid; plankton, light_attenuation,
                            open_bottom = false,
                            implicit_sinking = true)

# Soluble iron deposition: the annual mean of the OAEMIP climatology's monthly values (mmol m⁻² s⁻¹, negative into the
# ocean), as an additional surface flux on Fe
iron_dates = NumericalEarth.DataWrangling.all_dates(BGCInitClimatology(), :soluble_iron_deposition)
iron_deposition = Field{Center, Center, Nothing}(grid)
for date in iron_dates
    monthly = set!(Field{Center, Center, Nothing}(grid), Metadatum(:soluble_iron_deposition; dataset = BGCInitClimatology(), date))
    interior(iron_deposition) .+= interior(monthly) ./ length(iron_dates)
end

ocean = ocean_simulation(grid; Δt, momentum_advection, tracer_advection, free_surface, closure,
                         radiative_forcing, biogeochemistry,
                         additional_surface_fluxes = (; S = SurfaceFluxRestoring(S_restoring), Fe = iron_deposition))

#####
##### Initial condition: as NumericalOcean, annual WOA converted to TEOS-10 (Θ, Sᴬ) with January WOA blended in above
##### 1000 m (tapering to annual at the monthly climatology's ≈ 1500 m reach), see src/woa_teos10.jl
#####

T₀ = CenterField(grid)
S₀ = CenterField(grid)
set!(T₀, Metadatum(:temperature; dataset = WOAAnnual()))
set!(S₀, Metadatum(:salinity;    dataset = WOAAnnual()))
woa_to_teos10!(T₀, S₀)
blend_monthly_initial_condition!(T₀, S₀, start_date, 1000) # blend depth (m)

# BGC: the OAEMIP BGC climatology (data/bgc_init.nc) for January, in mmol m⁻³. DOP is slightly negative at depth in the
# file, so it is clipped at zero
bgc_init_date = first(NumericalEarth.DataWrangling.all_dates(BGCInitClimatology(), :phosphate))
bgc_init(name) = set!(CenterField(grid), Metadatum(name; dataset = BGCInitClimatology(), date = bgc_init_date))

DOP₀ = bgc_init(:dissolved_organic_phosphorus)
interior(DOP₀) .= max.(interior(DOP₀), 0)
bgc₀ = (PO₄ = bgc_init(:phosphate), Fe = bgc_init(:dissolved_iron), DOP = DOP₀,
        DIC = bgc_init(:dissolved_inorganic_carbon), Alk = bgc_init(:alkalinity), O₂ = bgc_init(:dissolved_oxygen))

set!(ocean.model; T = T₀, S = S₀, bgc₀...)

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
    @info @sprintf("    surface PO₄ ∈ (%.3f, %.3f), DIC ∈ (%.0f, %.0f), Alk ∈ (%.0f, %.0f), O₂ ∈ (%.0f, %.0f) mmol m⁻³",
                   extrema(view(c.PO₄, :, :, Nz))..., extrema(view(c.DIC, :, :, Nz))...,
                   extrema(view(c.Alk, :, :, Nz))..., extrema(view(c.O₂, :, :, Nz))...)

    wall_time[] = time_ns()
    return nothing
end

add_callback!(simulation, progress, TimeInterval(10days))

#####
##### Output - snapshots (`TimeInterval`), AveragedTimeInterval doesn't work on Metal
#####

ocean_outputs = merge(ocean.model.tracers, ocean.model.velocities)

ocean.output_writers[:surface] = JLD2Writer(ocean.model, ocean_outputs;
                                            schedule = TimeInterval(1days),
                                            filename = joinpath(@__DIR__, "output", "orca2_metal_bgc_surface"),
                                            indices = (:, :, grid.Nz),
                                            overwrite_files = true)

# Air–sea CO₂ and O₂ fluxes (mmol m⁻² s⁻¹, positive out of the ocean)
J = coupled_model.interfaces.net_fluxes.ocean
ocean.output_writers[:fluxes] = JLD2Writer(ocean.model, (; DIC_flux = J.DIC, O₂_flux = J.O₂);
                                           schedule = TimeInterval(1days),
                                           filename = joinpath(@__DIR__, "output", "orca2_metal_bgc_fluxes"),
                                           overwrite_files = true)

run!(simulation)

# Final state for the alkalinity-release runs (orca2_metal_oae.jl), see src/restart_state.jl
save_restart_state(joinpath(@__DIR__, "output", "orca2_metal_bgc_restart.jld2"), ocean)
