# # Workarounds for running NumericalEarth / Oceananigans on Apple GPUs (Metal, Float32)
#
# Include after setting `Oceananigans.defaults.FloatType = Float32` and before building anything:
#
#     include("metal_workarounds.jl")
#
# Each item works around a limitation found in Metal.jl 1.11.1, Oceananigans 0.113.6 and NumericalEarth 0.8.x (see
# the note on each); remove it once the package handles the case. The coupled (atmosphere) runs also need the
# NumericalEarth fixes for Float32 clock times, roughness lengths, humidity and radiation (metal_dev/ points at a
# checkout with them). Not a workaround but worth knowing: Oceananigans 0.113's `AveragedTimeInterval` broadcasts the
# Float64 clock time into Float32 arrays, which Metal rejects, so write snapshots (`TimeInterval`) instead.
#
# 1. `Base.unsafe_view` for Metal device vectors with any unit range (type piracy): views of views in kernels.
# 2. `Oceananigans.Fields.middle_point` with integer division on Metal: binary search on array coordinates.
# 3. `on_metal(series, grid)`: a FieldTimeSeries on a Metal grid with Float32 times.
# 4. `barotropic_time_array(arch)` and `set_barotropic_time!(barotropic_time, sim)`: a Float32 clock time for
#    functions called from the barotropic substeps (e.g. Flather boundary conditions).
# 5. `surface_dataset_restoring(metadata, cpu_grid, grid; rate)`: DatasetRestoring of the top level only.
# 7. NetCDF data with missing values moved or copied to Metal: missing → NaN on the host first.
# 8. `cpu_interpolating_time_indices` on Metal: a Float32 time-interpolation weight for prescribed atmosphere/radiation.
# 9. Float32 clocks, roughness lengths, radiation surface properties and humidity: the atmosphere-ocean flux solver.
# 10. `compute_mask` on Metal: Float32 bounds and mask value when a dataset is inpainted on the GPU.
# 11. `on_architecture` of a `Vector{Float64}` to Metal: copied as Float32 (e.g. the uneven times of a monthly series).
# 12. Float32 ZWENO weights that cannot overflow: τ / (β + ϵ) capped before squaring.
# 13. OceanBioME air–sea O₂ and CO₂ exchange built in the grid's float type (OceanBioME defaults to Float64).
# 12. Float32 carbon chemistry: the seawater density and the boron, sulfate and fluoride totals in the CO₂ gas exchange.
# 14. `Time{Float64}` adapted to a Metal kernel: Float32 (e.g. the river discharge scatter).

using Metal: Metal, MtlDeviceVector, MtlDeviceArray
using Oceananigans
using Oceananigans.Architectures: on_architecture
using Oceananigans.Fields: location
using Oceananigans.OutputReaders: Cyclical
using NumericalEarth.DataWrangling: oceananigans_fieldnames, DatasetRestoring

#####
##### 1. Views of views in Metal kernels
#####

# Metal.jl 1.11.1's `unsafe_view(::MtlDeviceVector, I)` (src/device/array.jl:150) reads `I[1].start` and `I[1].stop`,
# fields only a UnitRange has. A view of a device vector is a device vector, so viewing it again with `:` passes a
# `Base.Slice` and the kernel fails to compile ("call to jl_f_getfield"). Oceananigans' `λnodes`/`φnodes` do this on a
# LatitudeLongitudeGrid with array coordinates (e.g. NumericalEarth's JRA55 grid), in every interpolation from it. Same
# pointer arithmetic with `first` and `length`; see mwe_metal_nested_view.jl.
@inline function Base.unsafe_view(A::MtlDeviceVector{T}, I::AbstractUnitRange) where T
    return MtlDeviceArray(length(I), pointer(A, first(I)))
end

#####
##### 2. Binary search without Float64
#####

# Oceananigans 0.113's `middle_point(l, h) = unsafe_trunc(Int, (l + h) / 2)` (Fields/interpolate.jl:11), the midpoint
# of `index_binary_search`, goes through Float64. Same index with `÷`, in Metal device code only (as Oceananigans'
# Metal extension overrides `f32_safe_cbrt`).
Metal.@device_override @inline Oceananigans.Fields.middle_point(l, h) = (l + h) ÷ 2
Metal.@device_override @inline Oceananigans.Fields.find_λ_range(λ) = floor(λ / 360)

#####
##### 3. Time series with Float32 times
#####

"""
    on_metal(series, grid)

Copy of `series` on `grid` (the Metal grid holding the same points) with Float32 times. Oceananigans 0.113 turns
evenly spaced Float64 times into a TwicePrecision range, which its Metal extension does not convert to Float32, and
keeps uneven times as a `Vector{Float64}`, which Metal cannot hold; either way `ψ[i, j, k, Time(t)]` does not compile in
a Metal kernel. Float32 times are a plain range (or vector), which works. Equal to the CPU series to 1e-6 (relative) at
and between the times. Keeps the time indexing (with a Float32 period if cyclical) and any window (`indices`).
"""
function on_metal(series, grid)
    time_indexing = series.time_indexing isa Cyclical ? Cyclical(Float32(series.time_indexing.period)) : series.time_indexing
    gpu_series = FieldTimeSeries{location(series)...}(grid, Float32.(series.times); indices = series.indices, time_indexing)
    copyto!(parent(gpu_series.data), parent(series.data))
    return gpu_series
end

#####
##### 4. Clock time for functions called from the barotropic substeps
#####

# Oceananigans 0.113's split-explicit substeps call boundary-condition functions with a clock built on the host
# (`substep_clock`, step_split_explicit_free_surface.jl:153) whose Float64 time Metal kernels cannot use. Such functions
# read the time from a one-element Float32 array instead (`Time(@inbounds p.time[1])`), set at the end of every step:
#
#     barotropic_time = barotropic_time_array(arch)    # pass as a parameter of the boundary condition
#     add_callback!(simulation, sim -> set_barotropic_time!(barotropic_time, sim), IterationInterval(1))
#
# The substeps then use the time at the start of the step (at most Δt old).
barotropic_time_array(arch) = on_architecture(arch, zeros(Float32, 1))
set_barotropic_time!(barotropic_time, sim) = fill!(barotropic_time, Float32(sim.model.clock.time))

#####
##### 5. Restoring of the top level only
#####

"""
    surface_dataset_restoring(metadata, cpu_grid, grid; rate, variable = metadata.name)

NumericalEarth `DatasetRestoring` of the top level (k = Nz) of `grid` towards `metadata` (e.g. monthly WOA salinity),
for `SurfaceFluxRestoring`. The dataset is read on its own grid on the CPU, each time interpolated onto the top level of
`cpu_grid` (the CPU copy of `grid`), and the series copied to `grid` with Float32 times, windowed to k = Nz, so
`DatasetRestoring` reads it at (i, j, Nz) as usual. Built this way because `DatasetRestoring(metadata, grid)` puts the
dataset's uneven (Float64) times on the GPU, which Metal rejects, and because a dataset shallower than the grid (WOA
monthly reaches 1525 m) cannot be interpolated onto the full depth.
"""
function surface_dataset_restoring(metadata, cpu_grid, grid; rate, variable = metadata.name)
    Nz = size(grid, 3)
    native = FieldTimeSeries(metadata, CPU(); time_indices_in_memory = length(metadata), time_indexing = Cyclical())
    top = CenterField(cpu_grid; indices = (:, :, Nz:Nz))
    target = FieldTimeSeries{Center, Center, Center}(cpu_grid, native.times; indices = (:, :, Nz:Nz),
                                                     time_indexing = native.time_indexing)
    for n in eachindex(native.times)
        set!(top, native[n])
        set!(target[n], Array(interior(top)))
    end
    return DatasetRestoring(on_metal(target, grid), nothing, 1, oceananigans_fieldnames[variable], convert(eltype(grid), rate))
end

#####
##### 7. Dataset arrays with missing values
#####

# NetCDF variables with a fill value arrive as `Array{Union{Missing, Float32}}`, which NumericalEarth moves to the GPU
# as is (`architecture_ready`, DataWrangling/set_region_data.jl:211, and `set_jra55_slices!`, which also `copyto!`s
# each slice into a device buffer, JRA55/JRA55_field_time_series.jl:60) and NaN-converts in the kernel
# (`nan_convert_missing`). Metal.jl 1.11.1 cannot hold isbits-union arrays ("MtlArray does not yet support isbits-union
# arrays"), e.g. in every JRA55 FieldTimeSeries. Same missing → NaN, on the host first.
Oceananigans.Architectures.on_architecture(arch::GPU{Metal.MetalBackend}, a::Array{Union{Missing, T}}) where T <: AbstractFloat =
    on_architecture(arch, coalesce.(a, T(NaN)))

Base.copyto!(dst::Metal.MtlArray{T}, src::Array{Union{Missing, T}}) where T <: AbstractFloat =
    copyto!(dst, coalesce.(src, T(NaN)))

#####
##### 8. Float32 time-interpolation weight
#####

# NumericalEarth 0.8.1 builds the time interpolator for the prescribed atmosphere and radiation kernels on the host from
# the Float64 clock time (`cpu_interpolating_time_indices`, Atmospheres/interpolate_atmospheric_state.jl:57 and
# Radiations.jl), so its fractional index is Float64 and `ψ₂ * ñ + ψ₁ * (1 - ñ)` (field_time_series_indexing.jl:342)
# fails to compile on Metal ("unsupported use of double value"). Same interpolator, weight converted to Float32.
import Oceananigans.OutputReaders: cpu_interpolating_time_indices
using Oceananigans.OutputReaders: TimeInterpolator

function cpu_interpolating_time_indices(::GPU{Metal.MetalBackend}, times::AbstractVector, time_indexing, t)
    ti = TimeInterpolator(time_indexing, on_architecture(CPU(), times), t)
    return TimeInterpolator(convert(Float32, ti.fractional_index), ti.first_index, ti.second_index, ti.length)
end

#####
##### 9. Float32 atmosphere-ocean flux solver
#####

# NumericalEarth 0.8.1's flux solver (`_compute_atmosphere_ocean_interface_state!`) meets Float64 four ways on Metal,
# failing with "unsupported use of double value" and the type instability it causes; ported from the fixes in
# ~/Documents/NumericalEarth.jl (jsw/oceanbiome-ext-changes, uncommitted):
#
# - its launch sites pass the model `clock` (Float64 time) where they should pass `convert_time(grid, clock)`. Here
#   every Clock adapted to a Metal kernel gets Float32 times instead, as `convert_time` does (a no-op for Oceananigans'
#   own kernels, which already convert);
# - `MomentumRoughnessLength` and `ScalarRoughnessLength` keep the default `air_kinematic_viscosity = 1.5e-5` Float64
#   (built by keyword defaults in several places, e.g. similarity_theory_turbulent_fluxes.jl:174), and
#   `PrescribedRadiation`'s default surface albedos and emissivities are Float64 literals: both are converted as they
#   are adapted to a Metal kernel, as the clock is;
# - `default_ao_specific_humidity(ocean)` takes `eltype(ocean)`, which is `Any` for a `Simulation`, so 0.98 stays Float64.
using Oceananigans.TimeSteppers: Clock
using NumericalEarth.EarthSystemModels.InterfaceComputations: MomentumRoughnessLength, ScalarRoughnessLength,
                                                              ImpureSaturationSpecificHumidity,
                                                              AtmosphericThermodynamics
using NumericalEarth.Radiations: SurfaceRadiationProperties
import NumericalEarth.EarthSystemModels.InterfaceComputations: default_ao_specific_humidity
using Oceananigans.TimeSteppers: Adapt # not a direct dependency of this project

Adapt.adapt_structure(to::Metal.Adaptor, clock::Clock) =
    (time          = convert(Float32, clock.time),
     last_Δt       = convert(Float32, clock.last_Δt),
     last_stage_Δt = convert(Float32, clock.last_stage_Δt),
     iteration     = clock.iteration,
     stage         = clock.stage)

metal_float(x::Float64) = convert(Float32, x)
metal_float(x) = x

Adapt.adapt_structure(to::Metal.Adaptor, ℓ::MomentumRoughnessLength) =
    MomentumRoughnessLength(ℓ.gravitational_acceleration,
                            metal_float(Adapt.adapt(to, ℓ.air_kinematic_viscosity)),
                            Adapt.adapt(to, ℓ.wave_formulation),
                            ℓ.smooth_wall_parameter,
                            ℓ.maximum_roughness_length)

Adapt.adapt_structure(to::Metal.Adaptor, ℓ::ScalarRoughnessLength) =
    ScalarRoughnessLength(metal_float(Adapt.adapt(to, ℓ.air_kinematic_viscosity)),
                          Adapt.adapt(to, ℓ.reynolds_number_scaling_function),
                          ℓ.maximum_roughness_length)

Adapt.adapt_structure(to::Metal.Adaptor, s::SurfaceRadiationProperties) =
    SurfaceRadiationProperties(metal_float(Adapt.adapt(to, s.albedo)), metal_float(Adapt.adapt(to, s.emissivity)))

default_ao_specific_humidity(ocean::Simulation) =
    ImpureSaturationSpecificHumidity(AtmosphericThermodynamics.Liquid(), convert(eltype(ocean.model.grid), 0.98))

#####
##### 10. Inpainting masks without Float64
#####

# When a dataset has no inpainted cache, NumericalEarth 0.8.1 builds its land mask on the field's architecture with
# Float64 bounds and mask value (`compute_mask`, DataWrangling/metadata_field.jl:507), which `is_masked` compares with
# the Float32 data (metadata_field.jl:528): "unsupported use of double value" on Metal, e.g. inpainting monthly WOA
# salinity for `DatasetRestoring`. Same mask, with the values in the field's float type.
using Oceananigans.Architectures: architecture
using NumericalEarth.DataWrangling: Metadatum
import NumericalEarth.DataWrangling: compute_mask

function compute_mask(metadata::Metadatum, field::Field, mask_value, minimum_value, maximum_value)
    if architecture(field) isa GPU{Metal.MetalBackend}
        FT = eltype(field)
        mask_value isa Number && (mask_value = convert(FT, mask_value))
        minimum_value = convert(FT, minimum_value)
        maximum_value = convert(FT, maximum_value)
    end
    return invoke(compute_mask, Tuple{Metadatum, Any, Any, Any, Any},
                  metadata, field, mask_value, minimum_value, maximum_value)
end

#####
##### 11. Float64 vectors moved to Metal
#####

# A FieldTimeSeries with uneven times (e.g. monthly WOA salinity for `DatasetRestoring`) keeps them as a
# `Vector{Float64}`, which Oceananigans#452409b moves to the GPU with the field (field_time_series.jl:288); its Metal
# extension converts evenly spaced `StepRangeLen` times to Float32 but makes `MtlArray(::Vector{Float64})` of a vector,
# which Metal cannot allocate ("Metal does not support Float64 values"). Copied as Float32 instead.
Oceananigans.Architectures.on_architecture(arch::GPU{Metal.MetalBackend}, a::Vector{Float64}) =
    on_architecture(arch, convert(Vector{Float32}, a))

#####
##### 12. ZWENO weights without Float32 overflow
#####

# Oceananigans' ZWENO weights are α = C★ (1 + (τ / (β + ϵ))²) with ϵ = 1f-8 (Advection/weno_interpolants.jl:372). The
# divergence upwinding of `WENOVectorInvariant` reconstructs a transport divergence (m³ s⁻¹, ~10⁶ here), so β and τ are
# ~10¹²; where one sub-stencil is flat (β = 0, e.g. all-immersed faces next to a bottom step) τ / ϵ ~ 10²⁰ and its
# square overflows Float32 (max 3.4 × 10³⁸): α = Inf, the normalisation Inf × 0 = NaN, and the run blows up from one
# point (seen at 38°E, 65.7°S after 110 days). Same weights with τ / (β + ϵ) capped at 10¹⁵, where that stencil already
# takes all the weight. Fine in Float64, which never reaches the cap in practice.
using Oceananigans.Advection: WENO, C★, ϵ, advection_buffers
using Oceananigans.Utils: newton_div
import Oceananigans.Advection: zweno_alpha_loop

const max_zweno_ratio = convert(Float32, 1e15)

# Unrolled as Oceananigans' `metaprogrammed_zweno_alpha_loop` (a `Val` built from a loop index does not compile on GPUs)
for buffer in advection_buffers[2:end]
    α = [:(C★(scheme, Val($(s - 1))) * (1 + min(newton_div(WCT, τ, β[$s] + ϵ), max_zweno_ratio)^2)) for s in 1:buffer]
    @eval @inline zweno_alpha_loop(scheme::WENO{$buffer, Float32, WCT}, β, τ) where WCT = @inbounds ($(α...),)
end

#####
##### 13. Air–sea gas exchange in the grid's float type
#####

# NumericalEarth#247c263's OceanBioME extension builds the O₂ and CO₂ gas exchanges with OceanBioME's defaults, which
# are Float64 (`GarciaGordonOxygenSaturation()`, `CarbonChemistry()`, `FF{Float64}()`, ...; NumericalEarthOceanBioMEExt/
# gas_exchange.jl:52-80), so the gas exchange kernel fails on Metal ("unsupported use of double value", in the
# Schmidt number and solubility polynomials). Ported from the uncommitted fix in ~/Documents/NumericalEarth.jl-metal-dic
# (on the same commit), which builds every parameterisation in `eltype(grid)`: these are the extension's methods with
# `ocean::Simulation` (more specific than its untyped `ocean`, so nothing is overwritten).
using OceanBioME: OxygenGasExchangeBoundaryCondition, CarbonDioxideGasExchangeBoundaryCondition
using OceanBioME.Models.GasExchangeModel: GarciaGordonOxygenSaturation, CarbonDioxideAirConcentration,
                                          MolPerKgPerAtmToMMolPerCubicMPerMicroAtm
using OceanBioME.Models.CarbonChemistryModel: CarbonChemistry, FF
using OceanBioME.Models.NutrientsPlanktonDetritusModels.OxygenModels: Oxygen
using OceanBioME.Models.NutrientsPlanktonDetritusModels.InorganicCarbonModels: AbstractInorganicCarbon
import NumericalEarth.EarthSystemModels.InterfaceComputations: biogeochemical_interface

function biogeochemical_interface(exchanger, ocean::Simulation, ::Oxygen; gas_exchange_state, kwargs...)
    FT = eltype(ocean.model.grid)
    return (; O₂ = OxygenGasExchangeBoundaryCondition(FT;
                       wind_speed = gas_exchange_state.wind_speed,
                       air_concentration = GarciaGordonOxygenSaturation(FT; atmospheric_pressure = gas_exchange_state.atmospheric_pressure),
                       kwargs...).condition.func)
end

biogeochemical_interface(exchanger, ocean::Simulation, ::AbstractInorganicCarbon{1}; kwargs...) =
    (; DIC = float_type_carbon_dioxide_exchange(exchanger, :DIC, :Alk; grid = ocean.model.grid, kwargs...))

function biogeochemical_interface(exchanger, ocean::Simulation, ::AbstractInorganicCarbon{N}; kwargs...) where N
    names = ntuple(n -> Symbol(:DIC, n), Val(N))
    exchanges = ntuple(n -> float_type_carbon_dioxide_exchange(exchanger, Symbol(:DIC, n), Symbol(:Alk, n); grid = ocean.model.grid, kwargs...), Val(N))
    return NamedTuple{names}(exchanges)
end

function float_type_carbon_dioxide_exchange(exchanger, DIC, Alk; grid, gas_exchange_state, carbon_chemistry = CarbonChemistry(eltype(grid)), kwargs...)
    FT = eltype(grid)
    air_concentration = CarbonDioxideAirConcentration(FT; mole_fraction = exchanger.atmosphere.state.pCO₂,
                                                          atmospheric_pressure = gas_exchange_state.atmospheric_pressure,
                                                          solubility = MolPerKgPerAtmToMMolPerCubicMPerMicroAtm(FF{FT}(),
                                                                                                                carbon_chemistry.density_function))
    return CarbonDioxideGasExchangeBoundaryCondition(FT; grid,
                                                     wind_speed = gas_exchange_state.wind_speed,
                                                     air_concentration,
                                                     carbon_chemistry,
                                                     DIC, Alk,
                                                     kwargs...).condition.func
end

# some timestepping nonsense
# HACK: Oceananigans 0.113.6 converts the time step to the grid's float type before the kernels see it
# (`kernel_time_step`, TimeSteppers/split_runge_kutta.jl); the jsw/for-oaemip branch (0.113.4) passes the Float64 Δt
# on, which Metal rejects (in the vertically implicit diffusion solve). Convert it here instead, as 0.113.6 does.
# Metal models only (the third type parameter is the architecture), so Float64 runs on CUDA or the CPU keep their Δt.
import Oceananigans.Models.HydrostaticFreeSurfaceModels: rk_substep!, step_closure_prognostics!
const MetalHydrostaticModel = HydrostaticFreeSurfaceModel{<:Any, <:Any, <:GPU{<:Metal.MetalBackend}}
rk_substep!(model::MetalHydrostaticModel, Δτ::Float64, callbacks) = rk_substep!(model, Float32(Δτ), callbacks)
step_closure_prognostics!(model::MetalHydrostaticModel, Δt::Float64) = step_closure_prognostics!(model, Float32(Δt))


import Oceananigans.Architectures: on_architecture

function Oceananigans.Architectures.on_architecture(::GPU{Metal.MetalBackend}, s::StepRangeLen{FT, Float64, Float64}) where FT
    ref = Base.TwicePrecision{Float32}(s.ref)
    step = Base.TwicePrecision{Float32}(s.step)
    len = s.len
    offset = s.offset
    return StepRangeLen{FT}(ref, step, len, offset)
end

#####
##### 6. Evenly spaced Float64 times on Metal
#####

# Oceananigans' Metal extension converts a `StepRangeLen{FT, Float64, Float64}` to Float32 when it is moved to the GPU
# (OceananigansMetalExt.jl:29), but not the `StepRangeLen{Float64, TwicePrecision{Float64}, TwicePrecision{Float64}}`
# that `range` makes, which a FieldTimeSeries keeps as its times (field_time_series.jl:308): indexing such a series in
# a Metal kernel then fails ("unsupported use of double value", in `interpolating_time_indices`). The same range in
# Float32, as `on_metal` builds by hand.
function Oceananigans.Architectures.on_architecture(::GPU{<:Metal.MetalBackend},
                                                    s::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}})
    return StepRangeLen{Float32, Float32, Float32}(Float32(first(s)), Float32(step(s)), length(s))
end

#####
##### 12. Float32 carbon chemistry for the air–sea CO₂ exchange
#####

# OceanBioME's carbon chemistry, which the CO₂ gas exchange solves at the surface, has two Float64 paths that Metal
# rejects ("unsupported use of double value" in `compute_gas_exchange!`):
# (1) The density (mol kg⁻¹ ↔ mmol m⁻³) is SeawaterPolynomials' TEOS-10 `_ρ`, whose reference profile `r₀` multiplies
#     untyped Float64 constants. The same polynomial with the constants in Float32 (the anomaly `r′` is already typed);
#     more specific than OceanBioME's `teos10_polynomial_approximation(T::FT, ...)`, so it only replaces it for Float32.
# (2) The default boron, sulfate and fluoride totals (`0.000232 / 10.811 * S / 1.80655`, ...) are formed in Float64
#     before `convert`. The same totals with the constants in the salinity's float type, passed explicitly (new methods
#     for a `CarbonChemistry`, more specific than OceanBioME's, which leave it untyped).
using SeawaterPolynomials.TEOS10: TEOS10, τ, s, ζ, r′
import OceanBioME.Models: teos10_polynomial_approximation
import OceanBioME.Models.GasExchangeModel: surface_CO₂
using OceanBioME.Models.CarbonChemistryModel: CarbonChemistry

@inline function f32_r₀(ζ)
    R = (TEOS10.R₀₀, TEOS10.R₀₁, TEOS10.R₀₂, TEOS10.R₀₃, TEOS10.R₀₄, TEOS10.R₀₅)
    return (((((Float32(R[6]) * ζ + Float32(R[5])) * ζ + Float32(R[4])) * ζ + Float32(R[3])) * ζ + Float32(R[2])) * ζ + Float32(R[1])) * ζ
end

@inline function teos10_polynomial_approximation(T::Float32, Sp, Pbar = 0, lon = 0, lat = 0)
    Z = - 10 * Float32(Pbar)
    τ₀, s₀, ζ₀ = τ(T), s(Float32(Sp)), ζ(Z)
    return f32_r₀(ζ₀) + r′(τ₀, s₀, ζ₀)
end

@inline seawater_constituents(S::FT) where FT = (boron    = convert(FT, 0.000232 / 10.811 / 1.80655) * S,
                                                 sulfate  = convert(FT, 0.14 / 96.062 / 1.80655) * S,
                                                 fluoride = convert(FT, 0.000067 / 18.9984 / 1.80655) * S)

@inline surface_CO₂(::Nothing, carbon_chemistry::CarbonChemistry, i, j; S, kwargs...) =
    carbon_chemistry(; S, seawater_constituents(S)..., kwargs..., output = Val(:CO₂))

@inline function surface_CO₂(pH, carbon_chemistry::CarbonChemistry, i, j; DIC, S, kwargs...)
    pH⁻ = @inbounds pH[i, j, 1]
    initial_pH_guess = ifelse((pH⁻ > 0) & (pH⁻ < 14), pH⁻, convert(typeof(DIC), 8))
    CO₂, pHⁿ = carbon_chemistry(; DIC, S, seawater_constituents(S)..., kwargs..., initial_pH_guess, output = Val((:CO₂, :pHᶠ)))
    @inbounds pH[i, j, 1] = pHⁿ
    return CO₂
end

#####
##### 14. Float64 `Time` in Metal kernels
#####

# NumericalEarth's river routing (`interpolate_state!` for a `RoutedPrescribedLand`, Lands/river_routing.jl:387)
# passes `Time(coupled_model.clock.time)`, the Float64 clock time, to `_scatter_river_discharge!`, which interpolates
# the discharge series at it, so the kernel fails ("unsupported use of double value"). Item 9 converts a `Clock`, not
# a bare `Time`; here every `Time{Float64}` adapted to a Metal kernel is converted to Float32 as well.
using Oceananigans.Units: Time

Adapt.adapt_structure(to::Metal.Adaptor, t::Time{Float64}) = Time(convert(Float32, t.time))
