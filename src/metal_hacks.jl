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
# 7. `architecture_ready` for NetCDF data with missing values: missing → NaN on the host before the copy to Metal.
# 8. `cpu_interpolating_time_indices` on Metal: a Float32 time-interpolation weight for prescribed atmosphere/radiation.
# 9. Float32 clocks in Metal kernels, roughness lengths and interface humidity: the atmosphere-ocean flux solver.

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

# NetCDF variables with a fill value arrive as `Array{Union{Missing, Float32}}`, which NumericalEarth 0.8.1 copies to
# the GPU as is (`architecture_ready`, DataWrangling/set_region_data.jl:211) and NaN-converts in the kernel
# (`nan_convert_missing`). Metal.jl 1.11.1 cannot hold isbits-union arrays ("MtlArray does not yet support isbits-union
# arrays"), e.g. in every JRA55 FieldTimeSeries. Same missing → NaN, on the host first.
import NumericalEarth.DataWrangling: architecture_ready

architecture_ready(arch::GPU{Metal.MetalBackend}, data::AbstractArray{Union{Missing, T}}) where T <: AbstractFloat =
    on_architecture(arch, coalesce.(data, T(NaN)))

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
    return TimeInterpolator(Float32(ti.fractional_index), ti.first_index, ti.second_index, ti.length)
end

#####
##### 9. Float32 atmosphere-ocean flux solver
#####

# NumericalEarth 0.8.1's flux solver (`_compute_atmosphere_ocean_interface_state!`) meets Float64 three ways on Metal,
# failing with "unsupported use of double value" and the type instability it causes; ported from the fixes in
# ~/Documents/NumericalEarth.jl (jsw/oceanbiome-ext-changes, uncommitted):
#
# - its launch sites pass the model `clock` (Float64 time) where they should pass `convert_time(grid, clock)`. Here
#   every Clock adapted to a Metal kernel gets Float32 times instead, as `convert_time` does (a no-op for Oceananigans'
#   own kernels, which already convert);
# - `MomentumRoughnessLength` and `ScalarRoughnessLength` keep the default `air_kinematic_viscosity = 1.5e-5` Float64;
# - `default_ao_specific_humidity(ocean)` takes `eltype(ocean)`, which is `Any` for a `Simulation`, so 0.98 stays Float64.
using Oceananigans.TimeSteppers: Clock
using NumericalEarth.EarthSystemModels.InterfaceComputations: MomentumRoughnessLength, ScalarRoughnessLength,
                                                              SimilarityScales, ImpureSaturationSpecificHumidity,
                                                              AtmosphericThermodynamics
import NumericalEarth.EarthSystemModels.InterfaceComputations: default_roughness_lengths, default_ao_specific_humidity
using Oceananigans.TimeSteppers: Adapt # not a direct dependency of this project

Adapt.adapt_structure(to::Metal.Adaptor, clock::Clock) =
    (time          = Float32(clock.time),
     last_Δt       = Float32(clock.last_Δt),
     last_stage_Δt = Float32(clock.last_stage_Δt),
     iteration     = clock.iteration,
     stage         = clock.stage)

function default_roughness_lengths(FT::Type{Float32})
    ν = convert(FT, 1.5e-5) # m² s⁻¹
    return SimilarityScales(MomentumRoughnessLength(FT; air_kinematic_viscosity = ν),
                            ScalarRoughnessLength(FT; air_kinematic_viscosity = ν),
                            ScalarRoughnessLength(FT; air_kinematic_viscosity = ν))
end

default_ao_specific_humidity(ocean::Simulation) =
    ImpureSaturationSpecificHumidity(AtmosphericThermodynamics.Liquid(), convert(eltype(ocean.model.grid), 0.98))

# some timestepping nonsense
# HACK: Oceananigans 0.113.6 converts the time step to the grid's float type before the kernels see it
# (`kernel_time_step`, TimeSteppers/split_runge_kutta.jl); the jsw/for-oaemip branch (0.113.4) passes the Float64 Δt
# on, which Metal rejects (in the vertically implicit diffusion solve). Convert it here instead, as 0.113.6 does.
import Oceananigans.Models.HydrostaticFreeSurfaceModels: rk_substep!, step_closure_prognostics!
rk_substep!(model::HydrostaticFreeSurfaceModel, Δτ::Float64, callbacks) = rk_substep!(model, Float32(Δτ), callbacks)
step_closure_prognostics!(model::HydrostaticFreeSurfaceModel, Δt::Float64) = step_closure_prognostics!(model, Float32(Δt))


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
