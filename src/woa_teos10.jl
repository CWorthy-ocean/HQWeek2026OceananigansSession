# WOA initial conditions and restoring targets in TEOS-10, ported from OMIPSimulations (NumericalEarth
# experiments/OMIPSimulations, jsw/omip-bgc: `woa_to_teos10!`, `blend_monthly_initial_condition!` and
# `woa_salinity_fts_to_teos10!` in src/omip_simulation.jl), which NumericalOcean (GlobalOcean) uses. The model's
# equation of state is TEOS-10, so its T is Conservative Temperature Θ (°C) and S Absolute Salinity Sᴬ (g kg⁻¹); WOA
# holds in-situ temperature t (°C) and Practical Salinity Sᴾ.

using Dates: month
using Oceananigans.Architectures: architecture, on_architecture
using Oceananigans.DistributedComputations: cpu_architecture
using Oceananigans.Grids: λnode, φnode, znode, znodes
using Oceananigans.Fields: interpolate
using NumericalEarth.DataWrangling: DataWrangling, Metadatum
using NumericalEarth.DataWrangling.WOA: WOAMonthly
using SeawaterPolynomials.TEOS10: Sᴬ_from_Sᴾ, Θ_from_T

# Approximate hydrostatic pressure (dbar) from depth z (m, negative in the ocean)
@inline approx_pressure_dbar(z) = max(zero(z), -z)

"""
    woa_to_teos10!(T_field, S_field)

Convert WOA in-situ temperature `t` (°C) and Practical Salinity `Sᴾ` in `T_field` and `S_field` to TEOS-10
Conservative Temperature `Θ` (°C) and Absolute Salinity `Sᴬ` (g kg⁻¹), in place, on the host.
"""
function woa_to_teos10!(T_field, S_field)
    grid = T_field.grid
    cpu_grid = on_architecture(cpu_architecture(architecture(grid)), grid)
    Nx, Ny, Nz = size(grid)
    T_h = Array(interior(T_field))
    S_h = Array(interior(S_field))
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        t  = T_h[i, j, k]
        Sᴾ = S_h[i, j, k]
        (isnan(t) || isnan(Sᴾ)) && continue
        λ = λnode(i, j, k, cpu_grid, Center(), Center(), Center())
        φ = φnode(i, j, k, cpu_grid, Center(), Center(), Center())
        p = approx_pressure_dbar(znode(i, j, k, cpu_grid, Center(), Center(), Center()))
        Sᴬ = Sᴬ_from_Sᴾ(Sᴾ, p, λ, φ)
        T_h[i, j, k] = Θ_from_T(Sᴬ, t, p)
        S_h[i, j, k] = Sᴬ
    end
    copyto!(interior(T_field), T_h)
    copyto!(interior(S_field), S_h)
    return T_field, S_field
end

"""
    blend_monthly_initial_condition!(T_field, S_field, date, blend_depth)

Blend WOA Monthly for the month of `date` (converted to Θ and Sᴬ) into `T_field` and `S_field` (already Θ and Sᴬ,
e.g. annual WOA through `woa_to_teos10!`): weight 1 above `blend_depth` (m), tapering linearly to 0 at the monthly
climatology's reach (≈ 1500 m), annual values kept below. The annual mean starts 1 January with a seasonal
thermocline the season does not have; the monthly climatology has the right surface structure but is shallow.
"""
function blend_monthly_initial_condition!(T_field, S_field, date, blend_depth)
    grid = T_field.grid
    cpu_grid = on_architecture(cpu_architecture(architecture(grid)), grid)
    Nx, Ny, Nz = size(grid)

    Tmeta = Metadatum(:temperature; dataset = WOAMonthly(), date)
    Smeta = Metadatum(:salinity;    dataset = WOAMonthly(), date)
    Tnative = Field(Tmeta, CPU())
    Snative = Field(Smeta, CPU())

    # `set!` refuses a target deeper than the dataset, and interpolating a whole-depth field onto the grid reads
    # beyond the dataset's bottom (an out-of-bounds vertical index, unchecked in the kernel). Only cells the taper
    # keeps (shallower than the dataset) are interpolated, point by point, at depths clamped to its centres.
    deepest = Tnative.grid.Lz
    deepest > blend_depth ||
        throw(ArgumentError("blend_depth = $blend_depth m is below the monthly climatology's reach of $(round(deepest)) m"))
    z_native = znodes(Tnative.grid, Center())
    z_bottom, z_top = extrema(z_native)

    Ta = Array(interior(T_field)); Sa = Array(interior(S_field))

    # The monthly fields carry non-NaN fill values outside their coverage, so convert behind a physical-range test
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        isfinite(Ta[i, j, k]) && isfinite(Sa[i, j, k]) || continue
        z = znode(i, j, k, cpu_grid, Center(), Center(), Center())
        d = -z
        w = clamp((deepest - d) / (deepest - blend_depth), 0, 1)
        w > 0 || continue
        λ = λnode(i, j, k, cpu_grid, Center(), Center(), Center())
        φ = φnode(i, j, k, cpu_grid, Center(), Center(), Center())
        at = (λ, φ, clamp(z, z_bottom, z_top))
        t  = interpolate(at, Tnative)
        Sᴾ = interpolate(at, Snative)
        (isfinite(t) && isfinite(Sᴾ) && 1 < Sᴾ < 45 && -5 < t < 45) || continue
        p = approx_pressure_dbar(z)
        Sᴬᵐ = Sᴬ_from_Sᴾ(Sᴾ, p, λ, φ)
        Θᵐ  = Θ_from_T(Sᴬᵐ, t, p)
        Ta[i, j, k] = w * Θᵐ  + (1 - w) * Ta[i, j, k]
        Sa[i, j, k] = w * Sᴬᵐ + (1 - w) * Sa[i, j, k]
    end

    copyto!(interior(T_field), Ta)
    copyto!(interior(S_field), Sa)

    @info "blended WOA Monthly $(month(date)) into the initial condition above $(blend_depth) m, " *
          "tapering to the monthly climatology's reach of $(round(Int, deepest)) m"

    return T_field, S_field
end

"""
    woa_salinity_fts_to_teos10!(fts)

Convert every time slice of a WOA Practical Salinity `FieldTimeSeries` (e.g. a `DatasetRestoring` target) to
Absolute Salinity (g kg⁻¹), in place. All time indices must be in memory.
"""
function woa_salinity_fts_to_teos10!(fts)
    grid = fts.grid
    cpu_grid = on_architecture(cpu_architecture(architecture(grid)), grid)
    Nx, Ny, Nz = size(grid)
    for n in eachindex(fts.times)
        S_int = interior(fts[n])
        S_h   = Array(S_int)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            Sᴾ = S_h[i, j, k]
            isnan(Sᴾ) && continue
            λ = λnode(i, j, k, cpu_grid, Center(), Center(), Center())
            φ = φnode(i, j, k, cpu_grid, Center(), Center(), Center())
            p = approx_pressure_dbar(znode(i, j, k, cpu_grid, Center(), Center(), Center()))
            S_h[i, j, k] = Sᴬ_from_Sᴾ(Sᴾ, p, λ, φ)
        end
        copyto!(S_int, S_h)
    end
    return fts
end
