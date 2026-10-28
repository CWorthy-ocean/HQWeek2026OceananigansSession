# # Open-boundary, bottom drag and sponge functions for the regional (California Current) example
#
# Discrete-form boundary-condition functions; pass their parameters through `parameters`:
#
#     drag_parameters = (κ = 0.41f0, z₀ = 0.01f0)
#     bottom = IMEXFluxBoundaryCondition(0, u_bottom_drag; discrete_form = true, parameters = drag_parameters)
#     west   = GravityWaveRadiationBoundaryCondition(x_flather; discrete_form = true,
#                                                    parameters = (u = uᵉ, η = ηᵉ, i = 1, face = 1, time = barotropic_time))
#
# (on CUDA or the CPU leave out `time`: the Flather functions then read the clock's time)

using Oceananigans.Operators: Δzᶠᶜᶜ, Δzᶜᶠᶜ
using Oceananigans.Grids: immersed_peripheral_node, immersed_inactive_node, node
using Oceananigans.OutputReaders: Time
using NumericalEarth.Oceans: spᶠᶜᶜ, spᶜᶠᶜ

#####
##### Exterior values for the open boundaries: the series `ψ` at the boundary column `i` (x_boundary, west/east) or row
##### `j` (y_boundary, south/north), at the clock's time
#####

@inline x_boundary(j, k, grid, clock, fields, p) = @inbounds p.ψ[p.i, j, k, Time(clock.time)]
@inline y_boundary(i, k, grid, clock, fields, p) = @inbounds p.ψ[i, p.j, k, Time(clock.time)]

# Exterior values fixed in time: the field `c` at the boundary column `i` (west/east) or row `j` (south/north)
@inline x_fixed_boundary(j, k, grid, clock, fields, p) = @inbounds p.c[p.i, j, k]
@inline y_fixed_boundary(i, k, grid, clock, fields, p) = @inbounds p.c[i, p.j, k]

#####
##### Bottom drag: quadratic, with a log-layer drag coefficient Cd = [κ / ln(1 + Δz / 2z₀)]² for the bottom cell
##### (parameters κ, the von Kármán constant, and z₀, the roughness length in m), on the bottom of the grid and on the
##### immersed bottom
#####

@inline log_layer_Cd(Δz, p) = (p.κ / log(1 + Δz / (2 * p.z₀)))^2

@inline u_bottom_drag(i, j, grid, clock, Φ, p) = - log_layer_Cd(Δzᶠᶜᶜ(i, j, 1, grid), p) * spᶠᶜᶜ(i, j, 1, grid, Φ, 0)
@inline v_bottom_drag(i, j, grid, clock, Φ, p) = - log_layer_Cd(Δzᶜᶠᶜ(i, j, 1, grid), p) * spᶜᶠᶜ(i, j, 1, grid, Φ, 0)
@inline u_immersed_drag(i, j, k, grid, clock, Φ, p) = - log_layer_Cd(Δzᶠᶜᶜ(i, j, k, grid), p) * spᶠᶜᶜ(i, j, k, grid, Φ, 0)
@inline v_immersed_drag(i, j, k, grid, clock, Φ, p) = - log_layer_Cd(Δzᶜᶠᶜ(i, j, k, grid), p) * spᶜᶠᶜ(i, j, k, grid, Φ, 0)

#####
##### Flather: the exterior state (U, η) for `GravityWaveRadiationBoundaryCondition` on a west/east (`x_flather`) or
##### south/north (`y_flather`) boundary. The exterior transport is the velocity series `u` (or `v`) integrated over
##### the wet cells at column `i` (row `j`) and the face `face`; η is the series `η` there. Parameters: u (or v), η,
##### i (or j), face, and on Metal `time`, a one-element array holding the time to read the series at (the barotropic
##### substeps' clock time is Float64, which Metal kernels cannot use; see `barotropic_time_array`). Without `time` the
##### series are read at the clock's time (CUDA, CPU)
#####

@inline flather_time(clock, p) = hasproperty(p, :time) ? @inbounds(p.time[1]) : clock.time

@inline function x_flather(j, k, grid, clock, fields, p)
    n = Time(flather_time(clock, p))
    U = zero(grid)
    for k′ in 1:grid.Nz
        dry = immersed_peripheral_node(p.face, j, k′, grid, Face(), Center(), Center()) |
              immersed_inactive_node(p.face, j, k′, grid, Face(), Center(), Center())
        U += ifelse(dry, zero(grid), @inbounds(p.u[p.i, j, k′, n]) * Δzᶠᶜᶜ(p.face, j, k′, grid))
    end
    return (U, @inbounds p.η[p.i, j, 1, n])
end

@inline function y_flather(i, k, grid, clock, fields, p)
    n = Time(flather_time(clock, p))
    V = zero(grid)
    for k′ in 1:grid.Nz
        dry = immersed_peripheral_node(i, p.face, k′, grid, Center(), Face(), Center()) |
              immersed_inactive_node(i, p.face, k′, grid, Center(), Face(), Center())
        V += ifelse(dry, zero(grid), @inbounds(p.v[i, p.j, k′, n]) * Δzᶜᶠᶜ(i, p.face, k′, grid))
    end
    return (V, @inbounds p.η[i, p.j, 1, n])
end

#####
##### Sponge: a discrete-form viscosity and diffusivity for `HorizontalScalarDiffusivity` that rises linearly from zero
##### `width` cells inside the west, south and north boundaries to `ν` at them (ROMS / C-Star). Parameters: the
##### boundaries' longitude `west` and latitudes `south` and `north`, the grid spacing `Δ` in degrees, `width` and `ν`
#####

@inline function sponge_coefficient(i, j, k, grid, ℓx, ℓy, ℓz, clock, fields, p)
    λ, φ, _ = node(i, j, k, grid, ℓx, ℓy, ℓz)
    n = min(λ - p.west, φ - p.south, p.north - φ) / p.Δ + 0.5f0  # cells from the edge: 1 at the first interior point
    return p.ν * max(0, 1 - n / p.width)
end
