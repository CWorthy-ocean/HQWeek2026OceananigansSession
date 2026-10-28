# # Alkalinity releases into the surface cells
#
# As 02_california_current/metal_alkalinity.jl: a release of `total` mmol at a constant rate over `duration`, into the
# surface cells, Gaussian (standard deviation `σ`, cut off at 3σ) around a site, stored as the tendency
# (mmol m⁻³ s⁻¹) it gives while on.

using Oceananigans.Architectures: on_architecture, CPU
using Oceananigans.Grids: node
using Oceananigans.Operators: Vᶜᶜᶜ

"""
    alkalinity_release_tendency(grid, site; σ = 100e3, total = 1000e12, duration = 365days)

Tendency field (mmol m⁻³ s⁻¹) on `grid` releasing `total` mmol (default 1 Tmol) over `duration` (s) into the wet
surface cells around `site = (λ, φ)` (°E, °N), weighted by a Gaussian of standard deviation `σ` (m) in the great-circle
distance and cut off at 3σ. Normalised over the cells it covers, so the total is exact at any resolution; errors if no
wet surface cell centre lies within 3σ of the site.
"""
function alkalinity_release_tendency(grid, site; σ = 100e3, total = 1000e12, duration = 365 * 86400)
    cpu_grid = on_architecture(CPU(), grid)
    Nx, Ny, Nz = size(cpu_grid)
    λ₀, φ₀ = site
    bottom = cpu_grid.immersed_boundary.bottom_height
    weights = zeros(eltype(grid), Nx, Ny, Nz)
    for j in 1:Ny, i in 1:Nx
        λ, φ, z = node(i, j, Nz, cpu_grid, Center(), Center(), Center())
        bottom[i, j, 1] < z || continue # dry
        r = cpu_grid.radius * acos(clamp(sind(φ₀) * sind(φ) + cosd(φ₀) * cosd(φ) * cosd(λ - λ₀), -1, 1))
        weights[i, j, Nz] = ifelse(r > 3σ, 0, exp(-r^2 / 2σ^2))
    end
    volume = sum(weights[i, j, Nz] * Vᶜᶜᶜ(i, j, Nz, cpu_grid) for i in 1:Nx, j in 1:Ny) # m³
    volume > 0 || throw(ArgumentError("no wet surface cell centre within 3σ = $(3σ / 1e3) km of $site"))
    tendency = weights .* convert(eltype(grid), total / volume / duration)
    return set!(CenterField(grid), tendency)
end

# The forcing: the stored tendency, on for the whole run
@inline alkalinity_release(i, j, k, grid, clock, fields, tendency) = @inbounds tendency[i, j, k]
