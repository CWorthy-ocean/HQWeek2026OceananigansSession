# # Surface state of orca2_metal.jl, regridded to latitude–longitude
#
# ORCA2 is a tripolar grid: north of ~20°N its rows bend towards two poles over Siberia and Canada, so its (i, j)
# indices are neither longitude nor latitude and the fields cannot be drawn as a plain map (top left). This script
#
# 1. reads the last surface snapshot saved by orca2_metal.jl and masks the land with NaN,
# 2. builds a conservative regridder from ORCA2 to a regular 1/2° grid with ConservativeRegridding,
# 3. regrids temperature, salinity and surface speed, dividing by the regridded wet fraction so land never leaks in,
# 4. draws the native temperature next to the regridded maps on a Robinson projection.
#
# Run from the repository root, after orca2_metal.jl:
#
#     julia --project 03_global/plot_surface.jl

# Before `using NumericalEarth`: it reads the data directory when it loads (for the ORCA2 bathymetry)
ENV["NUMERICALEARTH_DATA_DIRECTORY"] = "data"

using Oceananigans
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using HQWeek2026OceananigansSession # ORCA2Grid
using ConservativeRegridding
using CairoMakie
using GeoMakie
using Dates

dir        = joinpath(@__DIR__, "output")
filename   = joinpath(dir, "orca2_metal_surface.jld2")
start_date = DateTime(1990, 1, 1)   # time 0 of orca2_metal.jl

#####
##### 1. The last surface snapshot
#####

T = FieldTimeSeries(filename, "T"; backend = OnDisk())   # conservative temperature, °C
S = FieldTimeSeries(filename, "S"; backend = OnDisk())   # absolute salinity, g kg⁻¹
u = FieldTimeSeries(filename, "u"; backend = OnDisk())   # m s⁻¹, along the grid's i direction
v = FieldTimeSeries(filename, "v"; backend = OnDisk())   # m s⁻¹, along the grid's j direction

n    = length(T.times)
date = start_date + Millisecond(round(Int, 1000 * T.times[n]))

Tₙ = T[n]
Sₙ = S[n]

# Speed at cell centres: u and v live on the cell faces, `@at` interpolates them to the centres first. Speed does not
# depend on the direction of the grid axes, so it can be regridded as a scalar even where they rotate in the north
speedₙ = Field(@at (Center, Center, Center) sqrt(u[n]^2 + v[n]^2))
compute!(speedₙ)

# Land is NaN from here on (the native fields are drawn as they are)
for field in (Tₙ, Sₙ, speedₙ)
    mask_immersed_field!(field, NaN)
end

#####
##### 2. The regridder
#####

# ConservativeRegridding computes the area of overlap of every ORCA2 cell with every destination cell on the sphere.
# Its geometry fails on the Float32 grid that orca2_metal.jl runs on (the Float32 cell corners are not exactly unit
# vectors), so the same grid is rebuilt in Float64, with the vertical grid of orca2_metal.jl so the land matches
z    = ExponentialDiscretization(30, -5500, 0; scale = 1240)
grid = ORCA2Grid(CPU(), Float64; z, major_basins = 1)
Nz   = size(grid, 3)

source = CenterField(grid; indices = (:, :, Nz))

destination_grid = LatitudeLongitudeGrid(size = (720, 360, 1), longitude = (-180, 180), latitude = (-90, 90), z = (0, 1))
destination      = CenterField(destination_grid)

regridder = ConservativeRegridding.Regridder(destination, source)   # built once, reused for every field

#####
##### 3. Regridding without land
#####

# Regridding a field that is 1 in the ocean and 0 on land gives the wet fraction of each destination cell
set!(source, 1)
mask_immersed_field!(source, 0)
ConservativeRegridding.regrid!(destination, regridder, source)
wet_fraction = copy(interior(destination, :, :, 1))

# Regrid with land as 0, then divide by the wet fraction: the mean over the ocean part of each destination cell.
# Cells less than half ocean are land (NaN). North of ~73°N the wet fraction reaches 2, because the cells of the
# fold row are stored twice; the field is counted twice there too, so the division cancels it
function regrid_ocean(field)
    set!(source, replace(interior(field), NaN => 0))
    mask_immersed_field!(source, 0)
    ConservativeRegridding.regrid!(destination, regridder, source)
    regridded = interior(destination, :, :, 1) ./ wet_fraction
    regridded[wet_fraction .< 1/2] .= NaN
    return regridded
end

T̄     = regrid_ocean(Tₙ)
S̄     = regrid_ocean(Sₙ)
speed̄ = regrid_ocean(speedₙ)

#####
##### 4. Plot
#####

# GeoMakie reprojects `surface!` (flat, with the data as colour) but not `heatmap!`, and needs the seam at ±180°
# closed: a column at each end with the mean of the first and last
λ = [-180; λnodes(destination_grid, Center()); 180]
φ = φnodes(destination_grid, Center())
with_seam(a) = (seam = (a[1:1, :] .+ a[end:end, :]) ./ 2; vcat(seam, a, seam))
flat = zeros(length(λ), length(φ))

fig = Figure(size = (1400, 950), fontsize = 15)
Label(fig[0, :], "ORCA2 on Metal, JRA55 repeat-year forcing: " * Dates.format(date, "d u yyyy"),
      fontsize = 20, halign = :left, tellwidth = false)

native = Axis(fig[1, 1]; title = "Native grid: temperature (°C) by index", titlealign = :left, xlabel = "i", ylabel = "j")
hm = heatmap!(native, interior(Tₙ, :, :, 1); colormap = :thermal, colorrange = (-2, 30), nan_color = :lightgray)
Colorbar(fig[1, 2], hm)

panels = [(fig[1, 3], fig[1, 4], T̄,     "Regridded: temperature (°C)",      :thermal, (-2, 30)),
          (fig[2, 1], fig[2, 2], S̄,     "Regridded: salinity (g kg⁻¹)",     :haline,  (32, 38)),
          (fig[2, 3], fig[2, 4], speed̄, "Regridded: surface speed (m s⁻¹)", :speed,   (0, 0.6))]

for (position, colorbar_position, data, title, colormap, colorrange) in panels
    ax = GeoAxis(position; dest = "+proj=robin", title, titlealign = :left)
    sf = surface!(ax, λ, φ, flat; color = with_seam(data), shading = NoShading, colormap, colorrange, nan_color = :lightgray)
    lines!(ax, GeoMakie.coastlines(); color = :black, linewidth = 0.5)
    Colorbar(colorbar_position, sf, height = Relative(0.7))
end

save(joinpath(dir, "surface.png"), fig)
@info "saved $(joinpath(dir, "surface.png"))"
