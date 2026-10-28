# # California Current surface movie of metal.jl
#
# Every daily surface snapshot written by metal.jl (1/4°, 2010–2011): temperature with the surface currents,
# salinity, surface speed and sea surface height, with land masked out.
#
# Run from the repository root, after metal.jl:
#
#     julia --project 02_california_current/plot_surface.jl

using Oceananigans
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using CairoMakie
using GeoMakie: coastlines
using Dates

dir        = joinpath(@__DIR__, "output")
filename   = joinpath(dir, "surface.jld2")
start_date = DateTime(2010, 1, 1)   # time 0 of metal.jl

# Loaded whole (InMemory): the open boundary conditions saved with the fields cannot fill halos when a single
# snapshot is read from disk
T = FieldTimeSeries(filename, "T")   # conservative temperature, °C
S = FieldTimeSeries(filename, "S")   # absolute salinity, g kg⁻¹
u = FieldTimeSeries(filename, "u")   # eastward velocity, m s⁻¹
v = FieldTimeSeries(filename, "v")   # northward velocity, m s⁻¹
η = FieldTimeSeries(filename, "η")   # free surface displacement, m

grid  = T.grid
Nz    = size(grid, 3)
times = T.times

#####
##### Surface fields at cell centres, land as NaN
#####

surface(field) = interior(field, :, :, 1)

# η sits on the top face, which is never immersed, so it is copied to the surface cells to mask it
ηᶜ = CenterField(grid; indices = (:, :, Nz))

function snapshot(n)
    Tₙ = T[n]
    Sₙ = S[n]
    uᶜ = Field(@at (Center, Center, Center) u[n])
    vᶜ = Field(@at (Center, Center, Center) v[n])
    speed = Field(@at (Center, Center, Center) sqrt(u[n]^2 + v[n]^2))
    set!(ηᶜ, interior(η[n]))

    for field in (uᶜ, vᶜ, speed)
        compute!(field)
    end

    for field in (Tₙ, Sₙ, uᶜ, vᶜ, speed, ηᶜ)
        mask_immersed_field!(field, NaN)
    end

    # Sea surface height relative to its mean over the domain
    ssh = surface(ηᶜ) .- sum(filter(!isnan, surface(ηᶜ))) / count(!isnan, surface(ηᶜ))

    return (T = copy(surface(Tₙ)), S = copy(surface(Sₙ)), u = copy(surface(uᶜ)), v = copy(surface(vᶜ)),
            speed = copy(surface(speed)), ssh)
end

@info "masking $(length(times)) snapshots..."
frames = [snapshot(n) for n in eachindex(times)]

#####
##### Movie
#####

λ = λnodes(grid, Center())
φ = φnodes(grid, Center())

n     = Observable(1)
frame = @lift frames[$n]
title = @lift "California Current at 1/4° on Metal: " *
              Dates.format(start_date + Millisecond(round(Int, 1000 * times[$n])), "d u yyyy")

# Temperature range over the whole run, so the colours mean the same in every frame
T_range = extrema(Iterators.filter(!isnan, Iterators.flatten(f.T for f in frames)))

fig = Figure(size = (1300, 1250), fontsize = 16)
Label(fig[0, :], title, fontsize = 22, halign = :left, tellwidth = false)

# Equal distances in x and y at the centre of the domain
aspect = (λ[end] - λ[1]) * cosd((φ[1] + φ[end]) / 2) / (φ[end] - φ[1])

panels = [(fig[1, 1], fig[1, 2], :T,     "Temperature (°C) and surface currents", :thermal, T_range),
          (fig[1, 3], fig[1, 4], :S,     "Salinity (g kg⁻¹)",                     :haline,  (32.5, 34)),
          (fig[2, 1], fig[2, 2], :speed, "Surface speed (m s⁻¹)",                 :speed,   (0, 0.4)),
          (fig[2, 3], fig[2, 4], :ssh,   "Sea surface height anomaly (m)",        :balance, (-0.15, 0.15))]

axes = map(panels) do (position, colorbar_position, name, title, colormap, colorrange)
    ax = Axis(position; title, titlealign = :left, aspect, xlabel = "Longitude (°E)", ylabel = "Latitude (°N)",
              limits = (extrema(λ), extrema(φ)))
    hm = heatmap!(ax, λ, φ, @lift(getproperty($frame, name)); colormap, colorrange, nan_color = :lightgray)
    lines!(ax, coastlines(); color = :black, linewidth = 1)
    Colorbar(colorbar_position, hm, height = Relative(0.8))
    return ax
end

# Every fourth current vector, 1 m s⁻¹ drawn as 3° long
s = 4
arrows2d!(axes[1], λ[1:s:end], φ[1:s:end], @lift($frame.u[1:s:end, 1:s:end]), @lift($frame.v[1:s:end, 1:s:end]);
          lengthscale = 3, color = :white, shaftwidth = 1.5, tipwidth = 6, tiplength = 6)

movie = joinpath(dir, "surface.mp4")
@info "recording $(length(times)) frames to $movie..."
record(fig, movie, eachindex(times); framerate = 24) do i
    n[] = i
end
@info "saved $movie"
