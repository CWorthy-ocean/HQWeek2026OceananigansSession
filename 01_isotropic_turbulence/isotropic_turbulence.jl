
using Oceananigans, Metal

grid = RectilinearGrid(GPU(MetalBackend()), Float32; 
                       size = (1024, 1024), 
                       topology = (Periodic, Periodic, Flat), 
                       extent = (1, 1))

model = NonhydrostaticModel(grid;
                            closure = ScalarDiffusivity(Float32; ν = 1f-5),
                            advection = WENO(Float32))

set!(model, u = (x, y) -> randn())

simulation = Simulation(model,
                        Δt = 0.5 * minimum_xspacing(grid),
                        stop_time = 1)

u, v, … = model.velocities 

fname = "01_isotropic_turbulence_Re_10^$(round(Int, log10(1/model.closure.ν)))"

simulation.output_writers[:speeds] = 
    JLD2Writer(model, 
               (; u, v, ζ = ∂x(v) - ∂y(u));
               filename = fname * ".jld2", 
               schedule = TimeInterval(0.01),
               overwrite_files = true)

prog(sim) = @info prettytime(sim)
add_callback!(simulation, prog,
             IterationInterval(10))

run!(simulation)

# Plotting

ζ = FieldTimeSeries(fname * ".jld2", "ζ")

using CairoMakie

n = Observable(1)

title = @lift prettytime(ζ.times[$n])
ζn = @lift ζ[$n]
colorrange = @lift (-1, 1) .* maximum(abs, ζ[$n])

fig = Figure()

ax = Axis(fig[1, 1]; title)

heatmap!(ax, ζn; colorrange, colormap = :balance)

record(fig, fname*".mp4", 1:length(ζ), framerate = 10) do nn
    n[] = nn
    @info nn
end
