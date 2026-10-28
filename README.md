# HQ Week 2026 Oceananigans session

Examples for the Oceananigans session at HQ Week 2026. Starting with a small 2-D turbulence run and work up to regional and global ocean models with biogeochemistry and OAE. Most of the examples run on an Apple-silicon Mac GPU through [Metal.jl](https://github.com/JuliaGPU/Metal.jl). A few are written for NVIDIA GPUs as an example of a normal workflow.

## Forcing data

The regional and global examples need about 30 GB of forcing and initial-condition data that is not in this repository. Please download it before the session from the Google drive. If you can't download it in advance, I will bring a flash drive with it on.

Please unpack it in the repository root so that you get a `data/` folder next to this README:

```bash
cd HQWeek2026OceananigansSession
tar -xzf /path/to/data.tar.gz      # creates ./data/
```

The biogeochemistry and OAE runs also need the UNIFIED BGC dataset to initialise the examples so if you want to run those examples as they are run them as they are  please download it and move it to `data/bgc_init.nc`:

```bash
mv /path/to/UNIFIED_BGC.nc data/bgc_init.nc
```

## Setup

1. **Install Julia:**
   ```bash
   curl -fsSL https://install.julialang.org | sh
   ```
2. **Clone the workshop:**
   ```bash
   git clone git@github.com:CWorthy-ocean/HQWeek2026OceananigansSession.git
   ```
3. **Go into the workshop and start Julia:**
   ```bash
   cd HQWeek2026OceananigansSession
   julia --project
   ```
4. **Install all the dependencies** (on an NVIDIA GPU, do the [CUDA steps](#running-on-an-nvidia-gpu-cuda) first):
   ```julia
   using Pkg; Pkg.instantiate()
   ```
5. Hope that everything magically installed.

Julia can take a while to compile packages the first time it is run, so the first instantiate and precompile take a while. Do this before the session as well.

### Running on an NVIDIA GPU (CUDA)

The project is set up for Apple-silicon Macs, where it uses Metal. **On an NVIDIA GPU you need CUDA 6.4.2** (LLVM 10, still KernelAbstractions 0.9). Remove Metal and add that version before you instantiate. In the package REPL (`]`) with the project active:

```
pkg> rm Metal
pkg> add CUDA@6.4.2
```

or in one line:

```bash
julia --project -e 'using Pkg; Pkg.rm("Metal"); Pkg.add(name = "CUDA", version = "6.4.2"); Pkg.instantiate()'
```

## Projects

### `01_isotropic_turbulence/` — 2-D turbulence on a Mac GPU

`isotropic_turbulence.jl` runs decaying 2-D turbulence on a doubly periodic 1024 × 1024 grid in a unit square, with `NonhydrostaticModel`, WENO advection and ν = 10⁻⁵ (non-dimensional, so Re = 10⁵). It writes u, v and the vorticity ζ = ∂v/∂x − ∂u/∂y every 0.01 time units and then makes a movie. **No data needed.**

```bash
julia --project 01_isotropic_turbulence/isotropic_turbulence.jl
```

Or start a REPL from the repository root with `julia --project` and include the file:

```julia
include("01_isotropic_turbulence/isotropic_turbulence.jl")
```

### `02_california_current/` — regional California Current

These are regional hydrostatic models from 134–115.5° W and 30–48° N, 42 levels, 5500 m deep, for 2010–2011. They are forced by JRA55 (already cut to the region) and use open boundaries from GLORYS (downsampled to 1/4°) with Flather conditions and sponges. Salinity is restored to WOA.

| Script | Hardware | Description |
|---|---|---|
| `metal.jl` | Mac (Metal), Float32 | Physics only, at 1/4° with Δt = 60 min (or 1/2° with Δt = 120 min) |
| `metal_alkalinity.jl` | Mac (Metal), Float32 | `metal.jl` plus OceanBioME `MITgcmDIC` biogeochemistry with two carbonate systems. 2010 is spin-up. Then 1 Tmol of alkalinity is released off Monterey Bay through 2011, and `Alk2 − Alk1` / `DIC2 − DIC1` give the perturbation and the extra CO₂ uptake |
| `gpu.jl` | NVIDIA (CUDA), Float64 | Physics only, at 1/36° (666 × 648 × 42, Δt = 5 min). 2010–2011 took 12.9 h on one H100. Restarts from the latest checkpoint |
| `plot_surface.jl` | CPU | Movie of the daily surface output of `metal.jl`: temperature with surface currents, salinity, surface speed and sea surface height anomaly, with land masked. Saves `output/surface.mp4`. **Run `metal.jl` first** |

```bash
julia --project 02_california_current/metal.jl
julia --project --threads=4 02_california_current/gpu.jl
julia --project 02_california_current/plot_surface.jl
```

Or start a REPL from the repository root with `julia --project` (`julia --project --threads=4` for `gpu.jl`) and include the files:

```julia
include("02_california_current/metal.jl")
include("02_california_current/plot_surface.jl")
```

The included GLORYS data only covers this region and the 2010–2012 period. Running a different region or period needs different data (usually `GLORYSDaily()` downloaded with CopernicusMarine).

### `03_global/` — global ORCA2

These are global ocean-only runs (no sea ice: temperature is clipped at the freezing point) on the ORCA2 tripolar grid with 30 levels and Δt = 90 min, forced by repeat-year JRA55 (RYF 1990–1991).

| Script | Hardware | Description |
|---|---|---|
| `orca2_metal.jl` | Mac | Physics only, 1 year |
| `orca2_metal_bgc.jl` | Mac | Plus `MITgcmDIC` biogeochemistry, 2 years. Writes `output/orca2_metal_bgc_restart.jld2` |
| `orca2_metal_oae.jl` | Mac | Restarts from the BGC run and releases 1 Tmol of alkalinity over a year at five sites (Kuroshio, Tasmania, Labrador, California, Arabian Sea). **Run `orca2_metal_bgc.jl` first** |
| `1degree.jl` | NVIDIA | ORCA1 with sea ice and BGC through `NumericalOcean`. This is a work in progress |
| `download_jra55.jl` | — | Downloads the RYF JRA55 files into `data/JRA55`. You don't need it if you unpacked `data.tar.gz` |
| `plot_surface.jl` | CPU | Plots the last surface snapshot of `orca2_metal.jl`. It shows why the tripolar grid can't be drawn as a plain map, then conservatively regrids temperature, salinity and surface speed to a regular 1/2° grid (land excluded via the wet fraction) and draws them on a Robinson projection. Saves `output/surface.png`. **Run `orca2_metal.jl` first** |

```bash
julia --project 03_global/orca2_metal.jl
julia --project 03_global/plot_surface.jl
```

Or start a REPL from the repository root with `julia --project` and include the files:

```julia
include("03_global/orca2_metal.jl")
include("03_global/plot_surface.jl")
```

### `src/` — the `HQWeek2026OceananigansSession` package

Shared code that the scripts load with `using HQWeek2026OceananigansSession`:

- **Metal workarounds:** `metal_hacks.jl`
- **Dataset definitions for the workshop data:** `GLORYSQuarterDegree`, `RegionalJRA55`, `BGCInitClimatology`, `MaunaLoaCO₂`, and the WOA → TEOS-10 conversion
- **The ORCA2 grid:** `ORCA2Grid`
- **Regional boundary conditions:** Flather boundaries, drag and sponge
- **Air–sea gas exchange without sea ice**
- **Alkalinity-release forcing**
- **Restart save/load**
