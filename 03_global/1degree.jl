using NumericalOcean # my wrapper package - likely to change 
using NumericalEarth
using Oceananigans
using Oceananigans.Units
using OceanBioME
using Dates
using CUDA 

# Setup ORCA1 grid which is nominally 1 degree with some manual fixes 
arch = GPU()
grid = ORCA1(arch)

# Get PAR for the BGC from the coupled atmosphere
surface_PAR = PARFromShortwave(grid)
light_attenuation = PrescribedAttenuationPAR(grid, surface_PAR;
                                             attenuation = PrescribedChlorophyllAttenuation(first_color_fraction = 0.0),
                                             attenuation_discrete_form = true)

# Setup the BGC model, parameters being tuned
plankton = ImplicitProductivity(; maximum_community_productivity = 9.0 / 360days,
                                  light_half_saturation = 30.0, # W / m²
                                  carbon_ratio = 117.0)         # mol C / mol P

biogeochemistry = MITgcmDIC(grid;
                            plankton,
                            light_attenuation,
                            open_bottom = false,
                            implicit_sinking = true,
                            store_flux = true)

# pCO₂ to put in the atmosphere
pCO₂ = MaunaLoaCO₂(arch; dates = DateTime(1990, 1, 15):Month(1):DateTime(1990, 12, 15),
                         start_date = DateTime(1990, 1, 1),
                         period = 365days)

# Setup a Simulation of an `EarthSystemModel` with an `Oceananigans` `HydrostaticFreeSurfaceModel`
# `ClimaSeaIce` ice model, JRA55 repeat year (1990-1991) atmosphere
simulation = forced_ocean_simulation(grid;
                                     biogeochemistry,
                                     atmosphere_tracers = (; pCO₂),
                                     jra55_dataset = RepeatYearJRA55(),
                                     start_date = DateTime(1990, 1, 1),
                                     backend_size = 2920,
                                     stop_time = 1000 * 365days)

# Pickup a spunup state
pickup_restart!(simulation, DateTime(1990, 1, 1); start_date = DateTime(1990, 1, 1), repeat_year = true)

# Add outputs (this adds a lot of out puts so maybe do this manually instead)
omip_diagnostics!(simulation; async = true)
oaemip_diagnostics!(simulation; async = true)

# Run the model (doing this try finally thing because the diagnostics do asyncronous output writing)
try
    run!(simulation)
finally
    flush_output!(simulation) # write anything still queued
end
