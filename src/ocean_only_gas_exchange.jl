# # Air–sea gas exchange for an ocean-only model with OceanBioME biogeochemistry
#
# HACK: NumericalEarth's OceanBioME extension scales the gas exchange by the open-water fraction, reading the ice
# concentration from `coupled_model.sea_ice.model`, which the ocean-only model's sea ice (`FreezingLimitedOceanTemperature`,
# as `OceanOnlyModel` builds) does not have. This is the same flux update with no ice, for `OceanOnlyModel`s with
# OceanBioME's NutrientsPlanktonDetritus models (e.g. `MITgcmDIC`). Remove it when the extension handles models without
# sea ice.

using OceanBioME: OceanBioME
using NumericalEarth: NumericalEarth
using NumericalEarth.EarthSystemModels: EarthSystemModel
using NumericalEarth.SeaIces: FreezingLimitedOceanTemperature
using Oceananigans.Biogeochemistry: required_biogeochemical_tracers
using Oceananigans.Fields: ZeroField, compute!
using Oceananigans.Utils: launch!
using Oceananigans.Architectures: architecture

function NumericalEarth.Oceans.update_net_ocean_biogeochemical_fluxes!(coupled_model::EarthSystemModel{<:Any, <:Any, <:Any, <:FreezingLimitedOceanTemperature},
                                                                      biogeochemistry::OceanBioME.DiscreteBiogeochemistry{<:OceanBioME.NutrientsPlanktonDetritus},
                                                                      ocean, grid)
    # The extension is loaded with OceanBioME, after this package is precompiled, so it is looked up here
    BGCExt = Base.get_extension(NumericalEarth, :NumericalEarthOceanBioMEExt)

    underlying = biogeochemistry.underlying_biogeochemistry
    properties = coupled_model.interfaces.properties
    compute!(properties.gas_exchange_state.wind_speed)
    compute!(properties.gas_exchange_state.atmospheric_pressure)
    exchangers = BGCExt.gas_transfer_parametrisations(biogeochemistry, properties)
    required_ocean_tracers = (:T, :S, BGCExt.if_phosphate_available(ocean.model.tracers)...,
                              required_biogeochemical_tracers(underlying.oxygen)...,
                              required_biogeochemical_tracers(underlying.inorganic_carbon)...)
    ocean_tracers = ocean.model.tracers[required_ocean_tracers]
    fluxes = BGCExt.tracer_fluxes(biogeochemistry, coupled_model.interfaces.net_fluxes.ocean)
    map(fluxes, exchangers) do flux, exchanger
        launch!(architecture(grid), grid, :xy, BGCExt.compute_gas_exchange!,
                flux, grid, ocean.model.clock, ZeroField(eltype(grid)), ocean_tracers, exchanger)
    end
    return nothing
end
