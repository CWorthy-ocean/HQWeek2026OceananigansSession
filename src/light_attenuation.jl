# # Chlorophyll-dependent PAR attenuation
#
# From NumericalOcean/src/light_attenuation.jl (itself from OAEMIP), for `PrescribedAttenuationPAR` with
# `attenuation_discrete_form = true`. NumericalOcean reads the chlorophyll from `model.forcing.T`; here it is stored,
# with the optics, from the ocean's `TwoColorRadiation`, so the BGC light sees the same water as the heat budget.

using Oceananigans.TimeSteppers: Adapt # not a direct dependency of this project
using Oceananigans.Grids: znode
using Oceananigans.Units: Time
using NumericalEarth.EarthSystemModels.InterfaceComputations: state2dindex

using NumericalEarth: TwoColorRadiation

struct PrescribedChlorophyllAttenuation{FT, C} <: Function # `DiscreteForcing` only calls `Function`s
       first_color_fraction :: FT
    first_decay_coefficient :: FT
    clear_water_attenuation :: FT
        chlorophyll_scaling :: FT
       chlorophyll_exponent :: FT
                chlorophyll :: C
end

"""
    PrescribedChlorophyllAttenuation(radiation::TwoColorRadiation; first_color_fraction = 0)

PAR attenuation (m⁻¹) from the chlorophyll (mg m⁻³) and optics of the ocean's `radiation`. The blue-green
attenuation is ``κ₂ = a + b C^n``; a red fraction `first_color_fraction` decays with `radiation`'s red coefficient,
and is zero by default as PAR is already the photosynthetic fraction of the shortwave (as NumericalOcean's 1degree
example).
"""
function PrescribedChlorophyllAttenuation(radiation::TwoColorRadiation; first_color_fraction = 0)
    FT = typeof(radiation.first_color_fraction)
    optics = radiation.chlorophyll_optics
    return PrescribedChlorophyllAttenuation(convert(FT, first_color_fraction),
                                            radiation.first_absorption_coefficient,
                                            optics.clear_water_attenuation,
                                            optics.chlorophyll_scaling,
                                            optics.chlorophyll_exponent,
                                            radiation.chlorophyll)
end

Adapt.adapt_structure(to, pca::PrescribedChlorophyllAttenuation) =
    PrescribedChlorophyllAttenuation(pca.first_color_fraction,
                                     pca.first_decay_coefficient,
                                     pca.clear_water_attenuation,
                                     pca.chlorophyll_scaling,
                                     pca.chlorophyll_exponent,
                                     Adapt.adapt(to, pca.chlorophyll))

@inline function log_light_fraction(z, f₁, κ₁, κ₂)
    κ = min(κ₁, κ₂)
    return κ * z + log(f₁ * exp((κ₁ - κ) * z) + (1 - f₁) * exp((κ₂ - κ) * z))
end

# The discrete form: the last argument is OceanBioME's `chlorophyll(bgc, model)`, a `ZeroField` for
# `ImplicitProductivity`, unused
@inline function (optics::PrescribedChlorophyllAttenuation)(i, j, k, grid, clock, bgc_chlorophyll)
    C = @inbounds state2dindex(optics.chlorophyll, i, j, grid, Time(clock.time))

    f₁ = optics.first_color_fraction
    κ₁ = optics.first_decay_coefficient
    κw = optics.clear_water_attenuation
    Cs = optics.chlorophyll_scaling
    Ce = optics.chlorophyll_exponent

    κ₂ = κw + Cs * max(0, C)^Ce

    z₁ = znode(i, j, k+1, grid, Center(), Center(), Face())
    z₂ = znode(i, j, k, grid, Center(), Center(), Face())

    return (log_light_fraction(z₁, f₁, κ₁, κ₂) - log_light_fraction(z₂, f₁, κ₁, κ₂)) / (z₁ - z₂)
end
