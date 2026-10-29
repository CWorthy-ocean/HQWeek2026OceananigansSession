# # Carbonate system sensitivities for linearised ("CDR-lite") CO₂ uptake
#
# As the CDR tracers of CWorthy's ROMS (CWorthy-ocean/ucla-roms#242), whose air–sea flux is -(k / β) (C_DIC - η C_Alk):
#     η = ∂DIC/∂Alk at constant [CO₂(aq)] (`ddic_dalk` there),
#     β = ∂DIC/∂[CO₂(aq)] at constant Alk (`ddic_dco2` there),
# both dimensionless. Exact derivatives at the [H⁺] of OceanBioME's pH solve: with c = [CO₂(aq)] and h = [H⁺],
#     DIC = c (1 + K₁/h + K₁K₂/h²),   Alk = c (K₁/h + 2K₁K₂/h²) + Alk_nc(h),
# so η = (∂DIC/∂h)_c / (∂Alk/∂h)_c and β = (∂DIC/∂c)_h - η (∂Alk/∂c)_h, where Alk_nc is the non-carbonate alkalinity
# (borate, water, phosphate, silicate, sulfate and fluoride, as OceanBioME's alkalinity residual). At the surface
# pressure everywhere (the conditions the water would exchange CO₂ at), with the model's salinity as the gas exchange
# passes it.

using OceanBioME.Models.CarbonChemistryModel: CarbonChemistry, solve_for_H, ∂ₕ_alkalinity_residual
using KernelAbstractions: @kernel, @index
using Oceananigans.Architectures: architecture
import Oceananigans.Biogeochemistry: update_biogeochemical_state!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: ZeroField, compute!
using Oceananigans.Utils: launch!

"""
    carbonate_sensitivities(carbon_chemistry; DIC, Alk, T, S, phosphate = 0, silicate = 0, initial_pH_guess = 8)

Returns `(η, β)`: `η = ∂DIC/∂Alk` at constant [CO₂(aq)] and `β = ∂DIC/∂[CO₂(aq)]` at constant Alk (dimensionless),
from `DIC` (mmol m⁻³), `Alk` (meq m⁻³), `T` (°C), `S`, `phosphate` and `silicate` (mmol m⁻³), at the surface.
"""
@inline function carbonate_sensitivities(cc::CarbonChemistry; DIC::FT, Alk, T, S, phosphate = zero(DIC),
                                         silicate = zero(DIC), initial_pH_guess = convert(typeof(DIC), 8)) where FT
    ρₒ = cc.density_function(T, S, zero(DIC), zero(DIC), zero(DIC))
    T += convert(FT, 273.15)

    # mmol (or meq) m⁻³ to mol (or eq) kg⁻¹
    to_mol_per_kg = convert(FT, 1e-3) / ρₒ
    DIC *= to_mol_per_kg
    Alk *= to_mol_per_kg
    phosphate *= to_mol_per_kg
    silicate  *= to_mol_per_kg

    Is = cc.ionic_strength(S)
    K1 = cc.carbonic_acid.K1(T, S; P = nothing)
    K2 = cc.carbonic_acid.K2(T, S; P = nothing)
    KS = cc.sulfate(T, S, Is; P = nothing)

    # the totals in the salinity's float type (OceanBioME's defaults form them in Float64, which Metal rejects)
    params = (; DIC, Alk,
                boron    = convert(FT, 0.000232 / 10.811 / 1.80655) * S,
                sulfate  = convert(FT, 0.14 / 96.062 / 1.80655) * S,
                fluoride = convert(FT, 0.000067 / 18.9984 / 1.80655) * S,
                silicate, phosphate, K1, K2,
                KB  = cc.boric_acid(T, S; P = nothing),
                KW  = cc.water(T, S; P = nothing),
                KS,
                KF  = cc.fluoride(T, S, Is, KS; P = nothing),
                KP1 = cc.phosphoric_acid.KP1(T, S; P = nothing),
                KP2 = cc.phosphoric_acid.KP2(T, S; P = nothing),
                KP3 = cc.phosphoric_acid.KP3(T, S; P = nothing),
                KSi = cc.silicic_acid(T, S, Is; P = nothing))

    h = solve_for_H(nothing, params, initial_pH_guess, cc.solver)

    # ∂Alk_nc/∂h: the residual's derivative with no carbon
    ∂ₕAlk_nc = ∂ₕ_alkalinity_residual(h, merge(params, (; DIC = zero(DIC))))

    Dc = 1 + K1 / h + K1 * K2 / h^2                    # (∂DIC/∂c)_h
    Ac = K1 / h + 2 * K1 * K2 / h^2                    # (∂Alk/∂c)_h
    c  = DIC / Dc
    Dh = - c * (K1 / h^2 + 2 * K1 * K2 / h^3)          # (∂DIC/∂h)_c
    Ah = - c * (K1 / h^2 + 4 * K1 * K2 / h^3) + ∂ₕAlk_nc # (∂Alk/∂h)_c

    η = Dh / Ah
    β = Dc - η * Ac

    return η, β
end

#####
##### Stored fields, both from one pH solve per cell, computed from a model's tracers
#####

"""
    CarbonateSensitivities(grid; carbon_chemistry = CarbonChemistry(eltype(grid)))

η (`.η`) and β (`.β`) of [`carbonate_sensitivities`](@ref) as `CenterField`s on `grid`, computed by
`compute!(sensitivities, model)` from the `model`'s `DIC` and `Alk` (or `DIC1` and `Alk1`, the control of a
`CarbonateSystem(N)`), `T`, `S`, and `PO₄` if it has one. Needs only the grid, so a biogeochemical model
can own it and compute it in its `update_biogeochemical_state!` (as `update_biogeochemical_state!(model, sensitivities)`
does, e.g. as an OceanBioME modifier); as a callback, `add_callback!(ocean, sensitivities, IterationInterval(1))`
computes it after every ocean step, before the ocean's output writers. Every cell, so a pH solve per cell.
"""
struct CarbonateSensitivities{E, B, C}
    η :: E
    β :: B
    carbon_chemistry :: C
end

CarbonateSensitivities(grid; carbon_chemistry = CarbonChemistry(eltype(grid))) =
    CarbonateSensitivities(CenterField(grid), CenterField(grid), carbon_chemistry)

# `DIC` and `Alk`, or with N copies the first
control_carbon(tracers::NamedTuple{names}) where names = :DIC in names ? (tracers.DIC, tracers.Alk) : (tracers.DIC1, tracers.Alk1)

@kernel function _compute_carbonate_sensitivities!(η, β, cc, DIC, Alk, T, S, PO₄)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        η[i, j, k], β[i, j, k] = carbonate_sensitivities(cc; DIC = DIC[i, j, k], Alk = Alk[i, j, k], T = T[i, j, k],
                                                         S = S[i, j, k], phosphate = PO₄[i, j, k])
    end
end

function Oceananigans.Fields.compute!(s::CarbonateSensitivities, model)
    grid = s.η.grid
    tracers = model.tracers
    PO₄ = get(tracers, :PO₄, ZeroField(eltype(grid)))
    launch!(architecture(grid), grid, :xyz, _compute_carbonate_sensitivities!, s.η, s.β, s.carbon_chemistry,
            control_carbon(tracers)..., tracers.T, tracers.S, PO₄)
    fill_halo_regions!((s.η, s.β))
    return s
end

update_biogeochemical_state!(model, s::CarbonateSensitivities) = compute!(s, model)

(s::CarbonateSensitivities)(sim) = compute!(s, sim.model)
