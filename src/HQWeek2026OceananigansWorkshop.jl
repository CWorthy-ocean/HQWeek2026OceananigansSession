module HQWeek2026OceananigansWorkshop

export on_metal, barotropic_time_array, set_barotropic_time!, surface_dataset_restoring
export BGCInitClimatology, BGCInitSolubleIronDeposition, BGCInitDustIronDeposition
export MaunaLoa, MaunaLoaCO₂, mauna_loa_co2
export ORCA2Grid
export woa_to_teos10!, blend_monthly_initial_condition!, woa_salinity_fts_to_teos10!
export GLORYSQuarterDegree
export RegionalJRA55
export x_boundary, y_boundary, x_fixed_boundary, y_fixed_boundary, x_flather, y_flather, u_bottom_drag, v_bottom_drag, u_immersed_drag, v_immersed_drag
export sponge_coefficient
export save_restart_state, load_restart_state!
export alkalinity_release_tendency, alkalinity_release
export PrescribedChlorophyllAttenuation

# only with Metal in the environment (elsewhere `]rm Metal; ]add CUDA`)
@static if !isnothing(Base.find_package("Metal"))
    include("metal_hacks.jl")
end
include("boundary_data.jl")
include("bgc_init.jl")  # will make it to NE eventually
include("mauna_loa.jl") # will make it to NE eventually
include("orca2.jl")      # from NumericalEarth PR #153
include("woa_teos10.jl") # from OMIPSimulations, as NumericalOcean
include("glorys_quarter_degree.jl")
include("jra55_region.jl")
include("boundary_functions.jl")
include("ocean_only_gas_exchange.jl")
include("restart_state.jl")
include("alkalinity_release.jl")
include("light_attenuation.jl")  # from NumericalOcean

end # module HQWeek2026OceananigansWorkshop

