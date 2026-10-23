module HQWeek2026OceananigansWorkshop

export on_metal, barotropic_time_array, set_barotropic_time!, surface_dataset_restoring
export BGCInitClimatology, BGCInitSolubleIronDeposition, BGCInitDustIronDeposition
export MaunaLoa, MaunaLoaCO₂, mauna_loa_co2
export ORCA2Grid
export GLORYSQuarterDegree
export RegionalJRA55

include("metal_hacks.jl")
include("boundary_data.jl")
include("bgc_init.jl")  # will make it to NE eventually
include("mauna_loa.jl") # will make it to NE eventually
include("orca2.jl")      # from NumericalEarth PR #153
include("glorys_quarter_degree.jl")
include("jra55_region.jl")

end # module HQWeek2026OceananigansWorkshop

