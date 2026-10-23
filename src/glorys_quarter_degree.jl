# # GLORYS daily means on the 1/4° Metal grid, as a NumericalEarth dataset
#
# `GLORYSQuarterDegree()` is the daily 3-D GLORYS state that metal/regrid_glorys_3d.jl interpolates onto the 1/4° Metal
# grid (134–115.5°W, 30–48°N, 42 levels; inpainted, 2010–2012), so it loads the way NumericalEarth loads any dataset,
# onto any grid inside that domain:
#
#     include("glorys_quarter_degree.jl")
#     Tᵉ = FieldTimeSeries(Metadata(:temperature; dataset = GLORYSQuarterDegree(), dates), grid; time_indices_in_memory = 3)
#
# The files are read whole (no `region`): NumericalEarth builds their grid from the hooks below, fills it from the file
# and interpolates onto `grid`. u and v are on the faces they were saved on. The files are already inpainted, so
# NumericalEarth does not inpaint them. They are not downloadable: make them on the cluster with
# metal/regrid_glorys_3d.jl and put them in `glorys_quarter_degree_directory`.

using NumericalEarth.DataWrangling: DataWrangling, Metadata, Metadatum, metadata_path
using NumericalEarth.DataWrangling: JLD2
using Oceananigans: Oceananigans, Center, Face, FieldTimeSeries, OnDisk, LatitudeLongitudeGrid, Bounded, Flat, znodes
using Dates: DateTime, Day, value
import Downloads

struct GLORYSQuarterDegree end

const glorys_quarter_degree_directory = joinpath(@__DIR__, "..", "..", "OceananigansHQWeek2026", "data", "GLORYS", "glorys_3d")
const glorys_quarter_degree_first_date = DateTime(2010, 1, 1)   # metal/regrid_glorys_3d.jl's GLORYS_FIRST_DATE

const GLORYSQuarterDegreeMetadatum = Metadatum{<:GLORYSQuarterDegree}

# Variable names in the files (one file and series per variable)
const GLORYS_QUARTER_DEGREE_VARIABLES = Dict(:temperature => "T", :salinity => "S", :u_velocity => "u",
                                             :v_velocity => "v", :free_surface => "η")

DataWrangling.available_variables(::GLORYSQuarterDegree) = GLORYS_QUARTER_DEGREE_VARIABLES
DataWrangling.dataset_variable_name(metadata::Metadata{<:GLORYSQuarterDegree}) = GLORYS_QUARTER_DEGREE_VARIABLES[metadata.name]
DataWrangling.default_download_directory(::GLORYSQuarterDegree) = glorys_quarter_degree_directory
DataWrangling.metadata_filename(::GLORYSQuarterDegree, name, date, region) = "glorys_$(GLORYS_QUARTER_DEGREE_VARIABLES[name])_3d.jld2"
DataWrangling.all_dates(::GLORYSQuarterDegree, name) = glorys_quarter_degree_first_date:Day(1):DateTime(2012, 12, 31)
DataWrangling.averaging_window(metadatum::GLORYSQuarterDegreeMetadatum) = (metadatum.dates, metadatum.dates + Day(1))
DataWrangling.default_inpainting(::Metadata{<:GLORYSQuarterDegree}) = nothing

DataWrangling.dataset_location(::GLORYSQuarterDegree, name) =
    name === :u_velocity   ? (Face, Center, Center) :
    name === :v_velocity   ? (Center, Face, Center) :
    name === :free_surface ? (Center, Center, Nothing) : (Center, Center, Center)

DataWrangling.is_three_dimensional(metadata::Metadata{<:GLORYSQuarterDegree}) = metadata.name !== :free_surface

# The saved grid (metal/regrid_glorys_3d.jl): 74 × 72 cells of 1/4°, the 42 levels of the Metal scripts
Base.size(::GLORYSQuarterDegree, name) = (74, 72, 42)
Base.size(::GLORYSQuarterDegreeMetadatum) = (74, 72, 42, 1)
DataWrangling.longitude_interfaces(::Metadata{<:GLORYSQuarterDegree}) = (-134, -115.5)
DataWrangling.latitude_interfaces(::Metadata{<:GLORYSQuarterDegree}) = (30, 48)

saved_grid(metadata) = FieldTimeSeries(joinpath(metadata.dir, "glorys_T_3d.jld2"), "T"; backend = OnDisk()).grid
DataWrangling.z_interfaces(metadata::Metadata{<:GLORYSQuarterDegree}) = Array(znodes(saved_grid(metadata), Face()))

# Bounded in longitude like the saved grid (NumericalEarth makes a 2-D grid without a region periodic in longitude)
function DataWrangling.construct_native_grid(metadata::Metadata{<:GLORYSQuarterDegree}, ::Nothing, arch; halo)
    longitude, latitude = DataWrangling.longitude_interfaces(metadata), DataWrangling.latitude_interfaces(metadata)
    Nx, Ny, Nz = size(metadata)
    DataWrangling.is_three_dimensional(metadata) ||
        return LatitudeLongitudeGrid(arch, eltype(metadata); size = (Nx, Ny), halo = halo[1:2], longitude, latitude,
                                     topology = (Bounded, Bounded, Flat))
    return LatitudeLongitudeGrid(arch, eltype(metadata); size = (Nx, Ny, Nz), halo, longitude, latitude,
                                 z = DataWrangling.z_interfaces(metadata), topology = (Bounded, Bounded, Bounded))
end

# One day's interior, from the series saved with one halo cell (none in z for η)
function DataWrangling.retrieve_data(metadatum::GLORYSQuarterDegreeMetadatum)
    n = value(Day(metadatum.dates - glorys_quarter_degree_first_date)) + 1
    name = DataWrangling.dataset_variable_name(metadatum)
    data = JLD2.jldopen(file -> file["timeseries/$name/$n"], metadata_path(metadatum))
    return DataWrangling.is_three_dimensional(metadatum) ? data[2:end-1, 2:end-1, 2:end-1] : data[2:end-1, 2:end-1, 1]
end

# The data are on the grid above, so no file coordinates or latitude remapping are needed
function DataWrangling.read_file_coords(metadatum::GLORYSQuarterDegreeMetadatum)
    grid = saved_grid(metadatum)
    return Array(Oceananigans.λnodes(grid, Center())), Array(Oceananigans.φnodes(grid, Center()))
end
DataWrangling.mangling_for(::GLORYSQuarterDegreeMetadatum, data_latitude_count) = nothing

# Not downloadable: check the files are there
function Downloads.download(metadata::Metadata{<:GLORYSQuarterDegree})
    paths = unique(metadata_path(metadata) isa AbstractVector ? metadata_path(metadata) : [metadata_path(metadata)])
    absent = filter(!isfile, paths)
    isempty(absent) || error("GLORYSQuarterDegree files are missing: $(join(absent, ", ")); make them with metal/regrid_glorys_3d.jl")
    return metadata_path(metadata)
end
