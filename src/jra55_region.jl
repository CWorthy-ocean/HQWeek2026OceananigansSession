# # Multi-year JRA55-do cut to the California Current region, as a NumericalEarth dataset
#
# `RegionalJRA55()` is the regional JRA55-do that metal/subset_jra55.jl cuts on the cluster (the series
# `JRA55PrescribedAtmosphere`, `JRA55PrescribedRadiation` and `JRA55PrescribedLand` read from the global files, on
# JRA55's own grid cut to the region, 2010–2012), so NumericalEarth's JRA55 constructors load it as they load JRA55:
#
#     include("jra55_region.jl")
#     atmosphere = JRA55PrescribedAtmosphere(arch; dataset = RegionalJRA55(), start_date, end_date)
#     radiation  = JRA55PrescribedRadiation(arch;  dataset = RegionalJRA55(), start_date, end_date)
#     land       = JRA55PrescribedLand(grid;       dataset = RegionalJRA55(), start_date, end_date)
#
# Each variable is one file, `jra55_<name>.jld2` in `jra55_region_directory`, holding a FieldTimeSeries on its JRA55 grid
# with times in seconds since 2010-01-01; the dates, grid and data come from the files. They are not downloadable: make
# them on the cluster with metal/subset_jra55.jl.

using NumericalEarth.DataWrangling: DataWrangling, Metadata, Metadatum, MetadataSet
using NumericalEarth.DataWrangling: JLD2
using Oceananigans: Oceananigans, Center, FieldTimeSeries, OnDisk, λnodes, φnodes
using Oceananigans.Architectures: on_architecture
using Dates: DateTime, Millisecond
import Downloads

struct RegionalJRA55 end

const jra55_region_directory = joinpath(@__DIR__, "..", "data", "JRA55_region")
const jra55_region_first_date = DateTime(2010, 1, 1)   # metal/subset_jra55.jl's JRA55_FIRST_DATE, the files' time 0

const RegionalJRA55Metadatum = Metadatum{<:RegionalJRA55}

const REGIONAL_JRA55_VARIABLES = Dict(name => string(name) for name in
    (:eastward_velocity, :northward_velocity, :temperature, :specific_humidity, :sea_level_pressure,
     :rain_freshwater_flux, :snow_freshwater_flux, :downwelling_shortwave_radiation, :downwelling_longwave_radiation,
     :river_freshwater_flux, :iceberg_freshwater_flux))

DataWrangling.available_variables(::RegionalJRA55) = REGIONAL_JRA55_VARIABLES
DataWrangling.dataset_variable_name(metadata::Metadata{<:RegionalJRA55}) = REGIONAL_JRA55_VARIABLES[metadata.name]
DataWrangling.default_download_directory(::RegionalJRA55) = jra55_region_directory
DataWrangling.metadata_filename(::RegionalJRA55, name, date, region) = "jra55_$(name).jld2"
DataWrangling.default_inpainting(::Metadata{<:RegionalJRA55}) = nothing
DataWrangling.is_three_dimensional(::Metadata{<:RegionalJRA55}) = false

# The files are always read from `jra55_region_directory` (NumericalEarth's JRA55 constructors pass JRA55's own
# directory as `dir`)
regional_jra55_path(name) = joinpath(jra55_region_directory, "jra55_$(name).jld2")
DataWrangling.metadata_path(metadatum::RegionalJRA55Metadatum) = regional_jra55_path(metadatum.name)
DataWrangling.metadata_path(metadata::Metadata{<:RegionalJRA55}) = [regional_jra55_path(metadata.name) for _ in metadata]

# Each file is opened once (its grid and times are read on every call below)
const regional_jra55_series_cache = Dict{Symbol, Any}()
regional_jra55_series(name) = get!(() -> FieldTimeSeries(regional_jra55_path(name), string(name); backend = OnDisk()),
                                   regional_jra55_series_cache, name)

# The dates of the saved times (evenly spaced, so the series' times stay a range)
function DataWrangling.all_dates(::RegionalJRA55, name)
    times = regional_jra55_series(name).times
    step = Millisecond(round(Int, 1000 * (times[2] - times[1])))
    return range(jra55_region_first_date + Millisecond(round(Int, 1000 * first(times))); step, length = length(times))
end

# The saved series' own grid and location
Base.size(::RegionalJRA55, name) = size(regional_jra55_series(name).grid)
Base.size(metadatum::RegionalJRA55Metadatum) = (size(metadatum.dataset, metadatum.name)..., 1)
DataWrangling.dataset_location(::RegionalJRA55, name) = (Center, Center, Center)
DataWrangling.construct_native_grid(metadata::Metadata{<:RegionalJRA55}, ::Nothing, arch; halo) =
    on_architecture(arch, regional_jra55_series(metadata.name).grid)

# One time's interior
function DataWrangling.retrieve_data(metadatum::RegionalJRA55Metadatum)
    series = regional_jra55_series(metadatum.name)
    n = round(Int, (Millisecond(metadatum.dates - jra55_region_first_date).value / 1000 - first(series.times)) /
                   (series.times[2] - series.times[1])) + 1
    return Array(Oceananigans.interior(series[n]))[:, :, 1]
end

function DataWrangling.read_file_coords(metadatum::RegionalJRA55Metadatum)
    grid = regional_jra55_series(metadatum.name).grid
    return Array(λnodes(grid, Center())), Array(φnodes(grid, Center()))
end
DataWrangling.mangling_for(::RegionalJRA55Metadatum, data_latitude_count) = nothing

# Not downloadable: check the file is there
function Downloads.download(metadata::Metadata{<:RegionalJRA55}; kwargs...)
    path = regional_jra55_path(metadata.name)
    isfile(path) || error("$path is missing; make it with metal/subset_jra55.jl")
    return DataWrangling.metadata_path(metadata)
end
