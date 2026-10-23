# Download the repeat-year (RYF 1990--1991) JRA55-do forcing into data/JRA55; files already present are skipped
using NumericalEarth
using NumericalEarth.DataWrangling: Metadata, metadata_path
using NumericalEarth.DataWrangling.JRA55: RepeatYearJRA55, JRA55_variable_names
using Downloads: Downloads

dir = joinpath(@__DIR__, "..", "data", "JRA55")

for name in JRA55_variable_names
    metadata = Metadata(name; dataset = RepeatYearJRA55(), dir)
    @info "$name → $(metadata_path(metadata))"
    Downloads.download(metadata)
end
