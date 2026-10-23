# Mauna Loa atmospheric CO₂, copied from GlobalOcean (NumericalOcean, src/mauna_loa.jl at 291416f, itself from
# OAEMIP/src/mlo_pco2.jl), with the `forcing_period` its `MaunaLoaCO₂` defaults to (from src/forcing_cycles.jl)

using CSV: CSV
using Dates: Dates, DateTime, Month
using Downloads: Downloads

using Oceananigans.Architectures: AbstractArchitecture, CPU
using Oceananigans.DistributedComputations: @root
using Oceananigans: Oceananigans
using Oceananigans.Fields: Field, interior
using Oceananigans.Grids: RectilinearGrid, Flat
using Oceananigans.OutputReaders: FieldTimeSeries, Cyclical

using NumericalEarth.DataWrangling: DataWrangling, Metadata, Metadatum, all_dates,
                                    dataset_variable_name, metadata_path, metadata_url,
                                    native_times
using NumericalEarth.DataWrangling.JRA55: MultiYearJRA55, RepeatYearJRA55
using Oceananigans.Units: days

#####
##### The JRA55 forcing period (GlobalOcean's src/forcing_cycles.jl)
#####

period_to_seconds(p) = Dates.value(Dates.Millisecond(p)) / 1000

"""
    forcing_period(dataset, start_date, end_date)

The period (seconds) over which `forced_ocean_simulation` cycles the JRA55 `dataset`: the calendar span from
`start_date` to `end_date` for `MultiYearJRA55` (21915 days for 1958--2018), one 365-day year for `RepeatYearJRA55`.

Left to infer it, a multi-year series cycles over its nodes plus one interval, and its nodes run to the first on or
after `end_date`: from 1958 to 2018 that is 21915 days + 3 hours for the 3-hourly variables and 21916 days for the
daily rivers and icebergs, so the rivers slip 0.875 days a cycle against the atmosphere and both slip against the
calendar. With this period the node past `end_date` is only ever the far end of the last interval before the wrap.
"""
forcing_period(::MultiYearJRA55, start_date, end_date) = period_to_seconds(end_date - start_date)
forcing_period(::RepeatYearJRA55, start_date, end_date) = 365days

#####
##### The dataset
#####

const MAUNA_LOA_URL = "https://pastebin.com/raw/kEa6vn7v"

const MEAN_MONTH_LENGTH = 365.2425 * 86400 / 12 # s, the mean Gregorian month

"""
    MaunaLoa(; url = MAUNA_LOA_URL)

The Scripps monthly mean atmospheric CO₂ record from Mauna Loa (`:carbon_dioxide`), with the
South Pole (`:south_pole_carbon_dioxide`) and two-station mean (`:mean_carbon_dioxide`) series
carried in the same file.

Values are **dry-air mole fractions in ppm** (µmol/mol) on the Scripps '12' calibration scale,
seasonally detrended, and stamped at midnight on the 15th of each month. They are *not* partial
pressures: no sea-level-pressure or water-vapour correction is applied.

The record runs from March 1958 to March 2024. Months the stations did not observe are filled
from a seasonally detrended spline fit; the file's `Flag` column records which, and is ignored
here.

`url` must be a *raw* URL serving the CSV itself — for pastebin, `https://pastebin.com/raw/<id>`
rather than the page that wraps it.

See [`MaunaLoaCO₂`](@ref) for the time series a simulation consumes.
"""
struct MaunaLoa{U}
    url :: U
end

MaunaLoa(; url = MAUNA_LOA_URL) = MaunaLoa(url)

Base.show(io::IO, ::MaunaLoa) = print(io, "MaunaLoa monthly mean atmospheric CO₂")
Base.summary(::MaunaLoa) = "MaunaLoa monthly mean atmospheric CO₂"

const MaunaLoaMetadata = Metadata{<:MaunaLoa}
const MaunaLoaMetadatum = Metadatum{<:MaunaLoa}

const MAUNA_LOA_VARIABLES = Dict(:carbon_dioxide            => "MLO",
                                 :south_pole_carbon_dioxide => "SPO",
                                 :mean_carbon_dioxide       => "Average")

DataWrangling.all_dates(::MaunaLoa, args...) = DateTime(1958, 1, 15) : Month(1) : DateTime(2024, 3, 15)

DataWrangling.default_download_directory(::MaunaLoa) = DataWrangling.download_cache("MaunaLoa")

DataWrangling.available_variables(::MaunaLoa) = MAUNA_LOA_VARIABLES
DataWrangling.dataset_variable_name(metadata::MaunaLoaMetadata) = MAUNA_LOA_VARIABLES[metadata.name]
DataWrangling.metaprefix(::MaunaLoaMetadata) = "MaunaLoaMetadata"

DataWrangling.metadata_filename(::MaunaLoa, name, date, region) = "mlo_spo_monthly_mean.csv"
DataWrangling.metadata_url(metadata::MaunaLoaMetadata) = metadata.dataset.url

# 0D dataset: no spatial dimensions
DataWrangling.is_three_dimensional(::MaunaLoaMetadata) = false
DataWrangling.dataset_location(::MaunaLoa, name) = (Nothing, Nothing, Nothing)
DataWrangling.default_inpainting(::MaunaLoaMetadata) = nothing
Base.size(::MaunaLoa, args...) = (0, 0, 0)

function Downloads.download(metadata::MaunaLoaMetadata; kwargs...)
    filepath = metadata_path(first(metadata))

    @root if !isfile(filepath)
        url = metadata_url(metadata)

        if !startswith(url, "http")
            throw(ArgumentError("MaunaLoa has no download URL: set `MAUNA_LOA_URL` in src/mlo_pco2.jl, " *
                                "construct the dataset as `MaunaLoa(url = \"https://pastebin.com/raw/<id>\")`, " *
                                "or place the CSV at $filepath yourself"))
        end

        @info "Downloading Mauna Loa CO₂ record in $(metadata.dir)..."
        Downloads.download(url, filepath; kwargs...)
    end

    return filepath
end

#####
##### Reading the CSV
#####

"""
    read_mauna_loa(metadata::MaunaLoaMetadata)

Return the `(dates, values)` of the series `metadata` names, read from the downloaded CSV.
"""
function read_mauna_loa(metadata::MaunaLoaMetadata)
    path = metadata_path(first(metadata))
    column = Symbol(dataset_variable_name(metadata))

    file = CSV.File(path; comment = "%")

    dates  = [DateTime(row.Year, row.Month, 15) for row in file]
    values = [getproperty(row, column) for row in file]

    # Pad Jan/Feb 1958 with the first observed value (Mar 1958)
    dates  = [DateTime(1958, 1, 15), DateTime(1958, 2, 15), dates...]
    values = [values[1], values[1], values...]

    return dates, values
end

#####
##### NumericalEarth interface: native_grid, retrieve_data, Field, FieldTimeSeries
#####

function DataWrangling.native_grid(metadata::MaunaLoaMetadata, arch=CPU(); halo=(3, 3, 3))
    FT = eltype(metadata)
    return RectilinearGrid(arch, FT; size = (), topology = (Flat, Flat, Flat))
end

function DataWrangling.retrieve_data(metadata::MaunaLoaMetadatum)
    all_dates_vals, all_values = read_mauna_loa(metadata)
    date = metadata.dates
    idx = findfirst(==(date), all_dates_vals)
    isnothing(idx) && throw(ArgumentError("Date $date not found in Mauna Loa record"))
    return fill(Float64(all_values[idx]))
end

"""
    mauna_loa_co2(date; dataset = MaunaLoa(), name = :carbon_dioxide)

Return the scalar CO₂ value (ppm) at `date` as a plain `Float64`.
"""
function mauna_loa_co2(date::Dates.AbstractDateTime;
                       dataset = MaunaLoa(),
                       name = :carbon_dioxide,
                       dir = nothing)

    metadata = if isnothing(dir)
        Metadatum(name; dataset, date)
    else
        Metadatum(name; dataset, date, dir)
    end

    Downloads.download(metadata)
    all_dates_vals, all_values = read_mauna_loa(metadata)
    idx = findfirst(==(date), all_dates_vals)
    isnothing(idx) && throw(ArgumentError("Date $date not found in Mauna Loa record"))
    return Float64(all_values[idx])
end

function Oceananigans.Fields.Field(metadata::MaunaLoaMetadatum, arch::AbstractArchitecture=CPU();
                                   inpainting = nothing, mask = nothing,
                                   halo = (3, 3, 3), cache_inpainted_data = false)
    Downloads.download(metadata)
    grid = DataWrangling.native_grid(metadata, arch; halo)
    field = Field{Nothing, Nothing, Nothing}(grid)
    data = DataWrangling.retrieve_data(metadata)
    interior(field) .= data
    return field
end

"""
    MaunaLoaCO₂(architecture = CPU();
                dataset = MaunaLoa(),
                name = :carbon_dioxide,
                start_date = DateTime(1958, 1, 1),
                end_date = DateTime(2018, 1, 1),
                dates = filter(date -> start_date ≤ date < end_date, all_dates(dataset, name)),
                dir = nothing,
                FT = Float64,
                period = forcing_period(MultiYearJRA55(), start_date, end_date))

Download the Scripps [`MaunaLoa`](@ref) record and return atmospheric CO₂ dry-air mole fraction
(ppm) as a `FieldTimeSeries` located at `(Nothing, Nothing, Nothing)` — a single value per time,
uniform in space — on a `Flat, Flat, Flat` grid. Sample it like any other `FieldTimeSeries`:

```julia
using Dates, Oceananigans

xCO₂ = MaunaLoaCO₂(arch; start_date = DateTime(1958, 3, 15))

xCO₂[1, 1, 1, Time(clock.time)] # ppm at the model's current time
```

`start_date` is the date model time zero represents — the simulation's own start date, not the
record's, whenever the two differ.

The samples are placed at evenly spaced mid-month times (`times` is a range), starting half a
month after the first day of the first selected month, rather than on the record's 15th-of-the-month
dates. This keeps `times` a range on the GPU, so interpolating the series does not copy it to the
host each time step; each sample is within ~2 days of its month's actual midpoint.

Time indexing is `Cyclical`. By default it takes the months from `start_date` to `end_date` and cycles them with
the multi-year JRA55 forcing `forced_ocean_simulation` builds for the same dates (21915 days for 1958--2018, see
[`forcing_period`](@ref)), so the CO₂ wraps back to 1958 with the atmosphere. The samples are spread evenly over
the `period`, `period / Nt` apart (30.4375 days for 1958--2018), with the same spacing across the seam. Pass e.g.
`period = 365days` with one year of `dates` to cycle in step with repeat-year forcing, or `period = nothing` for the
mean Gregorian month (365.2425 / 12 days) spacing and a period of that times the number of samples.

Keyword Arguments
=================

- `dataset`: the [`MaunaLoa`](@ref) dataset, carrying the URL the CSV is fetched from.
- `name`: which series to read — `:carbon_dioxide` (Mauna Loa, the default),
          `:south_pole_carbon_dioxide`, or `:mean_carbon_dioxide`.
- `start_date`: the date corresponding to model time zero. Default: 1958-01-01, as `forced_ocean_simulation`.
- `end_date`: the end of the forcing cycle. Default: 2018-01-01, as `forced_ocean_simulation`.
- `dates`: the window of the record to use. Default: the months from `start_date` up to `end_date`.
- `dir`: where the CSV is cached. Default: the `NumericalEarth` download cache.
- `FT`: element type of the returned series. Default: `Float64`.
- `period`: cycle length in seconds, or `nothing` for the span of `dates` plus one final mean month.
            Default: the multi-year JRA55 forcing's period from `start_date` to `end_date`.
"""
function MaunaLoaCO₂(architecture::AbstractArchitecture = CPU();
                     dataset = MaunaLoa(),
                     name = :carbon_dioxide,
                     start_date = DateTime(1958, 1, 1),
                     end_date = DateTime(2018, 1, 1),
                     dates = filter(date -> start_date ≤ date < end_date, all_dates(dataset, name)),
                     dir = nothing,
                     FT = Float64,
                     period = forcing_period(MultiYearJRA55(), start_date, end_date))

    metadata = if isnothing(dir)
        Metadata(name; dataset, dates)
    else
        Metadata(name; dataset, dates, dir)
    end

    Downloads.download(metadata)

    file_dates, file_values = read_mauna_loa(metadata)

    first_date, last_date = first(dates), last(dates)
    within_window = [first_date ≤ date ≤ last_date for date in file_dates]

    selected_dates  = file_dates[within_window]
    selected_values = file_values[within_window]

    length(selected_dates) < 2 &&
        throw(ArgumentError("The Mauna Loa record needs at least two dates to interpolate, got $(length(selected_dates))"))

    seconds_since_start(date) = Dates.value(Dates.Millisecond(date - start_date)) / 1000

    values = FT[value for value in selected_values]
    Nt = length(selected_dates)

    if isnothing(period)
        Δt = convert(FT, MEAN_MONTH_LENGTH)
        period = Nt * Δt
    else
        period = convert(FT, period)
        Δt = period / Nt
    end

    t₀ = convert(FT, seconds_since_start(Dates.firstdayofmonth(first(selected_dates)))) + Δt / 2
    times = range(t₀; step = Δt, length = Nt)

    grid = RectilinearGrid(architecture, FT; size = (), topology = (Flat, Flat, Flat))

    xCO₂ = FieldTimeSeries{Nothing, Nothing, Nothing}(grid, times; time_indexing = Cyclical(period))

    copyto!(interior(xCO₂), reshape(values, 1, 1, 1, :))

    return xCO₂
end
