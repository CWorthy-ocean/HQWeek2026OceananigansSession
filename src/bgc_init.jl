# BGC init climatology (`bgc_init.nc`, the OAEMIP 1° monthly climatology) as a NumericalEarth dataset, copied from
# GlobalOcean (NumericalOcean, src/unified_bgc.jl at 291416f). `bgc_init.nc` is not downloadable: it is read from
# `bgc_init_directory[]` (default: the `BGC_INIT_DIRECTORY` environment variable, else "data").

using Dates: Dates, DateTime, Month
using Downloads: Downloads
using NCDatasets: NCDataset

using Oceananigans.Architectures: AbstractArchitecture, CPU
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Fields: Center
using Oceananigans.OutputReaders: Cyclical, FieldTimeSeries

using NumericalEarth.DataWrangling: DataWrangling, Metadata, Metadatum, metadata_path,
                                    dataset_variable_name, reversed_vertical_axis

import NumericalEarth.DataWrangling: convert_units

const bgc_init_directory = Ref(get(ENV, "BGC_INIT_DIRECTORY", "data"))

struct BGCInitClimatology end

const BGCInitMetadata{D} = Metadata{<:BGCInitClimatology, D}
const BGCInitMetadatum   = Metadatum{<:BGCInitClimatology}

const BGC_INIT_VARIABLES = Dict(
    :nitrate                      => "NO3",
    :phosphate                    => "PO4",
    :silicate                     => "SiO3",
    :dissolved_oxygen             => "O2",
    :dissolved_iron               => "Fe",
    :dissolved_inorganic_carbon   => "DIC",
    :alkalinity                   => "Alk",
    :nitrous_oxide                => "N2O",
    :chlorophyll                  => "CHL",
    :dissolved_organic_carbon     => "DOC",
    :refractory_DOC               => "DOCr",
    :dissolved_organic_nitrogen   => "DON",
    :refractory_DON               => "DONr",
    :dissolved_organic_phosphorus => "DOP",
    :refractory_DOP               => "DOPr",
    :ammonium                     => "NH4",
    :small_phyto_carbon           => "spC",
    :small_phyto_phosphorus       => "spP",
    :small_phyto_chlorophyll      => "spChl",
    :small_phyto_iron             => "spFe",
    :coccolithophore_CaCO3        => "spCaCO3",
    :diatom_carbon                => "diatC",
    :diatom_phosphorus            => "diatP",
    :diatom_chlorophyll           => "diatChl",
    :diatom_iron                  => "diatFe",
    :diatom_silicon               => "diatSi",
    :diazotroph_carbon            => "diazC",
    :diazotroph_phosphorus        => "diazP",
    :diazotroph_chlorophyll       => "diazChl",
    :diazotroph_iron              => "diazFe",
    :zooplankton_carbon           => "zooC",
    :temperature                  => "temp_WOA",
    :salinity                     => "salt_WOA",
    :soluble_iron_deposition      => "iron",
    :dust_deposition              => "dust",
)

# Model tracer name => BGCInitClimatology variable, for the tracers it can initialize
const BGC_INIT_TRACERS = Dict(
    :NO₃ => :nitrate,
    :NH₄ => :ammonium,
    :PO₄ => :phosphate,
    :Si  => :silicate,
    :Fe  => :dissolved_iron,
    :O₂  => :dissolved_oxygen,
    :DIC => :dissolved_inorganic_carbon,
    :Alk => :alkalinity,
    :DOC => :dissolved_organic_carbon,
    :DON => :dissolved_organic_nitrogen,
    :DOP => :dissolved_organic_phosphorus,
)

const BGC_INIT_2D_VARIABLES = Set([:soluble_iron_deposition, :dust_deposition])
const BGC_INIT_SEASONAL_VARIABLES = Set([:dust_deposition])

# `bgc_init.nc` is not downloadable; it is expected in `bgc_init_directory[]`
DataWrangling.default_download_directory(::BGCInitClimatology) = bgc_init_directory[]

function DataWrangling.all_dates(::BGCInitClimatology, name)
    if name in BGC_INIT_SEASONAL_VARIABLES
        return [DateTime(2001, m, 1) for m in (1, 4, 7, 10)]
    else
        return [DateTime(2001, m, 1) for m in 1:12]
    end
end

function DataWrangling.averaging_window(metadatum::BGCInitMetadatum)
    if metadatum.name in BGC_INIT_SEASONAL_VARIABLES
        quarter_start = DateTime(Dates.year(metadatum.dates),
                                 Dates.month(metadatum.dates), 1)
        return (quarter_start, quarter_start + Month(3))
    else
        return DataWrangling.calendar_month_window(metadatum)
    end
end

DataWrangling.metadata_filename(::BGCInitClimatology, name, date, region) = "bgc_init.nc"

DataWrangling.available_variables(::BGCInitClimatology) = BGC_INIT_VARIABLES
DataWrangling.dataset_variable_name(data::BGCInitMetadata) = BGC_INIT_VARIABLES[data.name]

DataWrangling.is_three_dimensional(metadata::BGCInitMetadata) = !(metadata.name in BGC_INIT_2D_VARIABLES)

function DataWrangling.dataset_location(::BGCInitClimatology, name)
    if name in BGC_INIT_2D_VARIABLES
        return (Center, Center, Nothing)
    else
        return (Center, Center, Center)
    end
end

DataWrangling.longitude_interfaces(::BGCInitClimatology) = (-180, 180)
DataWrangling.latitude_interfaces(::BGCInitClimatology) = (-90, 90)
DataWrangling.longitude_name(::BGCInitMetadata) = "longitude"
DataWrangling.latitude_name(::BGCInitMetadata) = "latitude"

DataWrangling.reversed_vertical_axis(::BGCInitClimatology) = true

DataWrangling.metaprefix(::BGCInitMetadata) = "BGCInitMetadata"
DataWrangling.metaprefix(::BGCInitMetadatum) = "BGCInitMetadatum"

DataWrangling.default_inpainting(::BGCInitMetadata) = nothing

function Base.size(::BGCInitClimatology, variable)
    if variable in BGC_INIT_2D_VARIABLES
        return (360, 180, 1)
    else
        return (360, 180, 102)
    end
end

#####
##### Unit conversions for surface flux variables
#####

# nmol/cm²/s → mmol Fe/m²/s, negative into ocean
# 1 nmol/cm²/s = 1e-9 mol / 1e-4 m² / s = 1e-5 mol/m²/s = 1e-2 mmol/m²/s
struct NmolPerCm2sToNegMmolPerM2s end
@inline convert_units(d::FT, ::NmolPerCm2sToNegMmolPerM2s) where FT = -d * convert(FT, 0.01)

# kg dust/m²/s → mmol Fe/m²/s assuming 3.5 wt% iron, negative into ocean
struct KgDustToNegMmolFe end
@inline convert_units(d::FT, ::KgDustToNegMmolFe) where FT =
    -d * convert(FT, 0.035) / convert(FT, 55.845f-6)

function DataWrangling.conversion_units(metadata::BGCInitMetadata)
    if metadata.name == :soluble_iron_deposition
        return NmolPerCm2sToNegMmolPerM2s()
    else
        return nothing
    end
end

function DataWrangling.z_interfaces(metadata::BGCInitMetadata)
    ds = NCDataset(joinpath(metadata.dir, "bgc_init.nc"))
    depth_centers = Float64.(ds["depth"][:])
    close(ds)

    N = length(depth_centers)
    faces = Vector{Float64}(undef, N + 1)
    faces[1] = 0.0
    for k in 1:N-1
        faces[k+1] = (depth_centers[k] + depth_centers[k+1]) / 2
    end
    faces[N+1] = depth_centers[N] + (depth_centers[N] - depth_centers[N-1]) / 2

    return [-faces[N + 2 - k] for k in 1:N+1]
end

Downloads.download(metadata::BGCInitMetadata) = joinpath(metadata.dir, "bgc_init.nc")

function DataWrangling.retrieve_data(metadatum::BGCInitMetadatum)
    path = metadata_path(metadatum)
    name = dataset_variable_name(metadatum)

    dates = DataWrangling.all_dates(metadatum.dataset, metadatum.name)
    time_idx = findlast(d -> Dates.month(d) <= Dates.month(metadatum.dates), dates)

    ds = NCDataset(path)

    if metadatum.name in BGC_INIT_2D_VARIABLES
        data = Float32.(ds[name][:, :, time_idx])
    else
        data = Float32.(ds[name][:, :, :, time_idx])
    end

    close(ds)

    if metadatum.name in BGC_INIT_2D_VARIABLES
        return data
    else
        return reverse(data, dims=3)
    end
end

#####
##### Convenience constructors for iron deposition
#####

"""
    BGCInitSolubleIronDeposition(arch_or_grid = CPU(); kw...)

Monthly soluble iron deposition from Hamilton et al. (2022) via the BGC init climatology, as a
`FieldTimeSeries` in mmol Fe m⁻² s⁻¹ (negative into the ocean, i.e. a top boundary flux) with
`Cyclical` indexing.

Pass the model `grid` (rather than an architecture, which gives the dataset's native 1ᵒ grid) to
interpolate onto it, as is needed to use it as a boundary condition: the series is indexed with the
model's `i, j`. All the snapshots are kept in memory by default.
"""
function BGCInitSolubleIronDeposition(arch_or_grid::Union{AbstractArchitecture, AbstractGrid} = CPU();
                                      dataset = BGCInitClimatology(),
                                      name = :soluble_iron_deposition,
                                      dates = DataWrangling.all_dates(dataset, name),
                                      dir = nothing,
                                      time_indices_in_memory = length(dates),
                                      kw...)

    metadata = isnothing(dir) ? Metadata(name; dataset, dates) :
                                Metadata(name; dataset, dates, dir)
    return FieldTimeSeries(metadata, arch_or_grid; time_indices_in_memory, kw...)
end

"""
    BGCInitDustIronDeposition(arch_or_grid = CPU(); kw...)

Seasonal total dust deposition from Kok et al. (2021) via the BGC init climatology, as a
`FieldTimeSeries` in kg dust m⁻² s⁻¹ (positive into the ocean) with `Cyclical` indexing, for
`IronDustDepositionForcing`, which converts it to iron (3.5 wt% by default) dissolving with depth.

Pass the model `grid` to interpolate onto it (see [`BGCInitSolubleIronDeposition`](@ref)).

Note: this is total dust (so soluble + insoluble iron). If the model also applies
`BGCInitSolubleIronDeposition` as a separate surface flux, the soluble portion is double-counted.
"""
function BGCInitDustIronDeposition(arch_or_grid::Union{AbstractArchitecture, AbstractGrid} = CPU();
                                   dataset = BGCInitClimatology(),
                                   name = :dust_deposition,
                                   dates = DataWrangling.all_dates(dataset, name),
                                   dir = nothing,
                                   time_indices_in_memory = length(dates),
                                   kw...)

    metadata = isnothing(dir) ? Metadata(name; dataset, dates) :
                                Metadata(name; dataset, dates, dir)
    return FieldTimeSeries(metadata, arch_or_grid; time_indices_in_memory, kw...)
end
