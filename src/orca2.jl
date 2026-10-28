# ORCA2 tripolar grid, adapted from NumericalEarth PR #153 (jsw/orca2) onto NumericalEarth 0.8.1's `ORCAGrid`.
#
# NumericalEarth 0.8.1 only knows the F-pivot eORCA grids (ORCAOne, ORCAQuarter, ORCATwelfth). ORCA2 differs in three
# ways, all read from the NEMO 5 domain_cfg in Zenodo record 15705144 (global attribute `NFtype = T`, 180 × 148, no
# halo):
#
# 1. It folds about T points, so the topology is `RightCenterFolded` with a `TPivot` zipper (PR #153 used the default
#    UPivot pivot).
# 2. `RightCenterFolded` has Ny y-faces rather than Ny + 1, so NumericalEarth's `halo_filled_data` would put NEMO V
#    (the north face of T[:, j]) on the south face of T[:, j]. Here the southernmost row(s), which are land, are removed
#    and the south face of each kept row j is NEMO V/F[:, j - 1].
# 3. The file has no duplicated east columns (`periodic_overlap = 0`).

using Downloads: Downloads
using NCDatasets: Dataset
using Oceananigans.BoundaryConditions: fill_halo_regions!, TPivotZipperBoundaryCondition, NoFluxBoundaryCondition,
                                       FieldBoundaryConditions, PeriodicBoundaryCondition
using Oceananigans.BoundaryConditions: TPivot
using Oceananigans.Grids: RightCenterFolded, generate_coordinate, topology
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.OrthogonalSphericalShellGrids: Tripolar
using NumericalEarth.Bathymetry: read_orca_staggered_mesh, read_2d_nemo_variable, remove_minor_basins!

const ORCA2_url = "https://zenodo.org/records/15705144/files/ORCA2L75_domaincfg_forcings.tar.gz"
const ORCA2_domain_cfg = joinpath("ORCA2L75", "domain_cfg_orca2l75_nemo5.nc")

"""
    download_orca2(dir)

Path to the ORCA2 domain_cfg in `dir`, downloading and unpacking the Zenodo tarball (3 MB) there first if needed.
"""
function download_orca2(dir)
    path = joinpath(dir, ORCA2_domain_cfg)
    isfile(path) && return path
    mkpath(dir)
    tarball = joinpath(dir, "ORCA2L75.tar.gz")
    @info "Downloading ORCA2 domain_cfg to $dir..."
    Downloads.download(ORCA2_url, tarball)
    run(`tar -xzf $tarball -C $dir`)
    return path
end

# Halo-filled copy of `data` (already aligned to the kept rows, see `ORCA2Grid`) at (LX, LY) on the RightCenterFolded
# helper grid
function orca2_halo_filled_data(data, helper_grid, bcs, LX, LY)
    Nx, Ny, _ = size(helper_grid)
    field = Field{LX, LY, Center}(helper_grid; boundary_conditions = bcs)
    field.data[1:Nx, 1:Ny, 1] .= data[1:Nx, 1:Ny]
    fill_halo_regions!(field)

    return deepcopy(dropdims(field.data, dims = 3))
end

orca2_halo_fill_stagger(CC, FC, CF, FF, helper_grid, bcs) =
    (orca2_halo_filled_data(CC, helper_grid, bcs, Center, Center),
     orca2_halo_filled_data(FC, helper_grid, bcs, Face,   Center),
     orca2_halo_filled_data(CF, helper_grid, bcs, Center, Face),
     orca2_halo_filled_data(FF, helper_grid, bcs, Face,   Face))

"""
    ORCA2Grid(arch = CPU(), FT = Oceananigans.defaults.FloatType;
              z, Nz = 30, halo = (5, 5, 4), with_bathymetry = true, active_cells_map = true,
              major_basins = Inf, south_rows_to_remove = 1, dir = joinpath("data", "ORCA2"))

`OrthogonalSphericalShellGrid` with `(Periodic, RightCenterFolded, Bounded)` topology (T-pivot fold) from the NEMO 5
ORCA2L75 domain_cfg: nominally 2°, refined meridionally to ~0.5° at the equator and near Antarctica, and to ~0.5° in
the Mediterranean, Red, Black and Caspian Seas. Coordinates, scale factors (including NEMO's narrowed strait widths in
`e1v`/`e2u`) and areas are read from the file; with `with_bathymetry = true` (the default) the domain_cfg's
`bathy_metry` (m, positive down) is negated into a `GridFittedBottom` (m, positive up; land at +100 m) and an
`ImmersedBoundaryGrid` is returned. `major_basins` keeps the largest connected basins, as in `ORCAGrid`.
`south_rows_to_remove` (≥ 1) drops the southernmost rows, which are land: with at least one gone every kept row's south
face is a NEMO V/F row (the first row has none in the file).
"""
function ORCA2Grid(arch = CPU(), FT::DataType = Oceananigans.defaults.FloatType;
                   z,
                   Nz = 30,
                   halo = (5, 5, 4),
                   radius = Oceananigans.defaults.planet_radius,
                   with_bathymetry = true,
                   active_cells_map = true,
                   major_basins = Inf,
                   south_rows_to_remove = 1,
                   dir = joinpath("data", "ORCA2"))

    jr = south_rows_to_remove
    jr ≥ 1 || throw(ArgumentError("south_rows_to_remove = $jr must be ≥ 1: NEMO has no south face for its first row"))

    path = download_orca2(dir)

    ds = Dataset(path)
    mesh = read_orca_staggered_mesh(ds, 0; radius) # (3.)
    bathy_data = read_2d_nemo_variable(ds, "bathy_metry")
    close(ds)

    # Keep T rows jr+1:end. NEMO V/F[:, j] is the north face of T[:, j], so the south faces of the kept rows are
    # V/F rows jr:end-1 (2.)
    center_y(data) = data[:, jr+1:end]
    face_y(data)   = data[:, jr:end-1]

    λCC, λFC, λCF, λFF     = center_y(mesh.λCC),  center_y(mesh.λFC),  face_y(mesh.λCF),  face_y(mesh.λFF)
    φCC, φFC, φCF, φFF     = center_y(mesh.φCC),  center_y(mesh.φFC),  face_y(mesh.φCF),  face_y(mesh.φFF)
    e1t, e1u, e1v, e1f     = center_y(mesh.e1t),  center_y(mesh.e1u),  face_y(mesh.e1v),  face_y(mesh.e1f)
    e2t, e2u, e2v, e2f     = center_y(mesh.e2t),  center_y(mesh.e2u),  face_y(mesh.e2v),  face_y(mesh.e2f)
    AzCC, AzFC, AzCF, AzFF = center_y(mesh.AzCC), center_y(mesh.AzFC), face_y(mesh.AzCF), face_y(mesh.AzFF)
    bathy_data = center_y(bathy_data)

    Nx, Ny = size(λCC)
    Hx, Hy, Hz = halo

    pole_idx = argmin(mesh.φFF[:, end])
    north_poles_latitude  = Float64(mesh.φFF[pole_idx, end])
    first_pole_longitude  = Float64(mesh.λFF[pole_idx, end])
    southernmost_latitude = Float64(minimum(φCC))

    topo = (Periodic, RightCenterFolded, Bounded)
    Lz, z_coord = generate_coordinate(FT, topo, (Nx, Ny, Nz), halo, z, :z, 3, CPU())

    helper_grid = RectilinearGrid(; size = (Nx, Ny), halo = (Hx, Hy), x = (0, 1), y = (0, 1),
                                    topology = (Periodic, RightCenterFolded, Flat))

    bcs = FieldBoundaryConditions(north  = TPivotZipperBoundaryCondition(), # (1.)
                                  south  = NoFluxBoundaryCondition(),
                                  west   = PeriodicBoundaryCondition(),
                                  east   = PeriodicBoundaryCondition(),
                                  top    = nothing,
                                  bottom = nothing)

    λᶜᶜᵃ, λᶠᶜᵃ, λᶜᶠᵃ, λᶠᶠᵃ     = orca2_halo_fill_stagger(λCC,  λFC,  λCF,  λFF,  helper_grid, bcs)
    φᶜᶜᵃ, φᶠᶜᵃ, φᶜᶠᵃ, φᶠᶠᵃ     = orca2_halo_fill_stagger(φCC,  φFC,  φCF,  φFF,  helper_grid, bcs)
    Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ = orca2_halo_fill_stagger(e1t,  e1u,  e1v,  e1f,  helper_grid, bcs)
    Δyᶜᶜᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶠᵃ = orca2_halo_fill_stagger(e2t,  e2u,  e2v,  e2f,  helper_grid, bcs)
    Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ = orca2_halo_fill_stagger(AzCC, AzFC, AzCF, AzFF, helper_grid, bcs)

    to_arch(data) = on_architecture(arch, map(FT, data))

    underlying_grid = OrthogonalSphericalShellGrid{Periodic, RightCenterFolded, Bounded}(
        arch,
        Nx, Ny, Nz,
        Hx, Hy, Hz,
        convert(FT, Lz),
        to_arch(λᶜᶜᵃ), to_arch(λᶠᶜᵃ), to_arch(λᶜᶠᵃ), to_arch(λᶠᶠᵃ),
        to_arch(φᶜᶜᵃ), to_arch(φᶠᶜᵃ), to_arch(φᶜᶠᵃ), to_arch(φᶠᶠᵃ),
        on_architecture(arch, z_coord),
        to_arch(Δxᶜᶜᵃ), to_arch(Δxᶠᶜᵃ), to_arch(Δxᶜᶠᵃ), to_arch(Δxᶠᶠᵃ),
        to_arch(Δyᶜᶜᵃ), to_arch(Δyᶠᶜᵃ), to_arch(Δyᶜᶠᵃ), to_arch(Δyᶠᶠᵃ),
        to_arch(Azᶜᶜᵃ), to_arch(Azᶠᶜᵃ), to_arch(Azᶜᶠᵃ), to_arch(Azᶠᶠᵃ),
        convert(FT, radius),
        Tripolar(north_poles_latitude, first_pole_longitude, southernmost_latitude, RightCenterFolded, TPivot))

    with_bathymetry || return underlying_grid

    bottom_height  = FT.(coalesce.(bathy_data, FT(0)))
    bottom_height .= ifelse.(isfinite.(bottom_height) .& (bottom_height .> 0), .-bottom_height, FT(100))
    bottom_field   = Field{Center, Center, Nothing}(underlying_grid)
    set!(bottom_field, on_architecture(arch, bottom_height))

    major_basins < Inf && remove_minor_basins!(bottom_field, major_basins)

    return ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_field); active_cells_map)
end

