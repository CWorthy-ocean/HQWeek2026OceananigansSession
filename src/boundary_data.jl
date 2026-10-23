function boundary_data(path, variable)
    file_path = joinpath(path, "glorys_$(variable)_boundaries.jld2")
    
    east = on_metal(FieldTimeSeries(file_path, "$(variable)_west"), grid)
    south = on_metal(FieldTimeSeries(file_path, "$(variable)_south"), grid)
    north = on_metal(FieldTimeSeries(file_path, "$(variable)_north"), grid)

    return east, south, north
end