# # Restarting from the state of another run, possibly with other tracers
#
# A `Checkpointer` restores the model it was written from. These save the interiors of the prognostic fields of an
# ocean `Simulation` as CPU arrays, and set them in another ocean, where a saved tracer can also be copied into several
# (e.g. the spun-up `DIC` and `Alk` into the replicate carbonate systems `DIC1`, `Alk1`, `DIC2`, ... of
# `CarbonateSystem(N)`). Only the interiors are kept: the halos are filled when the model is set, and the run starts
# with a first-order step as after any `set!`.

using NumericalEarth.DataWrangling: JLD2 # not a direct dependency of this project

restart_fields(ocean) = merge(ocean.model.tracers,
                              (u = ocean.model.velocities.u,
                               v = ocean.model.velocities.v,
                               η = ocean.model.free_surface.displacement,
                               U = ocean.model.free_surface.barotropic_velocities.U,
                               V = ocean.model.free_surface.barotropic_velocities.V))

"""
    save_restart_state(filename, ocean)

Save the interiors of the velocities, tracers, free surface displacement and barotropic transports of `ocean` (an
ocean `Simulation`) and its clock time (s) to `filename`.
"""
function save_restart_state(filename, ocean)
    JLD2.jldopen(filename, "w") do file
        file["time"] = ocean.model.clock.time
        for (name, field) in pairs(restart_fields(ocean))
            file[string(name)] = Array(interior(field))
        end
    end
    return filename
end

"""
    load_restart_state!(ocean, filename; replicates = NamedTuple())

Set the fields of `ocean` saved in `filename` by `save_restart_state`. `replicates` maps a saved name to the fields it
is copied into, e.g. `(DIC = (:DIC1, :DIC2), Alk = (:Alk1, :Alk2))`. Fields that are neither saved nor replicated keep
their values; returns the names that were set.
"""
function load_restart_state!(ocean, filename; replicates = NamedTuple())
    fields = restart_fields(ocean)
    set_names = Symbol[]
    JLD2.jldopen(filename, "r") do file
        for (name, field) in pairs(fields)
            key = string(name)
            if haskey(file, key)
                set!(field, file[key])
                push!(set_names, name)
            end
        end
        for (saved, targets) in pairs(replicates)
            data = file[string(saved)]
            for target in targets
                set!(fields[target], data)
                push!(set_names, target)
            end
        end
    end
    foreach(fill_halo_regions!, fields)
    return set_names
end
