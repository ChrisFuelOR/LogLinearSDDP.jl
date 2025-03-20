
""" The idea is to first identify and store for each stage the set of all corresponding nodes. 
Then, for each node, arcs are created to all nodes at the following stage, using the existing probability 
(if the child node is contained in the "successors" field) or 0.0 otherwise."""

function SDDP.MSPFormat._parse_lattice(filename::String)

    data = JuMP.MOI.FileFormats.compressed_open(
        JSON.parse,
        filename,
        "r",
        JuMP.MOI.FileFormats.AutomaticCompression(),
   )
    graph = SDDP.Graph("root")

    max_stage = 0
    for (key, value) in data
        if value["stage"] > max_stage
            max_stage = value["stage"]
        end
    end

    stage_node_combinations = Vector{Vector{String}}(undef, max_stage+1)
    for stage in 0:max_stage
        stage_node_combinations[stage+1] = Vector{String}()
    end
    
    for key in keys(data)
        SDDP.add_node(graph, key)
        push!(stage_node_combinations[data[key]["stage"]+1], key)
    end

    for (key, value) in data
        if value["stage"] == max_stage
            continue
        end

        for child in stage_node_combinations[value["stage"]+2]
            if child in keys(value["successors"])
                SDDP.add_edge(graph, key => child, value["successors"][child])
            else
                SDDP.add_edge(graph, key => child, 0.0)
            end
        end
    end
end