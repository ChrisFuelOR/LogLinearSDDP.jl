import SDDP
import LogLinearSDDP
import Gurobi
import JuMP
import Infiltrator
import Revise
import DataFrames
import CSV
import Dates

const GRB_ENV = Gurobi.Env()

function read_data(file_name::String, first_line_number::Int, number_of_rows::Int)

    df = CSV.read(file_name, header=false, skipto=first_line_number, limit=number_of_rows, delim=";", DataFrames.DataFrame)
    return df
end


function closest_node(nodes::Dict{String, SDDP.Node{String}}, previous_node::Union{Nothing,SDDP.Node{String}}, noise::Dict{String, Float64}) # or Any instead of Float64
    if isnothing(previous_node)
        return "0"
    end
    
    children = SDDP.get_children(SDDP.InSampleMonteCarlo(), previous_node, previous_node.index)
    closest_child = nothing
    minimum_dist = Inf

    for child in children
        @assert length(nodes[child.term].noise_terms) == 1

        dist = 0.0
        # Compute distance from child to noise
        for (key, value) in nodes[child.term].noise_terms[1].term
            dist += (value - noise[key])^2
            #println(key, ", ", value, ", ", noise[key])
        end
        #println(child.term, ", ", dist)

        # Update closest_child
        if dist < minimum_dist 
            closest_child = child
            minimum_dist = dist
        end

    end

    return closest_child.term
end    


function get_hydrothermal_model_markov(nodes_per_stage::Int)

    if nodes_per_stage == 10
        file_identifier = "(12_0)_10"
    elseif nodes_per_stage == 100
        file_identifier = "(12_0)_100"
    else
        Error("Nodes per stage must be 10 or 100")
    end

    model = SDDP.MSPFormat.read_from_file(file_identifier, bound = 0.0)

    return model
end


function get_inflows_for_forward_pass(model::SDDP.PolicyGraph, model_approach::String, seed::Int, number_of_iterations::Int, number_of_stages::Int)
    
    file_name = "MarkovPreparation/inflows_fp_" * model_approach * "_" * string(seed) * ".txt"

    # Read data from CSV files
    df = read_data(file_name, 1, number_of_iterations * number_of_stages)

    # Initialize vector
    all_sample_paths = Vector{Vector{Tuple{String,Dict{String,Float64}}}}()

    # Construct vector of sample paths
    for scenario_path_index in 1:number_of_iterations
        # Construct a single sample path
        sample_path = Vector{Tuple{String,Dict{String,Float64}}}()
        previous_node = nothing
        for stage in 1:number_of_stages
            closest_node_index = Inf
            row = (scenario_path_index - 1) * number_of_stages + stage
            @assert df[row, :Column1] == stage

            realization = Dict("PLANT_1" => df[row, :Column2], "PLANT_2" => df[row, :Column3], "PLANT_3" => df[row, :Column4], "PLANT_4" => df[row, :Column5])
            closest_node_index = closest_node(model.nodes, previous_node, realization)
            # println(stage, ",", closest_node_index)

            push!(sample_path, (closest_node_index, realization))
            previous_node = model.nodes[closest_node_index]
        end
        push!(all_sample_paths, sample_path)
    end

    return all_sample_paths
end


function starter()
    # PARAMETER SET-UP
    ###########################################################################################################
    model_approach = "bic_model"
    forward_pass_seed = 11111
    number_of_replications = 100 # 1000
    number_of_stages = 120
    number_of_markov_nodes = 100

    file_identifier = "Run_" * string(model_approach) * "_" * string(forward_pass_seed)
    file_path = "C:/Users/cg4102/Documents/julia_logs/Cut-sharing/MC-SDDP_" * file_identifier * "/"
    ispath(file_path) || mkdir(file_path)
    log_file = file_path * "MC-SDDP.log"

    simulation_regime = LogLinearSDDP.Simulation(sampling_scheme = SDDP.InSampleMonteCarlo(), number_of_replications = 1) #TODO: to be adapted
    algo_params = LogLinearSDDP.AlgoParams(stopping_rules = [SDDP.TimeLimit(3600)], forward_pass_seed = forward_pass_seed, simulation_regime = simulation_regime, log_file = log_file, silent = false)
  
    # ADDITIONAL LOGGING TO SDDP.jl
    ###########################################################################################################
    log_f = open(log_file, "a")
    println(log_f, "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
    println(log_f, "PATH")
    println(log_f, "calling ")
    println(log_f, @__DIR__)
    println(log_f, Base.source_path())

    # Printing the time
    println(log_f, "DATETIME")
    println(log_f, Dates.now())

    # Printing the algo params
    println(log_f, "RUN DESCRIPTION")
    println(log_f, algo_params)
    close(log_f)

    # CREATE AND RUN MODEL
    ###########################################################################################################
    model = get_hydrothermal_model_markov(number_of_markov_nodes)
    JuMP.set_optimizer(model, () -> Gurobi.Optimizer(GRB_ENV))
    all_sample_paths = get_inflows_for_forward_pass(model, model_approach, forward_pass_seed, number_of_replications, number_of_stages)
    sampling_scheme_fp = SDDP.Historical(all_sample_paths)

    # Train model
    SDDP.train(
        model,
        #sampling_scheme = sampling_scheme_fp,
        print_level = algo_params.print_level,
        log_file = algo_params.log_file,
        log_frequency = algo_params.log_frequency,
        stopping_rules = algo_params.stopping_rules,
        run_numerical_stability_report = algo_params.run_numerical_stability_report,
        log_every_seconds = 0.0
    )

    close(log_f)

end

starter()

# TODO: generate inflows to seed and write to file: for FP
# TODO: generate inflows to seed and write to file: for OOS
# TODO: analogously: get_inflows_for_out_of_sample_simulation()
