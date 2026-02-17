# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Copyright (c) 2026 Christian Fuellner <christian.fuellner@kit.edu>
################################################################################

import SDDP
import LogLinearSDDP
import Gurobi
import JuMP
import Infiltrator
import Revise
import DataFrames
import CSV
import Dates
import JSON

const GRB_ENV = Gurobi.Env()

""" The idea is to first identify and store for each stage the set of all corresponding nodes. 
Then, for each node, arcs are created to all nodes at the following stage, using the existing probability 
(if the child node is contained in the "successors" field) or 0.0 otherwise."""

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
        file_identifier = "PreparationMarkov/(12_0)_10"
    elseif nodes_per_stage == 100
        file_identifier = "PreparationMarkov/(12_0)_100"
    else
        Error("Nodes per stage must be 10 or 100")
    end

    model = SDDP.MSPFormat.read_from_file(file_identifier, bound = 0.0)

    return model
end


function get_inflows_for_forward_pass(model::SDDP.PolicyGraph, model_approach::String, seed::Int, number_of_iterations::Int, number_of_stages::Int)
    
    file_name = "PreparationMarkov/inflows_fp_" * model_approach * "_" * string(seed) * ".txt"

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
            #println(stage, ", ", closest_node_index, ", ", realization, ", ", model.nodes[closest_node_index].noise_terms[1])

            push!(sample_path, (closest_node_index, realization))
            previous_node = model.nodes[closest_node_index]
        end
        push!(all_sample_paths, sample_path)
    end

    return all_sample_paths
end


function get_inflows_for_simulation(model::SDDP.PolicyGraph, model_approach::String, seed::Int, number_of_replications::Int, number_of_stages::Int)

    file_name = "PreparationMarkov/inflows_oos_" * model_approach * "_" * string(seed) * ".txt"

    # Read data from CSV files
    df = read_data(file_name, 1, number_of_replications * number_of_stages)

    # Initialize vector
    all_sample_paths = Vector{Vector{Tuple{String,Dict{String,Float64}}}}()

    # Construct vector of sample paths
    for scenario_path_index in 1:number_of_replications
        # Construct a single sample path
        sample_path = Vector{Tuple{String,Dict{String,Float64}}}()
        previous_node = nothing
        for stage in 1:number_of_stages
            closest_node_index = Inf
            row = (scenario_path_index - 1) * number_of_stages + stage
            @assert df[row, :Column1] == stage

            realization = Dict("PLANT_1" => round(df[row, :Column2], digits=2), "PLANT_2" => round(df[row, :Column3], digits=2), "PLANT_3" => round(df[row, :Column4], digits=2), "PLANT_4" => round(df[row, :Column5], digits=2))
            closest_node_index = closest_node(model.nodes, previous_node, realization)

            push!(sample_path, (closest_node_index, realization))
            previous_node = model.nodes[closest_node_index]
        end
        push!(all_sample_paths, sample_path)
    end

    return all_sample_paths
end


function get_inflows_historical(model::SDDP.PolicyGraph, number_of_stages::Int)

    file_name = "PreparationMarkov/inflows_historical.txt"

    # Read data from CSV files
    df = read_data(file_name, 1, 70 * number_of_stages)

    # Initialize vector
    all_sample_paths = Vector{Vector{Tuple{String,Dict{String,Float64}}}}()

    # Construct vector of sample paths
    for scenario_path_index in 1:70
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


function extended_simulation_analysis_markov(simulation_results::Any, file_path::String, problem_params::LogLinearSDDP.ProblemParams, policy_approach::String, simulation_approach::String)

    # GET TOTAL COST FOR ALL REPLICATIONS AND STORE (ONLY FOR FIRST 60 STAGES)
    ############################################################################
    file_name = file_path * policy_approach * "_" * simulation_approach * "_total_cost.txt"
    f = open(file_name, "w")

    objectives_all_stages = map(simulation_results) do simulation
        return sum(stage[:stage_objective] for stage in simulation)
    end

    if problem_params.number_of_stages == 120
        objectives = map(simulation_results) do simulation
            return sum(simulation[stage][:stage_objective] for stage in 1:60)
        end
    else
        objectives = objectives_all_stages
    end

    for i in eachindex(objectives)
        println(f, objectives[i], ", ", objectives_all_stages[i])
    end
    close(f)

    # GET EMPIRICAL CUMULATIVE DISTRIBUTION OF COSTS (ONLY FOR FIRST 60 STAGES)
    ############################################################################
    file_name = file_path * policy_approach * "_" * simulation_approach * "_cum_distrib.txt"
    f = open(file_name, "w")

    distrib = StatsBase.ecdf(objectives)
    stepsize = round((maximum(objectives)-minimum(objectives))/100, digits=0)
    for x in minimum(objectives)-stepsize:stepsize:maximum(objectives)+stepsize
        y = distrib(x)
        println(f, "(", round(x, digits = 0), ",", round(y, digits = 2), ")")
    end
    close(f)

    # OBTAINING HYDRO RESERVOIR LEVELS
    ############################################################################
    reservoir_names = ["SE", "S", "NE", "N"]

    for k in 1:4
        # Declare file name
        file_name = file_path * policy_approach * "_" * simulation_approach * "_volumes_" * reservoir_names[k] * ".txt"
        f = open(file_name, "w")

        volume_sym = Symbol("v_" * string(k))

        # Get volume data for given reservoir
        column_names = [Symbol(i) for i in 1:problem_params.number_of_stages]
        volume_df = DataFrames.DataFrame([name => Float64[] for name in column_names])
        for i in eachindex(simulation_results)
            outgoing_volume = map(simulation_results[i]) do node
                return node[volume_sym].out
            end    
            push!(volume_df, outgoing_volume)
        end

        # mean
        for stage in 1:problem_params.number_of_stages
            println(f, "(", stage, ",", round(Statistics.mean(volume_df[!, Symbol(stage)]), digits = 2), ")")
        end
        println(f, "###################")

        # 0.05 quantile
        for stage in 1:problem_params.number_of_stages
            println(f, "(", stage, ",", round(Statistics.quantile(volume_df[!, Symbol(stage)], 0.05), digits = 2), ")")
        end
        println(f, "###################")

        # 0.95 quantile
        for stage in 1:problem_params.number_of_stages
            println(f, "(", stage, ",", round(Statistics.quantile(volume_df[!, Symbol(stage)], 0.95), digits = 2), ")")
        end
        println(f, "###################")

        close(f)
    end

    # OBTAINING COST-RELEVANT VARIABLES
    ############################################################################
    # Declare file name
    file_name = file_path * policy_approach * "_" * simulation_approach * "_data.txt"
    f = open(file_name, "w")

    # Get data
    deficit_unit_cost = [1142.8, 2465.4, 5152.46, 5845.54]
    column_names = [Symbol(i) for i in 1:problem_params.number_of_stages]
    value_df = DataFrames.DataFrame([name => Vector{Vector{Float64}}() for name in column_names])
    for i in eachindex(simulation_results)
        outgoing_values = map(simulation_results[i]) do node
            gen = node[:th_1] + node[:th_2] + node[:th_3] + node[:th_4]
            # gen has been wrong in my experiments; gen = gen - deficit gives the correct value
            hydro_gen = node[:q_1] + node[:q_2] + node[:q_3] + node[:q_4]
            exchange = sum(node[exchange_var] for exchange_var in [:f_12, :f_13, :f_15, :f_21, :f_31, :f_35, :f_45, :f_51, :f_53, :f_54])
            spillage = node[:s_1] + node[:s_2] + node[:s_3] + node[:s_4]
            deficit = sum(node[deficit_var] for deficit_var in [:gd_1_1, :gd_1_2, :gd_1_3, :gd_1_4, :gd_2_1, :gd_2_2, :gd_2_3, :gd_2_4, :gd_3_1, :gd_3_2, :gd_3_3, :gd_3_4, :gd_4_1, :gd_4_2, :gd_4_3, :gd_4_4])
            deficit_cost = sum(deficit_unit_cost[1] * node[deficit_var] for deficit_var in [:gd_1_1, :gd_2_1, :gd_3_1, :gd_4_1]) +  sum(deficit_unit_cost[2] * node[deficit_var] for deficit_var in [:gd_1_2, :gd_2_2, :gd_3_2, :gd_4_2]) +  sum(deficit_unit_cost[3] * node[deficit_var] for deficit_var in [:gd_1_3, :gd_2_3, :gd_3_3, :gd_4_3]) + +  sum(deficit_unit_cost[4] * node[deficit_var] for deficit_var in [:gd_1_4, :gd_2_4, :gd_3_4, :gd_4_4])

            return [gen, hydro_gen, deficit, deficit_cost, exchange, spillage]
        end    
        push!(value_df, outgoing_values)
    end
   
    # mean
    for stage in 1:problem_params.number_of_stages
        print(f, "(", stage, ",", )
        print(f, round(Statistics.mean([value_df[!, Symbol(stage)][j][1] for j in 1:DataFrames.nrow(value_df)]), digits = 2), ",")
        print(f, round(Statistics.mean([value_df[!, Symbol(stage)][j][2] for j in 1:DataFrames.nrow(value_df)]), digits = 2), ",")
        print(f, round(Statistics.mean([value_df[!, Symbol(stage)][j][3] for j in 1:DataFrames.nrow(value_df)]), digits = 2), ",")
        print(f, round(Statistics.mean([value_df[!, Symbol(stage)][j][4] for j in 1:DataFrames.nrow(value_df)]), digits = 2), ",")
        print(f, round(Statistics.mean([value_df[!, Symbol(stage)][j][5] for j in 1:DataFrames.nrow(value_df)]), digits = 2), ",")
        println(f, round(Statistics.mean([value_df[!, Symbol(stage)][j][6] for j in 1:DataFrames.nrow(value_df)]), digits = 2), ")")
    end
    println(f, "###################")

    # 0.05 quantile
    for stage in 1:problem_params.number_of_stages
        print(f, "(", stage, ",", )
        print(f, round(Statistics.quantile([value_df[!, Symbol(stage)][j][1] for j in 1:DataFrames.nrow(value_df)], 0.05), digits = 2), ",")
        print(f, round(Statistics.quantile([value_df[!, Symbol(stage)][j][2] for j in 1:DataFrames.nrow(value_df)], 0.05), digits = 2), ",")
        print(f, round(Statistics.quantile([value_df[!, Symbol(stage)][j][3] for j in 1:DataFrames.nrow(value_df)], 0.05), digits = 2), ",")
        print(f, round(Statistics.quantile([value_df[!, Symbol(stage)][j][4] for j in 1:DataFrames.nrow(value_df)], 0.05), digits = 2), ",")
        print(f, round(Statistics.quantile([value_df[!, Symbol(stage)][j][5] for j in 1:DataFrames.nrow(value_df)], 0.05), digits = 2), ",")
        println(f, round(Statistics.quantile([value_df[!, Symbol(stage)][j][6] for j in 1:DataFrames.nrow(value_df)], 0.05), digits = 2), ")")
    end
    println(f, "###################")

    # 0.95 quantile
    for stage in 1:problem_params.number_of_stages
        print(f, "(", stage, ",", )
        print(f, round(Statistics.quantile([value_df[!, Symbol(stage)][j][1] for j in 1:DataFrames.nrow(value_df)], 0.95), digits = 2), ",")
        print(f, round(Statistics.quantile([value_df[!, Symbol(stage)][j][2] for j in 1:DataFrames.nrow(value_df)], 0.95), digits = 2), ",")
        print(f, round(Statistics.quantile([value_df[!, Symbol(stage)][j][3] for j in 1:DataFrames.nrow(value_df)], 0.95), digits = 2), ",")
        print(f, round(Statistics.quantile([value_df[!, Symbol(stage)][j][4] for j in 1:DataFrames.nrow(value_df)], 0.95), digits = 2), ",")
        print(f, round(Statistics.quantile([value_df[!, Symbol(stage)][j][5] for j in 1:DataFrames.nrow(value_df)], 0.95), digits = 2), ",")
        println(f, round(Statistics.quantile([value_df[!, Symbol(stage)][j][6] for j in 1:DataFrames.nrow(value_df)], 0.95), digits = 2), ")")
    end
    println(f, "###################")

    close(f)

    return
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

# starter()

