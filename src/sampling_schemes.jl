# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Copyright (c) 2025 Christian Fuellner <christian.fuellner@kit.edu>

# Note that this code reuses functions from SDDP.jl by Oscar Dowson,
# which are licensed under the Mozilla Public License, Version 2.0 as well. 
# Copyright (c) 2017-2025: Oscar Dowson and SDDP.jl contributors.
################################################################################

function sample_scenario(
    graph::SDDP.PolicyGraph{T},
    ar_process::LogLinearSDDP.AutoregressiveProcess,
    sampling_scheme::Union{SDDP.InSampleMonteCarlo,SDDP.OutOfSampleMonteCarlo{T}},
    keep_full_history::Bool = false,
) where {T}
    max_depth = min(sampling_scheme.max_depth, sampling_scheme.rollout_limit())
    # Storage for our scenario. Each tuple is (node_index, noise.term).
    scenario_path = Tuple{T,Any}[]
    # We only use visited_nodes if terminate_on_cycle=true. Just initialize anyway.
    visited_nodes = Set{T}()
    # Begin by sampling a node from the children of the root node.
    node_index = something(
        sampling_scheme.initial_node,
        SDDP.sample_noise(SDDP.get_root_children(sampling_scheme, graph)),
    )::T
    while true
        node = graph[node_index]

        # CHANGES TO SDDP.jl
        ####################################################################################
        # Get the realizations for the stagewise independent term η
        all_independent_noises = SDDP.get_noise_terms(sampling_scheme, node, node_index)
        children = SDDP.get_children(sampling_scheme, node, node_index)

        # Sample an independent noise term (vector/tuple of index ℓ or a single AbstractFloat)
        independent_noise_terms = SDDP.sample_noise(all_independent_noises)

        # Get the current process state
        process_state = node.ext[:process_state]

        # # JUST FOR TESTING
        # if node_index == 2
        #     independent_noise_terms = 1.0
        # elseif node_index == 3
        #     independent_noise_terms = -2.0
        # end

        # # JUST FOR TESTING
        # if node_index == 2
        #     independent_noise_terms = (3/4, -0.5)
        # elseif node_index == 3
        #     independent_noise_terms = (3/4, 0.0)
        # end

        # First stage is deterministic
        if node_index == 1
            noise_term = zeros(length(independent_noise_terms))
            for ℓ in eachindex(noise_term)    
                noise_term[ℓ] = ar_process.history[1][ℓ]
            end
        else      
            # Compute the actual noise ξ using the formula for the log-linear AR process
            # Note: No matter how the noises are defined in parameterize in the model description, noise_term here is always 
            # a vector with component index ℓ (or an AbstractFloat).
            # Further note that process_state is first converted to an array for more efficient computations.
            noise_term = Vector{Float64}(undef, length(independent_noise_terms))
            
            TimerOutputs.@timeit graph.timer_output "process_state_to_array" begin
                ps_array = Array{Float64,2}(undef, ar_process.dimension, ar_process.lag_order)
                process_state_to_array!(ps_array, process_state, node.index)
            end

            TimerOutputs.@timeit graph.timer_output "set_noise_terms" begin
                set_noise_terms!(noise_term, node.index, ar_process.dimension, ar_process.lag_order, ar_process.parameters[node.index], collect(independent_noise_terms), ps_array)
            end

        end

        ####################################################################################

        push!(scenario_path, (node_index, noise_term))
        # Termination conditions:
        if length(children) == 0
            # 1. Our node has no children, i.e., we are at a leaf node.
            return scenario_path, false
        elseif sampling_scheme.terminate_on_cycle && node_index in visited_nodes
            # 2. terminate_on_cycle = true and we have detected a cycle.
            return scenario_path, true
        elseif 0 < sampling_scheme.max_depth <= length(scenario_path)
            # 3. max_depth > 0 and we have explored max_depth number of nodes.
            return scenario_path, false
        elseif sampling_scheme.terminate_on_dummy_leaf &&
               rand() < 1 - sum(child.probability for child in children)
            # 4. we sample a "dummy" leaf node in the next step due to the
            # probability of the child nodes summing to less than one.
            return scenario_path, false
        end
        # We only need to store a list of visited nodes if we want to terminate
        # due to the presence of a cycle.
        if sampling_scheme.terminate_on_cycle
            push!(visited_nodes, node_index)
        end
        # Sample a new node to transition to.
        node_index = SDDP.sample_noise(children)::T

        # CHANGES TO SDDP.jl
        ####################################################################################
        # Store the updated dict as the process state for the following stage (node)
        graph[node_index].ext[:process_state] = update_process_state(graph, ar_process.lag_order, node_index, process_state, noise_term, keep_full_history)
        ####################################################################################

    end
    # Throw an error because we should never end up here.
    return error(
        "Internal SDDP error: something went wrong sampling a scenario.",
    )
end

function set_noise_terms!(
    noise_term::Any,
    t::Int64,
    L::Int64,
    p::Int64,
    ar_process_stage::LogLinearSDDP.AutoregressiveProcessStage,
    independent_noise_terms::Vector{Float64},
    process_state::Array{Float64,2},
    )

    coefficients = ar_process_stage.coefficients
    
    @turbo for ℓ in eachindex(noise_term)
        expo = exp(ar_process_stage.intercept[ℓ] + independent_noise_terms[ℓ] * ar_process_stage.psi[ℓ])
        noise_term[ℓ] = expo

        for k in 1:p
            for m in 1:L
                noise_term[ℓ] *= process_state[m,k]^coefficients[ℓ,m,k]
            end
        end
    end
    return
end

function process_state_to_array!(
    ps_array::Array{Float64,2},
    process_state::Dict{Int64,Any},
    t::Int64,
)   
    
    for k in 1:size(ps_array,2)
        for m in 1:size(ps_array,1)
            ps_array[m,k] = process_state[t-k][m]
        end
    end
    return
end

function update_process_state(
    graph::SDDP.PolicyGraph{T},
    lag_order::Int,
    node_index::Int64,
    process_state::Dict{Int64,Any},
    noise_term::Any,
    keep_full_history::Bool,
) where {T}

    new_process_state = Dict{Int64,Any}()

    # Get lowest required index
    if keep_full_history
        min_time_index = minimum(collect(keys(process_state)))
    else
        min_time_index = node_index - lag_order
    end
    
    # Determine new process state
    for k in min_time_index:node_index-1
        if k == node_index - 1
            if isa(noise_term, AbstractFloat)
                new_process_state[k] = [noise_term]
            else
                new_process_state[k] = noise_term
            end
        else
            new_process_state[k] = process_state[k]
        end
    end

    return new_process_state
end


"""
     CompleteSampler()

Backward sampler that returns all noises of the corresponding node for the log-linear case.
"""
struct CompleteSampler <: SDDP.AbstractBackwardSamplingScheme end

function sample_backward_noise_terms(
    ::CompleteSampler, 
    node::SDDP.Node,
    ) 

    # CHANGES TO SDDP.jl
    ####################################################################################
    # Note that here no special consideration for the first stage data is required, as we do not consider this stage.
    graph = SDDP.get_policy_graph(node.subproblem)

    # Get the realizations for the stagewise independent term η
    all_independent_noises = SDDP.get_noise_terms(SDDP.InSampleMonteCarlo(), node, node.index) #TODO

    # Get the current process state matrix
    process_state = node.ext[:process_state]

    # Get autoregressive data
    ar_process = graph.ext[:ar_process]
    ar_process_stage = ar_process.parameters[node.index]
    
    all_dependent_noises = Vector{Any}(undef, length(all_independent_noises))

    # Compute the actual noise ξ using the formula for the log-linear AR process
    # Differentiation between one- and multi-dimensional noise, because noise.term has different type in both cases.
    for i in eachindex(all_independent_noises)
        independent_noise = all_independent_noises[i]
        noise_values = zeros(ar_process.dimension)

        TimerOutputs.@timeit graph.timer_output "process_state_to_array" begin
                ps_array = Array{Float64,2}(undef, ar_process.dimension, ar_process.lag_order)
                process_state_to_array!(ps_array, process_state, node.index)
        end

        TimerOutputs.@timeit graph.timer_output "set_noise_terms" begin
            set_noise_terms!(noise_values, node.index, ar_process.dimension, ar_process.lag_order, ar_process.parameters[node.index], collect(independent_noise.term), ps_array)
        end

        # Note: No matter how the noises are defined in parameterize in the model description, noise_values here is always a vector containing all components.
        all_dependent_noises[i] = SDDP.Noise(noise_values, independent_noise.probability)
    end

    return (all_dependent_noises = all_dependent_noises, all_independent_noises = all_independent_noises)
end

