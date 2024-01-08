# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Copyright (c) 2023 Christian Fuellner <christian.fuellner@kit.edu>

# Note that this code reuses functions from SDDP.jl by Oscar Dowson,
# which are licensed under the Mozilla Public License, Version 2.0 as well. 
# Copyright (c) 2017-2023: Oscar Dowson and SDDP.jl contributors.
################################################################################

function sample_scenario(
    graph::SDDP.PolicyGraph{T},
    sampling_scheme::Union{SDDP.InSampleMonteCarlo,SDDP.OutOfSampleMonteCarlo{T}},
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

        # Get autoregressive data
        ar_process = graph.ext[:ar_process]
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
            ar_process_stage = ar_process.parameters[node_index]
            # Compute the actual noise ξ using the formula for the log-linear AR process
            # Note: No matter how the noises are defined in parameterize in the model description, noise_term here is always 
            # a vector with component index ℓ (or an AbstractFloat).
            noise_term = zeros(length(independent_noise_terms))
            for ℓ in eachindex(noise_term)
                t = node_index
                intercept = ar_process_stage.intercept[ℓ]
                independent_term = independent_noise_terms[ℓ]
                error_term_factor = ar_process_stage.psi[ℓ]
                coefficients = ar_process_stage.coefficients
                lag_order = ar_process.lag_order
                lag_dimensions = get_lag_dimensions(ar_process, t)
                noise_term[ℓ] = exp(intercept) * exp(independent_term * error_term_factor) * prod(process_state[t-k][m]^coefficients[m,k,ℓ] for k in 1:lag_order for m in 1:lag_dimensions[k])
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
        graph[node_index].ext[:process_state] = update_process_state(graph, node_index, process_state, noise_term)
        ####################################################################################

    end
    # Throw an error because we should never end up here.
    return error(
        "Internal SDDP error: something went wrong sampling a scenario.",
    )
end


function update_process_state(
    graph::SDDP.PolicyGraph{T},
    node_index::Int64,
    process_state::Dict{Int64,Any},
    noise_term::Any,
) where {T}

    new_process_state = Dict{Int64,Any}()

    # Get lowest required index
    min_time_index = 1 - graph.ext[:ar_process].lag_order
    
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

        noise_dimension = length(independent_noise.term)
        noise_values = zeros(noise_dimension)

        for ℓ in eachindex(noise_values)
            t = node.index
            intercept = ar_process_stage.intercept[ℓ]
            independent_value = independent_noise.term[ℓ]
            error_term_factor = ar_process_stage.psi[ℓ]
            coefficients = ar_process_stage.coefficients
            lag_order = ar_process.lag_order
            lag_dimensions = get_lag_dimensions(ar_process, t)

            noise_values[ℓ] = exp(intercept) * exp(independent_value * error_term_factor) * prod(process_state[t-k][m]^coefficients[m,k,ℓ] for k in 1:lag_order for m in 1:lag_dimensions[k])
        end

        # Note: No matter how the noises are defined in parameterize in the model description, noise_values here is always a vector containing all components.
        all_dependent_noises[i] = SDDP.Noise(noise_values, independent_noise.probability)
    end

    return (all_dependent_noises = all_dependent_noises, all_independent_noises = all_independent_noises)
end

