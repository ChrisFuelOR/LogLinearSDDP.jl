function sample_scenario(
    graph::SDDP.PolicyGraph{T},
    sampling_scheme::Union{SDDP.InSampleMonteCarlo,SDDP.OutOfSampleMonteCarlo{T}},
) where {T}
    max_depth = min(sampling_scheme.max_depth, sampling_scheme.rollout_limit())
    # Storage for our scenario. Each tuple is (node_index, noise.term).
    scenario_path = Tuple{T,Any}[]
    # We only use visited_nodes if terminate_on_cycle=true. Just initialize
    # anyway.
    visited_nodes = Set{T}()
    # Begin by sampling a node from the children of the root node.
    node_index = something(
        sampling_scheme.initial_node,
        sample_noise(get_root_children(sampling_scheme, graph)),
    )::T
    while true
        node = graph[node_index]

        # CHANGES TO SDDP.jl
        ####################################################################################
        # Get the realizations for the stagewise independent term η
        independent_noise_terms = get_noise_terms(sampling_scheme, node, node_index)
        children = get_children(sampling_scheme, node, node_index)

        # Sample an independent noise term
        independent_noise = sample_noise(independent_noise_terms)

        # Get the current process state matrix
        process_state = node.ext[:process_state]

        # Get autoregressive data
        autoregressive_data = graph.ext[:autoregressive_data]
        autoregressive_data_stage = autoregressive_data.ar_data[node_index]

        # Compute the actual noise ξ using the formula for the log-linear AR process
        noise = zeros(length(independent_noise))
        for ℓ in eachindex(noise)
            t = node_index
            intercept = autoregressive_data_stage.ar_intercept[ℓ]
            independent_term = independent_noise[ℓ]
            coefficients = autoregressive_data_stage.ar_coefficients
            ar_lag_order = autoregressive_data.ar_lag_order
            ar_lag_dimensions = get_lag_dimensions(autoregressive_data, t)

            noise[ℓ] = exp(intercept) * exp(independent_term) * prod(process_state[t-k][m]^coefficients[k][ℓ][m] for k in 1:ar_lag_order for m in 1:ar_lag_dimensions[t-k])
        end
        ####################################################################################

        push!(scenario_path, (node_index, noise))
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
        node_index = sample_noise(children)::T

        # CHANGES TO SDDP.jl
        ####################################################################################
        # Set the process state for the new (following) node
        _update_process_state(graph, node_index, process_state, noise)
        ####################################################################################

    end
    # Throw an error because we should never end up here.
    return error(
        "Internal SDDP error: something went wrong sampling a scenario.",
    )
end


function _update_process_state(
    graph::SDDP.PolicyGraph{T},
    node_index::Int,
    process_state::Dict{Int64,Vector{Float64}},
    noise::SDDP.Noise,
) where {T}

    new_process_state = Dict{Int64,Vector{Float64}}()

    # Get lowest required index
    min_time_index = 1 - graph.ext[:autoregressive_data].ar_lag_order
    
    # Determine new process state
    for k in min_time_index:node_index-1
        if k == node_index - 1
            new_process_state[k] = noise
        else
            new_process_state[k] = process_state[k]
        end
    end

    # Store the updated dict as the process state for the following stage (node)
    node = graph[node_index]
    node.ext[:process_state] = new_process_state
   
    return
end


"""
     CompleteSampler()

Backward sampler that returns all noises of the corresponding node for the log-linear case.
"""
struct CompleteSampler <: SDDP.AbstractBackwardSamplingScheme end

function sample_backward_noise_terms(
    ::CompleteSampler, 
    node) 

    # CHANGES TO SDDP.jl
    ####################################################################################

    # Get the realizations for the stagewise independent term η
    independent_noise_terms = SDDP.get_noise_terms(sampling_scheme, node, node_index)

    # Get the current process state matrix
    process_state = node.ext[:process_state]

    # Get autoregressive data
    graph = SDDP.get_policy_graph(node.subproblem)
    autoregressive_data = graph.ext[:autoregressive_data]
    autoregressive_data_stage = autoregressive_data.ar_data[node.index]

    noise_terms = Vector{Any}(undef, length(independent_noise_terms))

    for i in eachindex(independent_noise_terms)
        independent_noise = independent_noise_terms[i]

        # Compute the actual noise ξ using the formula for the log-linear AR process
        noise = zeros(length(independent_noise))
        for ℓ in eachindex(noise)
            t = node.index
            intercept = autoregressive_data_stage.ar_intercept[ℓ]
            independent_term = independent_noise[ℓ]
            coefficients = autoregressive_data_stage.ar_coefficients
            ar_lag_order = autoregressive_data.ar_lag_order
            ar_lag_dimensions = get_lag_dimensions(autoregressive_data, t)

            noise[ℓ] = exp(intercept) * exp(independent_term) * prod(process_state[t-k][m]^coefficients[k][ℓ][m] for k in 1:ar_lag_order for m in 1:ar_lag_dimensions[t-k])
        end

        noise_terms[i] = noise
    end

    return noise_terms = noise_terms
end

