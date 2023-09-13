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
        SDDP.sample_noise(SDDP.get_root_children(sampling_scheme, graph)),
    )::T
    while true
        node = graph[node_index]

        # CHANGES TO SDDP.jl
        ####################################################################################
        # Get the realizations for the stagewise independent term η
        independent_noise_terms = SDDP.get_noise_terms(sampling_scheme, node, node_index)
        children = SDDP.get_children(sampling_scheme, node, node_index)

        # Sample an independent noise term
        independent_noise = SDDP.sample_noise(independent_noise_terms)

        # JUST FOR TESTING
        if node_index == 2
            independent_noise = 1.0
        elseif node_index == 3
            independent_noise = -2.0
        end

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

            noise[ℓ] = exp(intercept) * exp(independent_term) * prod(process_state[t-k][m]^coefficients[k][ℓ][m] for k in 1:ar_lag_order for m in 1:ar_lag_dimensions[k])
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
        node_index = SDDP.sample_noise(children)::T

        # CHANGES TO SDDP.jl
        ####################################################################################
        # Store the updated dict as the process state for the following stage (node)
        graph[node_index].ext[:process_state] = update_process_state(graph, node_index, process_state, noise)
        ####################################################################################

    end
    # Throw an error because we should never end up here.
    return error(
        "Internal SDDP error: something went wrong sampling a scenario.",
    )
end


function update_process_state(
    graph::SDDP.PolicyGraph{T},
    node_index::Int,
    process_state::Dict{Int64,Vector{Float64}},
    noise::Union{Float64,Vector{Float64}},
) where {T}

    new_process_state = Dict{Int64,Vector{Float64}}()

    # Get lowest required index
    min_time_index = 1 - graph.ext[:autoregressive_data].ar_lag_order
    
    # Determine new process state
    for k in min_time_index:node_index-1
        if k == node_index - 1
            if isa(noise, AbstractFloat)
                new_process_state[k] = [noise]
            else
                new_process_state[k] = noise
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

    graph = SDDP.get_policy_graph(node.subproblem)

    # Get the realizations for the stagewise independent term η
    independent_noises = SDDP.get_noise_terms(SDDP.InSampleMonteCarlo(), node, node.index) #TODO

    # Get the current process state matrix
    process_state = node.ext[:process_state]

    # Get autoregressive data
    autoregressive_data = graph.ext[:autoregressive_data]
    autoregressive_data_stage = autoregressive_data.ar_data[node.index]
    
    dependent_noises = Vector{Any}(undef, length(independent_noises))

    # Compute the actual noise ξ using the formula for the log-linear AR process
    # Differentiation between one- and multi-dimensional noise, because noise.term has different type in both cases.
    for i in eachindex(independent_noises)
        independent_noise = independent_noises[i]

        if isa(independent_noise.term, Float64)
            t = node.index
            intercept = autoregressive_data_stage.ar_intercept[1]
            independent_value = independent_noise.term
            coefficients = autoregressive_data_stage.ar_coefficients
            ar_lag_order = autoregressive_data.ar_lag_order
            ar_lag_dimensions = get_lag_dimensions(autoregressive_data, t)

            noise_values = exp(intercept) * exp(independent_value) * prod(process_state[t-k][m]^coefficients[k][1][m] for k in 1:ar_lag_order for m in 1:ar_lag_dimensions[k])

        elseif isa(independent_noise.term, Vector{Float64})
            noise_dimension = length(independent_noise.term)
            noise_values = zeros(noise_dimension)

            for ℓ in eachindex(noise_values)
                t = node.index
                intercept = autoregressive_data_stage.ar_intercept[ℓ]
                independent_value = independent_noise.term[ℓ]
                coefficients = autoregressive_data_stage.ar_coefficients
                ar_lag_order = autoregressive_data.ar_lag_order
                ar_lag_dimensions = get_lag_dimensions(autoregressive_data, t)

                noise_values[ℓ] = exp(intercept) * exp(independent_value) * prod(process_state[t-k][m]^coefficients[k][ℓ][m] for k in 1:ar_lag_order for m in 1:ar_lag_dimensions[k])
            end
        end

        dependent_noises[i] = SDDP.Noise(noise_values, independent_noise.probability)
    end

    return (dependent_noise_terms = dependent_noises, independent_noise_terms = independent_noises)
end

