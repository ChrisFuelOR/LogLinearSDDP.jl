""" 
Default function to initialize the history of the stochastic process.

As we do not know explicitly which lagged values are required for which dimension 
at later stages, we defined values for the worst possible case.

Per default, we define historical values of ξ as ones.
"""
function initialize_process_state(
    model::SDDP.PolicyGraph,
    autoregressive_data::LogLinearSDDP.AutoregressiveData,
    user_process_state::Nothing,
)

    min_time_index = 1 - autoregressive_data.ar_lag_order
    max_dimension = get_max_dimension(autoregressive_data)
    process_state = Dict{Int64,Vector{Float64}}()

    # Determine the historical process states
    for t in min_time_index:0
        process_state[t] = ones(max_dimension)
    end

    # Store the initial process state (also required for re-initialization later on)
    model.ext[:initial_process_state] = process_state

    return
end


""" 
Function to initialize the history of the stochastic process with user-defined values.

Values which are not provided, but required are set to 1.

Per default, we define historical values of ξ as ones.
"""
function initialize_process_state(
    model::SDDP.PolicyGraph,
    autoregressive_data::LogLinearSDDP.AutoregressiveData,
    user_process_state::Dict{Int64,Vector{Float64}},
)

    min_time_index = 1 - autoregressive_data.ar_lag_order
    max_dimension = get_max_dimension(autoregressive_data)
    process_state = Dict{Int64,Vector{Float64}}()

    # Determine the historical process states
    for t in min_time_index:0
        # Default case
        process_state[t] = ones(max_dimension)

        for ℓ in 1:max_dimension
            if haskey(user_process_state, t) && length(user_process_state[t]) <= ℓ && isa(user_process_state[t][ℓ], AbstractFloat)
                process_state[t][ℓ] = user_process_state[t][ℓ]
            end
        end
    end

    # Store the initial process state (also required for re-initialization later on)
    model.ext[:initial_process_state] = process_state
    model.nodes[1].ext[:process_state] = process_state

    return
end


""" 
Returns the maximum dimension of the random variables in the AR process over all stages.
Usually, the model should be defined such that the process dimension is the same for all stages.
"""
function get_max_dimension(
    autoregressive_data::LogLinearSDDP.AutoregressiveData
)

    max_dimension = 0

    for t in eachindex(autoregressive_data.ar_data)
        if autoregressive_data.ar_data[t].ar_dimension > max_dimension
            max_dimension = autoregressive_data.ar_data[t].ar_dimension
        end
    end

    return max_dimension
end



function get_lag_dimensions(
    autoregressive_data::LogLinearSDDP.AutoregressiveData,
    t::Int64,
)

    lag_dimensions = Vector{Int64}(undef, autoregressive_data.ar_lag_order)

    for k in eachindex(lag_dimensions)
        if t - k >= 1
            lag_dimensions[k] = autoregressive_data.ar_data[t-k].ar_dimension
        else
            lag_dimensions[k] = get_max_dimension(autoregressive_data)
        end
    end

    return lag_dimensions

end

