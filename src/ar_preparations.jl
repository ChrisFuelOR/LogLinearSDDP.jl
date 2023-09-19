""" 
Default function to initialize the history of the stochastic process.

As we do not know explicitly which lagged values are required for which dimension 
at later stages, we defined values for the worst possible case.

Per default, we define historical values of ξ as ones.
"""
function initialize_process_state(
    model::SDDP.PolicyGraph,
    ar_process::LogLinearSDDP.AutoregressiveProcess,
)

    min_time_index = 1 - ar_process.lag_order
    max_dimension = get_max_dimension(ar_process)
    process_state = Dict{Int64,Any}()

    # Determine the historical process states
    for t in min_time_index:0
        # Default case if not a sufficient amount of history is defined
        process_state[t] = ones(max_dimension)

        for ℓ in 1:max_dimension
            if haskey(ar_process.history, t) && length(ar_process.history[t]) >= ℓ && isa(ar_process.history[t][ℓ], AbstractFloat)
                process_state[t][ℓ] = ar_process.history[t][ℓ]
            end
        end    
    end

    # Store the initial process state
    model.ext[:initial_process_state] = process_state
    model.nodes[1].ext[:process_state] = process_state

    return
end


""" 
Returns the maximum dimension of the random variables in the AR process over all stages.
Usually, the model should be defined such that the process dimension is the same for all stages.
"""
function get_max_dimension(
    ar_process::LogLinearSDDP.AutoregressiveProcess
)

    max_dimension = 0

    for t in eachindex(ar_process.parameters)
        if ar_process.parameters[t].dimension > max_dimension
            max_dimension = ar_process.parameters[t].dimension
        end
    end

    return max_dimension
end



function get_lag_dimensions(
    ar_process::LogLinearSDDP.AutoregressiveProcess,
    t::Int64,
)

    lag_dimensions = Vector{Int64}(undef, ar_process.lag_order)

    for k in eachindex(lag_dimensions)
        if t - k > 1
            lag_dimensions[k] = ar_process.parameters[t-k].dimension
        else
            lag_dimensions[k] = get_max_dimension(ar_process)
        end
    end

    return lag_dimensions

end

