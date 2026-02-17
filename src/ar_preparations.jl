# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Copyright (c) 2026 Christian Fuellner <christian.fuellner@kit.edu>
################################################################################

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

    min_time_index = minimum(collect(keys(ar_process.history)))
    dim = ar_process.dimension
    process_state = Dict{Int64,Any}()

    # Determine the historical process states
    for t in min_time_index:0
        # Default case if not a sufficient amount of history is defined
        process_state[t] = ones(dim)

        for ℓ in 1:dim
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