# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Copyright (c) 2023 Christian Fuellner <christian.fuellner@kit.edu>

# Note that this code reuses functions from SDDP.jl by Oscar Dowson,
# which are licensed under the Mozilla Public License, Version 2.0 as well. 
# Copyright (c) 2017-2023: Oscar Dowson and SDDP.jl contributors.
################################################################################

import LogLinearSDDP
import SDDP


include("set_up_ar_process.jl")
include("simulation.jl")

function cross_sample_scenario_ll(
    graph::SDDP.PolicyGraph{T},
    loglin_ar_process::LogLinearSDDP.AutoregressiveProcess,
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

        # Get the current process state
        process_state = node.ext[:process_state]

        # First stage is deterministic
        if node_index == 1
            noise_term = zeros(length(independent_noise_terms))
            for ℓ in eachindex(noise_term)    
                noise_term[ℓ] = loglin_ar_process.history[1][ℓ]
            end
        else      
            ar_process_stage = loglin_ar_process.parameters[node_index]
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
                lag_order = loglin_ar_process.lag_order
                lag_dimensions = LogLinearSDDP.get_lag_dimensions(loglin_ar_process, t)
                noise_term[ℓ] = exp(intercept) * exp(independent_term * error_term_factor) * prod(process_state[t-k][m]^coefficients[ℓ,m,k] for k in 1:lag_order for m in 1:lag_dimensions[k])
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
        graph[node_index].ext[:process_state] = LogLinearSDDP.update_process_state(graph, node_index, process_state, noise_term)
        ####################################################################################

    end
    # Throw an error because we should never end up here.
    return error(
        "Internal SDDP error: something went wrong sampling a scenario.",
    )
end

function solve_subproblem_cross(
    model::SDDP.PolicyGraph{T},
    node::SDDP.Node{T},
    state::Dict{Symbol,Float64},
    noise,
    scenario_path::Vector{Tuple{T,S}};
    duality_handler::Union{Nothing,SDDP.AbstractDualityHandler},
) where {T,S}
    SDDP._initialize_solver(node; throw_error = false)
    # Parameterize the model. First, fix the value of the incoming state
    # variables. Then parameterize the model depending on `noise`. Finally,
    # set the objective.
    SDDP.set_incoming_state(node, state)

    # Use adapted parameterize function instead of the one stored in the node
    # to fix the inflows to the realization of the loglinear process.
    for ℓ in 1:4
        JuMP.fix(node.subproblem[:inflow][ℓ].out, noise[ℓ], force=true)
        if node.index > 1 && JuMP.is_fixed(node.subproblem[:inflow_noise][ℓ])
            JuMP.unfix(node.subproblem[:inflow_noise][ℓ])
        end
    end
    SDDP.set_objective(node)
  
    JuMP.optimize!(node.subproblem)
    if haskey(model.ext, :total_solves)
        model.ext[:total_solves] += 1
    else
        model.ext[:total_solves] = 1
    end

    if JuMP.primal_status(node.subproblem) == JuMP.MOI.INTERRUPTED
        throw(InterruptException())
    end
    if JuMP.primal_status(node.subproblem) != JuMP.MOI.FEASIBLE_POINT
        SDDP.attempt_numerical_recovery(model, node)
    end
    state = SDDP.get_outgoing_state(node)
    stage_objective = SDDP.stage_objective_value(node.stage_objective)
    objective, dual_values = SDDP.get_dual_solution(node, duality_handler)
    
    return (
        state = state,
        duals = dual_values,
        objective = objective,
        stage_objective = stage_objective,
    )
end


function _cross_simulate_ll(
    model::SDDP.PolicyGraph,
    ::SDDP.Serial,
    loglin_ar_process::LogLinearSDDP.AutoregressiveProcess,
    number_replications::Int64,
    variables::Vector{Symbol};
    kwargs...,
)
SDDP._initialize_solver(model; throw_error = false)
    return map(
        i -> _cross_simulate_ll(model, loglin_ar_process, variables; kwargs...),
        1:number_replications,
    )
end


# Internal function: helper to conduct a single simulation. Users should use the
# documented, user-facing function SDDP.simulate instead.
function _cross_simulate_ll(
    model::SDDP.PolicyGraph{T},
    loglin_ar_process::LogLinearSDDP.AutoregressiveProcess,
    variables::Vector{Symbol};
    sampling_scheme::SDDP.AbstractSamplingScheme,
    custom_recorders::Dict{Symbol,Function},
    duality_handler::Union{Nothing,SDDP.AbstractDualityHandler},
    skip_undefined_variables::Bool,
    incoming_state::Dict{Symbol,Float64},
) where {T}

    # Initialize process state for the loglinear process
    LogLinearSDDP.initialize_process_state(model, loglin_ar_process)
    model.ext[:ar_process] = loglin_ar_process

    # Sample a scenario path.
    scenario_path, _ = cross_sample_scenario_ll(model, loglin_ar_process, sampling_scheme)

    # Storage for the simulation results.
    simulation = Dict{Symbol,Any}[]
    current_belief = SDDP.initialize_belief(model)
    # A cumulator for the stage-objectives.
    cumulative_value = 0.0

    # Objective state interpolation.
    objective_state_vector, N =
    SDDP.initialize_objective_state(model[scenario_path[1][1]])
    objective_states = NTuple{N,Float64}[]
    for (depth, (node_index, noise)) in enumerate(scenario_path)
        node = model[node_index]
        # Objective state interpolation.
        objective_state_vector = SDDP.update_objective_state(
            node.objective_state,
            objective_state_vector,
            noise,
        )
        if objective_state_vector !== nothing
            push!(objective_states, objective_state_vector)
        end
        if node.belief_state !== nothing
            belief = node.belief_state::SDDP.BeliefState{T}
            partition_index = belief.partition_index
            current_belief = belief.updater(
                belief.belief,
                current_belief,
                partition_index,
                noise,
            )
        else
            current_belief = Dict(node_index => 1.0)
        end
        # Solve the subproblem.
        subproblem_results = solve_subproblem_cross(
            model,
            node,
            incoming_state,
            noise,
            scenario_path[1:depth],
            duality_handler = duality_handler,
        )
        # Add the stage-objective
        cumulative_value += subproblem_results.stage_objective
        # Record useful variables from the solve.
        store = Dict{Symbol,Any}(
            :node_index => node_index,
            :noise_term => noise,
            :stage_objective => subproblem_results.stage_objective,
            :bellman_term =>
                subproblem_results.objective -
                subproblem_results.stage_objective,
            :objective_state => objective_state_vector,
            :belief => copy(current_belief),
        )
        if objective_state_vector !== nothing && N == 1
            store[:objective_state] = store[:objective_state][1]
        end
        # Loop through the primal variable values that the user wants.
        for variable in variables
            if haskey(node.subproblem.obj_dict, variable)
                # Note: we broadcast the call to value for variables which are
                # containers (like Array, Containers.DenseAxisArray, etc). If
                # the variable is a scalar (e.g. just a plain VariableRef), the
                # broadcast preseves the scalar shape.
                # TODO: what if the variable container is a dictionary? They
                # should be using Containers.SparseAxisArray, but this might not
                # always be the case...
                store[variable] = JuMP.value.(node.subproblem[variable])
            elseif skip_undefined_variables
                store[variable] = NaN
            else
                error(
                    "No variable named $(variable) exists in the subproblem.",
                    " If you want to simulate the value of a variable, make ",
                    "sure it is defined in _all_ subproblems, or pass ",
                    "`skip_undefined_variables=true` to `simulate`.",
                )
            end
        end
        # Loop through any custom recorders that the user provided.
        for (sym, recorder) in custom_recorders
            store[sym] = recorder(node.subproblem)
        end
        # Add the store to our list.
        push!(simulation, store)
        # Set outgoing state as the incoming state for the next node.
        incoming_state = copy(subproblem_results.state)
    end
    return simulation
end


"""
Perform a simulation of the policy model with `number_replications` replications
using the sampling scheme `sampling_scheme`.

Returns a vector with one element for each replication. Each element is a vector
with one-element for each node in the scenario that was sampled. Each element in
that vector is a dictionary containing information about the subproblem that was
solved.

In that dictionary there are four special keys:
 - :node_index, which records the index of the sampled node in the policy model
 - :noise_term, which records the noise observed at the node
 - :stage_objective, which records the stage-objective of the subproblem
 - :bellman_term, which records the cost/value-to-go of the node.
The sum of :stage_objective + :bellman_term will equal the objective value of
the solved subproblem.

In addition to the special keys, the dictionary will contain the result of
`JuMP.value(subproblem[key])` for each `key` in `variables`. This is
useful to obtain the primal value of the state and control variables.

For more complicated data, the `custom_recorders` keyword argument can be used.
"""
function cross_simulate_ll(
    model::SDDP.PolicyGraph,
    loglin_ar_process::LogLinearSDDP.AutoregressiveProcess,
    number_replications::Int64 = 1,
    variables::Vector{Symbol} = Symbol[];
    sampling_scheme::SDDP.AbstractSamplingScheme = SDDP.InSampleMonteCarlo(),
    custom_recorders = Dict{Symbol,Function}(),
    duality_handler::Union{Nothing,SDDP.AbstractDualityHandler} = nothing,
    skip_undefined_variables::Bool = false,
    parallel_scheme::SDDP.AbstractParallelScheme = SDDP.Serial(),
    incoming_state::Dict{String,Float64} = SDDP._initial_state(model),
)
    return _cross_simulate_ll(
        model,
        parallel_scheme,
        loglin_ar_process,
        number_replications,
        variables;
        sampling_scheme = sampling_scheme,
        custom_recorders = custom_recorders,
        duality_handler = duality_handler,
        skip_undefined_variables = skip_undefined_variables,
        incoming_state = Dict(Symbol(k) => v for (k, v) in incoming_state),
    )
end


function cross_simulate_loglinear(
    model::SDDP.PolicyGraph,
    algo_params::LogLinearSDDP.AlgoParams,
    loglin_ar_process::LogLinearSDDP.AutoregressiveProcess,
    description::String,
    simulation_regime::LogLinearSDDP.Simulation
    )

    cross_simulate_loglinear(model, algo_params, loglin_ar_process, description, simulation_regime.number_of_replications, simulation_regime.sampling_scheme)

    return
end


function cross_simulate_loglinear(
    model::SDDP.PolicyGraph,
    algo_params::LogLinearSDDP.AlgoParams,
    loglin_ar_process::LogLinearSDDP.AutoregressiveProcess,
    description::String,
    simulation_regime::LogLinearSDDP.NoSimulation,
    )

    return
end


function cross_simulate_loglinear(
    model::SDDP.PolicyGraph,
    algo_params::LogLinearSDDP.AlgoParams,
    loglin_ar_process::LogLinearSDDP.AutoregressiveProcess,
    description::String,
    number_of_replications::Int64,
    sampling_scheme::Union{SDDP.InSampleMonteCarlo,SDDP.OutOfSampleMonteCarlo},
    )

    # SIMULATE THE MODEL
    ############################################################################
    simulations = cross_simulate_ll(model, loglin_ar_process, number_of_replications, sampling_scheme=sampling_scheme)

    # OBTAINING BOUNDS AND CONFIDENCE INTERVAL
    ############################################################################
    objectives = map(simulations) do simulation
        return sum(stage[:stage_objective] for stage in simulation)
    end

    μ, ci = SDDP.confidence_interval(objectives)
    # get last lower bound again
    lower_bound = SDDP.calculate_bound(model)

    # LOGGING OF SIMULATION RESULTS
    ############################################################################
    LogLinearSDDP.log_simulation_results(algo_params, μ, ci, lower_bound, description)

    return
end
