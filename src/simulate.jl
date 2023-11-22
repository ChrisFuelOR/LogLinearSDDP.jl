# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Copyright (c) 2023 Christian Fuellner <christian.fuellner@kit.edu>

# Note that this code reuses functions from SDDP.jl by Oscar Dowson,
# which are licensed under the Mozilla Public License, Version 2.0 as well. 
# Copyright (c) 2017-2023: Oscar Dowson and SDDP.jl contributors.
################################################################################

function _simulate(
    model::SDDP.PolicyGraph,
    ::SDDP.Serial,
    number_replications::Int64,
    variables::Vector{Symbol};
    kwargs...,
)
SDDP._initialize_solver(model; throw_error = false)
    return map(
        i -> _simulate(model, variables; kwargs...),
        1:number_replications,
    )
end


# Internal function: helper to conduct a single simulation. Users should use the
# documented, user-facing function SDDP.simulate instead.
function _simulate(
    model::SDDP.PolicyGraph{T},
    variables::Vector{Symbol};
    sampling_scheme::SDDP.AbstractSamplingScheme,
    custom_recorders::Dict{Symbol,Function},
    duality_handler::Union{Nothing,SDDP.AbstractDualityHandler},
    skip_undefined_variables::Bool,
    incoming_state::Dict{Symbol,Float64},
) where {T}

    # Sample a scenario path.
    scenario_path, _ = sample_scenario(model, sampling_scheme)

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
        subproblem_results = solve_subproblem(
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
function simulate(
    model::SDDP.PolicyGraph,
    number_replications::Int64 = 1,
    variables::Vector{Symbol} = Symbol[];
    sampling_scheme::SDDP.AbstractSamplingScheme = SDDP.InSampleMonteCarlo(),
    custom_recorders = Dict{Symbol,Function}(),
    duality_handler::Union{Nothing,SDDP.AbstractDualityHandler} = nothing,
    skip_undefined_variables::Bool = false,
    parallel_scheme::SDDP.AbstractParallelScheme = SDDP.Serial(),
    incoming_state::Dict{String,Float64} = SDDP._initial_state(model),
)
    return _simulate(
        model,
        parallel_scheme,
        number_replications,
        variables;
        sampling_scheme = sampling_scheme,
        custom_recorders = custom_recorders,
        duality_handler = duality_handler,
        skip_undefined_variables = skip_undefined_variables,
        incoming_state = Dict(Symbol(k) => v for (k, v) in incoming_state),
    )
end


function simulate_loglinear(
    model::SDDP.PolicyGraph,
    algo_params::LogLinearSDDP.AlgoParams,
    problem_params::LogLinearSDDP.ProblemParams,
    simulation_regime::LogLinearSDDP.Simulation
    )

    simulate_loglinear(model, algo_params, problem_params, simulation_regime.sampling_scheme)

    return
end


function simulate_loglinear(
    model::SDDP.PolicyGraph,
    algo_params::LogLinearSDDP.AlgoParams,
    problem_params::LogLinearSDDP.ProblemParams,
    simulation_regime::LogLinearSDDP.NoSimulation,
    )

    return
end


function simulate_loglinear(
    model::SDDP.PolicyGraph,
    algo_params::LogLinearSDDP.AlgoParams,
    problem_params::LogLinearSDDP.ProblemParams,
    sampling_scheme::Union{SDDP.InSampleMonteCarlo,SDDP.OutOfSampleMonteCarlo},
    )

    # SIMULATE THE MODEL
    ############################################################################
    simulations = simulate(model, algo_params.simulation_regime.number_of_replications)

    # OBTAINING BOUNDS AND CONFIDENCE INTERVAL
    ############################################################################
    objectives = map(simulations) do simulation
        return sum(stage[:stage_objective] for stage in simulation)
    end

    μ, ci = SDDP.confidence_interval(objectives)
    # get last lower bound again
    lower_bound = calculate_bound(model)

    # LOGGING OF SIMULATION RESULTS
    ############################################################################
    log_simulation_results(algo_params, μ, ci, lower_bound)

    return
end


function log_simulation_results(
    algo_params::LogLinearSDDP.AlgoParams,
    μ::Float64,
    ci::Float64,
    lower_bound::Float64
)

    log_file_handle = open(algo_params.log_file, "a")
    LogLinearSDDP.print_helper(LogLinearSDDP.print_simulation, log_file_handle, algo_params, μ, ci, lower_bound)
    close(log_file_handle)

    return
end


function simulate_linear(
    model::SDDP.PolicyGraph,
    algo_params::LogLinearSDDP.AlgoParams,
    problem_params::LogLinearSDDP.ProblemParams,
    simulation_regime::LogLinearSDDP.Simulation
    )

    simulate_linear(model, algo_params, problem_params, simulation_regime.sampling_scheme)

    return
end


function simulate_linear(
    model::SDDP.PolicyGraph,
    algo_params::LogLinearSDDP.AlgoParams,
    problem_params::LogLinearSDDP.ProblemParams,
    simulation_regime::LogLinearSDDP.NoSimulation,
    )

    return
end


function simulate_linear(
    model::SDDP.PolicyGraph,
    algo_params::LogLinearSDDP.AlgoParams,
    problem_params::LogLinearSDDP.ProblemParams,
    sampling_scheme::Union{SDDP.InSampleMonteCarlo,SDDP.OutOfSampleMonteCarlo},
    )

    # SIMULATE THE MODEL
    ############################################################################
    simulations = SDDP.simulate(model, algo_params.simulation_regime.number_of_replications)

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
    log_simulation_results(algo_params, μ, ci, lower_bound)

    return
end

