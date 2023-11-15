# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Copyright (c) 2023 Christian Fuellner <christian.fuellner@kit.edu>

# Note that this code reuses functions from SDDP.jl by Oscar Dowson,
# which are licensed under the Mozilla Public License, Version 2.0 as well. 
# Copyright (c) 2017-2023: Oscar Dowson and SDDP.jl contributors.
################################################################################

# Internal function: set the objective of node to the stage objective, plus the
# cost/value-to-go term.
function set_objective(node::SDDP.Node{T}) where {T}
    objective_state_component = SDDP.get_objective_state_component(node)
    belief_state_component = SDDP.get_belief_state_component(node)
    if objective_state_component != JuMP.AffExpr(0.0) ||
       belief_state_component != JuMP.AffExpr(0.0)
        node.stage_objective_set = false
    end
    if !node.stage_objective_set
        JuMP.set_objective(
            node.subproblem,
            JuMP.objective_sense(node.subproblem),
            JuMP.@expression(
                node.subproblem,
                node.stage_objective +
                objective_state_component +
                belief_state_component +
                bellman_term(node.bellman_function)
            )
        )
    end
    node.stage_objective_set = true
    return
end

"""
    parameterize(node::Node, noise)

Parameterize node `node` with the noise `noise`.

For log-linear AR processes, this includes the adaptation of 
the existing cuts to the scenario at hand.
"""
function parameterize(
    node::SDDP.Node, 
    noise_term
)
    node.parameterize(noise_term)
    set_objective(node)
    
    model = SDDP.get_policy_graph(node.subproblem)
    TimerOutputs.@timeit model.timer_output "evaluate_cut_intercepts" begin
        evaluate_cut_intercepts(node, noise_term)
    end
    
    return
end       


# Internal struct: storage for SDDP options and cached data. Users shouldn't
# interact with this directly.
struct Options{T}
    # The initial state to start from the root node.
    initial_state::Dict{Symbol,Float64}
    # The sampling scheme to use on the forward pass.
    sampling_scheme::SDDP.AbstractSamplingScheme
    backward_sampling_scheme::SDDP.AbstractBackwardSamplingScheme
    # Storage for the set of possible sampling states at each node. We only use
    # this if there is a cycle in the policy graph.
    starting_states::Dict{T,Vector{Dict{Symbol,Float64}}}
    # Risk measure to use at each node.
    risk_measures::Dict{T,SDDP.AbstractRiskMeasure}
    # The delta by which to check if a state is close to a previously sampled
    # state.
    cycle_discretization_delta::Float64
    # Flag to add cuts to similar nodes.
    refine_at_similar_nodes::Bool
    # The node transition matrix.
    Φ::Dict{Tuple{T,T},Float64}
    # A list of nodes that contain a subset of the children of node i.
    similar_children::Dict{T,Vector{T}}
    stopping_rules::Vector{SDDP.AbstractStoppingRule}
    dashboard_callback::Function
    print_level::Int
    start_time::Float64
    log::Vector{SDDP.Log}
    log_file_handle::Any
    log_frequency::Union{Int,Function}
    forward_pass::SDDP.AbstractForwardPass
    duality_handler::SDDP.AbstractDualityHandler
    # A callback called after the forward pass.
    forward_pass_callback::Any
    post_iteration_callback::Any
    last_log_iteration::Ref{Int}
    # Internal function: users should never construct this themselves.
    function Options(
        model::SDDP.PolicyGraph{T},
        initial_state::Dict{Symbol,Float64};
        sampling_scheme::SDDP.AbstractSamplingScheme = InSampleMonteCarlo(),
        backward_sampling_scheme::SDDP.AbstractBackwardSamplingScheme = CompleteSampler(),
        risk_measures = SDDP.Expectation(),
        cycle_discretization_delta::Float64 = 0.0,
        refine_at_similar_nodes::Bool = true,
        stopping_rules::Vector{SDDP.AbstractStoppingRule} = SDDP.AbstractStoppingRule[],
        dashboard_callback::Function = (a, b) -> nothing,
        print_level::Int = 0,
        start_time::Float64 = 0.0,
        log::Vector{SDDP.Log} = SDDP.Log[],
        log_file_handle = IOBuffer(),
        log_frequency::Union{Int,Function} = 1,
        forward_pass::SDDP.AbstractForwardPass = SDDP.DefaultForwardPass(),
        duality_handler::SDDP.AbstractDualityHandler = ContinuousConicDuality(),
        forward_pass_callback = x -> nothing,
        post_iteration_callback = result -> nothing,
    ) where {T}
        return new{T}(
            initial_state,
            sampling_scheme,
            backward_sampling_scheme,
            SDDP.to_nodal_form(model, x -> Dict{Symbol,Float64}[]),
            SDDP.to_nodal_form(model, risk_measures),
            cycle_discretization_delta,
            refine_at_similar_nodes,
            SDDP.build_Φ(model),
            SDDP.get_same_children(model),
            stopping_rules,
            dashboard_callback,
            print_level,
            start_time,
            log,
            log_file_handle,
            log_frequency,
            forward_pass,
            duality_handler,
            forward_pass_callback,
            post_iteration_callback,
            Ref{Int}(0),  # last_log_iteration
        )
    end
end


# Internal function: solve the subproblem associated with node given the
# incoming state variables state and realization of the stagewise-independent
# noise term noise.
function solve_subproblem(
    model::SDDP.PolicyGraph{T},
    node::SDDP.Node{T},
    state::Dict{Symbol,Float64},
    noise_term,
    scenario_path::Vector{Tuple{T,S}};
    duality_handler::Union{Nothing,SDDP.AbstractDualityHandler},
) where {T,S}
    SDDP._initialize_solver(node; throw_error = false)
    # Parameterize the model. First, fix the value of the incoming state
    # variables. Then parameterize the model depending on `noise`. Finally,
    # set the objective.
    SDDP.set_incoming_state(node, state)
    LogLinearSDDP.parameterize(node, noise_term)
    pre_optimize_ret = if node.pre_optimize_hook !== nothing
        node.pre_optimize_hook(
            model,
            node,
            state,
            noise_term,
            scenario_path,
            duality_handler,
        )
    else
        nothing
    end

    TimerOutputs.@timeit model.timer_output "solver_call" begin
        JuMP.optimize!(node.subproblem)
    end
   
    if haskey(model.ext, :total_solves)
        model.ext[:total_solves] += 1
    else
        model.ext[:total_solves] = 1
    end
    if JuMP.primal_status(node.subproblem) == JuMP.MOI.INTERRUPTED
        # If the solver was interrupted, the user probably hit CTRL+C but the
        # solver gracefully exited. Since we're in the middle of training or
        # simulation, we need to throw an interrupt exception to keep the
        # interrupt percolating up to the user.
        throw(InterruptException())
    end
    if JuMP.primal_status(node.subproblem) != JuMP.MOI.FEASIBLE_POINT
        SDDP.attempt_numerical_recovery(model, node)
    end
    state = SDDP.get_outgoing_state(node)
    stage_objective = SDDP.stage_objective_value(node.stage_objective)

    # CHANGES TO SDDP.jl
    ####################################################################################
    TimerOutputs.@timeit model.timer_output "get_dual_solution" begin
        objective, dual_values, intercept_factors = get_dual_solution(node, duality_handler)
    end
    ####################################################################################
    if node.post_optimize_hook !== nothing
        node.post_optimize_hook(pre_optimize_ret)
    end
    return (
        state = state,
        duals = dual_values,
        objective = objective,
        stage_objective = stage_objective,
        intercept_factors = intercept_factors,
    )
end


struct BackwardPassItems{T,U}
    "Given a (node, noise) tuple, index the element in the array."
    cached_solutions::Dict{Tuple{T,Any},Int}
    duals::Vector{Dict{Symbol,Float64}}
    intercept_factors::Vector{Array{Float64,2}}
    supports::Vector{U}
    nodes::Vector{T}
    probability::Vector{Float64}
    objectives::Vector{Float64}
    belief::Vector{Float64}
    function BackwardPassItems(T, U)
        return new{T,U}(
            Dict{Tuple{T,Any},Int}(),
            Dict{Symbol,Float64}[],
            Array{Float64,2}[],
            U[],
            T[],
            Float64[],
            Float64[],
            Float64[],
        )
    end
end


"""
Calculate the lower bound (if minimizing, otherwise upper bound) of the problem
model at the point state, assuming the risk measure at the root node is
risk_measure.
"""
function calculate_bound(
    model::SDDP.PolicyGraph{T},
    root_state::Dict{Symbol,Float64} = model.initial_root_state;
    risk_measure::SDDP.AbstractRiskMeasure = SDDP.Expectation(),
) where {T}
    # Initialization.
    noise_supports = Any[]
    probabilities = Float64[]
    objectives = Float64[]
    current_belief = SDDP.initialize_belief(model)
    # Solve all problems that are children of the root node.
    for child in model.root_children
        if isapprox(child.probability, 0.0, atol = 1e-6)
            continue
        end
        node = model[child.term]
        for noise in node.noise_terms
            # CHANGES TO SDDP.jl
            ####################################################################################
            ar_process = model.ext[:ar_process]
            noise_term = zeros(length(noise.term))
            for ℓ in eachindex(noise_term)    
                noise_term[ℓ] = ar_process.history[1][ℓ]
            end

            if node.objective_state !== nothing
                SDDP.update_objective_state(
                    node.objective_state,
                    node.objective_state.initial_value,
                    noise_term,
                )
            end
            # Update belief state, etc.
            if node.belief_state !== nothing
                belief = node.belief_state::SDDP.BeliefState{T}
                partition_index = belief.partition_index
                belief.updater(
                    belief.belief,
                    current_belief,
                    partition_index,
                    noise_term,
                )
            end
            subproblem_results = solve_subproblem(
                model,
                node,
                root_state,
                noise_term,
                Tuple{T,Any}[(child.term, noise)],
                duality_handler = nothing,
            )
            push!(objectives, subproblem_results.objective)
            push!(probabilities, child.probability * noise.probability)
            push!(noise_supports, noise_term)
        end
    end
    # Now compute the risk-adjusted probability measure:
    risk_adjusted_probability = similar(probabilities)
    offset = SDDP.adjust_probability(
        risk_measure,
        risk_adjusted_probability,
        probabilities,
        noise_supports,
        objectives,
        model.objective_sense == MOI.MIN_SENSE,
    )
    # Finally, calculate the risk-adjusted value.
    return sum(
        obj * prob for (obj, prob) in zip(objectives, risk_adjusted_probability)
    ) + offset
end

# Internal function: perform a backward pass of the SDDP algorithm along the
# scenario_path, refining the bellman function at sampled_states. Assumes that
# scenario_path does not end in a leaf node (i.e., the forward pass was solved
# with include_last_node = false)
function backward_pass(
    model::SDDP.PolicyGraph{T},
    options::LogLinearSDDP.Options,
    scenario_path::Vector{Tuple{T,NoiseType}},
    sampled_states::Vector{Dict{Symbol,Float64}},
    objective_states::Vector{NTuple{N,Float64}},
    belief_states::Vector{Tuple{Int,Dict{T,Float64}}},
) where {T,NoiseType,N}
    TimerOutputs.@timeit model.timer_output "prepare_backward_pass" begin
        restore_duality =
            prepare_backward_pass(model, options.duality_handler, options)
    end
    # TODO(odow): improve storage type.
    cuts = Dict{T,Vector{Any}}(index => Any[] for index in keys(model.nodes))
    for index in length(scenario_path):-1:1
        outgoing_state = sampled_states[index]
        objective_state = get(objective_states, index, nothing)
        partition_index, belief_state = get(belief_states, index, (0, nothing))
        items = BackwardPassItems(T, SDDP.Noise)
        if belief_state !== nothing
            # Update the cost-to-go function for partially observable model.
            for (node_index, belief) in belief_state
                if iszero(belief)
                    continue
                end
                solve_all_children(
                    model,
                    model[node_index],
                    items,
                    belief,
                    belief_state,
                    objective_state,
                    outgoing_state,
                    options.backward_sampling_scheme,
                    scenario_path[1:index],
                    options.duality_handler,
                )
            end
            # We need to refine our estimate at all nodes in the partition.
            for node_index in model.belief_partition[partition_index]
                node = model[node_index]
                # Update belief state, etc.
                current_belief = node.belief_state::BeliefState{T}
                for (idx, belief) in belief_state
                    current_belief.belief[idx] = belief
                end
                new_cuts = refine_bellman_function(
                    model,
                    node,
                    node.bellman_function,
                    options.risk_measures[node_index],
                    outgoing_state,
                    items.duals,
                    items.supports,
                    items.probability .* items.belief,
                    items.objectives,
                    items.intercept_factors,
                )
                push!(cuts[node_index], new_cuts)
            end
        else
            node_index, _ = scenario_path[index]
            node = model[node_index]
            if length(node.children) == 0
                continue
            end
            solve_all_children(
                model,
                node,
                items,
                1.0,
                belief_state,
                objective_state,
                outgoing_state,
                options.backward_sampling_scheme,
                scenario_path[1:index],
                options.duality_handler,
            )
            new_cuts = refine_bellman_function(
                model,
                node,
                node.bellman_function,
                options.risk_measures[node_index],
                outgoing_state,
                items.duals,
                items.supports,
                items.probability,
                items.objectives,
                items.intercept_factors,
            )
            push!(cuts[node_index], new_cuts)
            if options.refine_at_similar_nodes
                # Refine the bellman function at other nodes with the same
                # children, e.g., in the same stage of a Markovian policy graph.
                for other_index in options.similar_children[node_index]
                    copied_probability = similar(items.probability)
                    other_node = model[other_index]
                    for (idx, child_index) in enumerate(items.nodes)
                        copied_probability[idx] =
                            get(options.Φ, (other_index, child_index), 0.0) *
                            items.supports[idx].probability
                    end

                    TimerOutputs.@timeit model.timer_output "bellman_update" begin
                        new_cuts = refine_bellman_function(
                            model,
                            other_node,
                            other_node.bellman_function,
                            options.risk_measures[other_index],
                            outgoing_state,
                            items.duals,
                            items.supports,
                            copied_probability,
                            items.objectives,
                            items.intercept_factors,
                        )
                    end
                    push!(cuts[other_index], new_cuts)
                end
            end
        end
    end
    TimerOutputs.@timeit model.timer_output "prepare_backward_pass" begin
        restore_duality()
    end
    return cuts
end


function solve_all_children(
    model::SDDP.PolicyGraph{T},
    node::SDDP.Node{T},
    items::BackwardPassItems,
    belief::Float64,
    belief_state,
    objective_state,
    outgoing_state::Dict{Symbol,Float64},
    backward_sampling_scheme::CompleteSampler,
    scenario_path,
    duality_handler::Union{Nothing,SDDP.AbstractDualityHandler},
) where {T}
    length_scenario_path = length(scenario_path)
    for child in node.children
        if isapprox(child.probability, 0.0, atol = 1e-6)
            continue
        end
        child_node = model[child.term]

        TimerOutputs.@timeit model.timer_output "get_backward_noise" begin
            sampling_output = sample_backward_noise_terms(backward_sampling_scheme, child_node)
        end

        for noise_index in eachindex(sampling_output.all_dependent_noises)
            noise = sampling_output.all_dependent_noises[noise_index]
            child_node.ext[:current_independent_noise_term] = sampling_output.all_independent_noises[noise_index].term
            
            if length(scenario_path) == length_scenario_path
                push!(scenario_path, (child.term, noise.term))
            else
                scenario_path[end] = (child.term, noise.term)
            end
            if haskey(items.cached_solutions, (child.term, noise.term))
                sol_index = items.cached_solutions[(child.term, noise.term)]
                push!(items.duals, items.duals[sol_index])
                push!(items.intercept_factors, items.intercept_factors[sol_index])
                push!(items.supports, items.supports[sol_index])
                push!(items.nodes, child_node.index)
                push!(items.probability, items.probability[sol_index])
                push!(items.objectives, items.objectives[sol_index])
                push!(items.belief, belief)
            else
                # Update belief state, etc.
                if belief_state !== nothing
                    current_belief = child_node.belief_state::BeliefState{T}
                    current_belief.updater(
                        current_belief.belief,
                        belief_state,
                        current_belief.partition_index,
                        noise.term,
                    )
                end
                if objective_state !== nothing
                    update_objective_state(
                        child_node.objective_state,
                        objective_state,
                        noise.term,
                    )
                end
                TimerOutputs.@timeit model.timer_output "solve_subproblem" begin
                    subproblem_results = solve_subproblem(
                        model,
                        child_node,
                        outgoing_state,
                        noise.term,
                        scenario_path,
                        duality_handler = duality_handler,
                    )
                end
                push!(items.duals, subproblem_results.duals)
                push!(items.intercept_factors, subproblem_results.intercept_factors)
                push!(items.supports, noise)
                push!(items.nodes, child_node.index)
                push!(items.probability, child.probability * noise.probability)
                push!(items.objectives, subproblem_results.objective)
                push!(items.belief, belief)
                items.cached_solutions[(child.term, noise.term)] =
                    length(items.duals)
            end
        end
    end
    if length(scenario_path) == length_scenario_path
        # No-op. There weren't any children to solve.
    else
        # Drop the last element (i.e., the one we added).
        pop!(scenario_path)
    end
    return
end


function forward_pass(
    model::SDDP.PolicyGraph{T},
    options::LogLinearSDDP.Options,
    pass::SDDP.DefaultForwardPass,
) where {T}
    # First up, sample a scenario. Note that if a cycle is detected, this will
    # return the cycle node as well.
    TimerOutputs.@timeit model.timer_output "sample_scenario" begin
        scenario_path, terminated_due_to_cycle =
            sample_scenario(model, options.sampling_scheme)
    end
    final_node = scenario_path[end]
    if terminated_due_to_cycle && !pass.include_last_node
        pop!(scenario_path)
    end
    # Storage for the list of outgoing states that we visit on the forward pass.
    sampled_states = Dict{Symbol,Float64}[]
    # Storage for the belief states: partition index and the belief dictionary.
    belief_states = Tuple{Int,Dict{T,Float64}}[]
    current_belief = SDDP.initialize_belief(model)
    # Our initial incoming state.
    incoming_state_value = copy(options.initial_state)
    # A cumulator for the stage-objectives.
    cumulative_value = 0.0
    # Objective state interpolation.
    objective_state_vector, N =
        SDDP.initialize_objective_state(model[scenario_path[1][1]])
    objective_states = NTuple{N,Float64}[]
    # Iterate down the scenario.
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
        # Update belief state, etc.
        if node.belief_state !== nothing
            belief = node.belief_state::SDDP.BeliefState{T}
            partition_index = belief.partition_index
            current_belief = belief.updater(
                belief.belief,
                current_belief,
                partition_index,
                noise,
            )
            push!(belief_states, (partition_index, copy(current_belief)))
        end
        # ===== Begin: starting state for infinite horizon =====
        starting_states = options.starting_states[node_index]
        if length(starting_states) > 0
            # There is at least one other possible starting state. If our
            # incoming state is more than δ away from the other states, add it
            # as a possible starting state.
            if distance(starting_states, incoming_state_value) >
               options.cycle_discretization_delta
                push!(starting_states, incoming_state_value)
            end
            # TODO(odow):
            # - A better way of randomly sampling a starting state.
            # - Is is bad that we splice! here instead of just sampling? For
            #   convergence it is probably bad, since our list of possible
            #   starting states keeps changing, but from a computational
            #   perspective, we don't want to keep a list of discretized points
            #   in the state-space δ distance apart...
            incoming_state_value =
                splice!(starting_states, rand(1:length(starting_states)))
        end
        # ===== End: starting state for infinite horizon =====
        # Solve the subproblem, note that `duality_handler = nothing`.
        TimerOutputs.@timeit model.timer_output "solve_subproblem" begin
            subproblem_results = solve_subproblem(
                model,
                node,
                incoming_state_value,
                noise,
                scenario_path[1:depth],
                duality_handler = nothing,
            )
        end
        # Cumulate the stage_objective.
        cumulative_value += subproblem_results.stage_objective
        # Set the outgoing state value as the incoming state value for the next
        # node.
        incoming_state_value = copy(subproblem_results.state)
        # Add the outgoing state variable to the list of states we have sampled
        # on this forward pass.
        push!(sampled_states, incoming_state_value)
    end
    if terminated_due_to_cycle
        # We terminated due to a cycle. Here is the list of possible starting
        # states for that node:
        starting_states = options.starting_states[final_node[1]]
        # We also need the incoming state variable to the final node, which is
        # the outgoing state value of the second to last node:
        incoming_state_value = if pass.include_last_node
            sampled_states[end-1]
        else
            sampled_states[end]
        end
        # If this incoming state value is more than δ away from another state,
        # add it to the list.
        if distance(starting_states, incoming_state_value) >
           options.cycle_discretization_delta
            push!(starting_states, incoming_state_value)
        end
    end
    # ===== End: drop off starting state if terminated due to cycle =====
    #println(scenario_path)
    return (
        scenario_path = scenario_path,
        sampled_states = sampled_states,
        objective_states = objective_states,
        belief_states = belief_states,
        cumulative_value = cumulative_value,
    )
end


function iteration(model::SDDP.PolicyGraph{T}, options::LogLinearSDDP.Options) where {T}
    model.ext[:numerical_issue] = false
    TimerOutputs.@timeit model.timer_output "forward_pass" begin
        forward_trajectory = forward_pass(model, options, options.forward_pass)
        options.forward_pass_callback(forward_trajectory)
    end
    TimerOutputs.@timeit model.timer_output "backward_pass" begin
        cuts = backward_pass(
            model,
            options,
            forward_trajectory.scenario_path,
            forward_trajectory.sampled_states,
            forward_trajectory.objective_states,
            forward_trajectory.belief_states,
        )
    end
    TimerOutputs.@timeit model.timer_output "calculate_bound" begin
        bound = calculate_bound(model)
    end
   
    #Infiltrator.@infiltrate
   
    push!(
        options.log,
        SDDP.Log(
            model.ext[:iteration], #length(options.log) + 1,
            bound,
            forward_trajectory.cumulative_value,
            #forward_trajectory.sampled_states,
            time() - options.start_time,
            Distributed.myid(),
            #TODO: total_cuts
            model.ext[:total_solves],
            duality_log_key(options.duality_handler),
            model.ext[:numerical_issue],
        ),
    )
    has_converged, status =
    SDDP.convergence_test(model, options.log, options.stopping_rules)
    return SDDP.IterationResult(
        Distributed.myid(),
        bound,
        forward_trajectory.cumulative_value,
        has_converged,
        status,
        cuts,
        model.ext[:numerical_issue],
    )
end


function master_loop(
    ::SDDP.Serial,
    model::SDDP.PolicyGraph{T},
    options::LogLinearSDDP.Options,
) where {T}
SDDP._initialize_solver(model; throw_error = false)
model.ext[:iteration] = 0
    while true
        model.ext[:iteration] += 1
        result = iteration(model, options)
        options.post_iteration_callback(result)
        log_iteration(model.ext[:algo_params], options.log_file_handle, options.log)

        if result.has_converged
            return result.status
        end
    end
    return
end


"""
    SDDP.train(model::PolicyGraph; kwargs...)

Train the policy for `model`.

## Keyword arguments: see SDDP.jl

"""
function train(
    model::SDDP.PolicyGraph;
    iteration_limit::Union{Int,Nothing} = nothing,
    time_limit::Union{Real,Nothing} = nothing,
    print_level::Int = 1,
    log_file::String = "SDDP.log",
    log_frequency::Int = 1,
    log_every_seconds::Float64 = log_frequency == 1 ? -1.0 : 0.0,
    run_numerical_stability_report::Bool = false, # TODO: Not implemented yet
    stopping_rules = SDDP.AbstractStoppingRule[],
    risk_measure = SDDP.Expectation(),
    sampling_scheme = SDDP.InSampleMonteCarlo(),
    cut_type = SDDP.SINGLE_CUT,
    cycle_discretization_delta::Float64 = 0.0,
    refine_at_similar_nodes::Bool = true,
    cut_deletion_minimum::Int = 1,
    backward_sampling_scheme::SDDP.AbstractBackwardSamplingScheme = CompleteSampler(),
    dashboard::Bool = false,
    parallel_scheme::SDDP.AbstractParallelScheme = SDDP.Serial(),
    forward_pass::SDDP.AbstractForwardPass = SDDP.DefaultForwardPass(),
    forward_pass_resampling_probability::Union{Nothing,Float64} = nothing,
    add_to_existing_cuts::Bool = false,
    duality_handler::SDDP.AbstractDualityHandler = SDDP.ContinuousConicDuality(),
    forward_pass_callback::Function = (x) -> nothing,
    post_iteration_callback = result -> nothing,
)
    function log_frequency_f(log::Vector{SDDP.Log})
        if mod(length(log), log_frequency) != 0
            return false
        end
        last = options.last_log_iteration[]
        if last == 0
            return true
        elseif last == length(log)
            return false
        end
        seconds = log_every_seconds
        if log_every_seconds < 0.0
            if log[end].time <= 10
                seconds = 1.0
            elseif log[end].time <= 120
                seconds = 5.0
            else
                seconds = 30.0
            end
        end
        return log[end].time - log[last].time >= seconds
    end

    if !add_to_existing_cuts && model.most_recent_training_results !== nothing
        @warn("""
        Re-training a model with existing cuts!

        Are you sure you want to do this? The output from this training may be
        misleading because the policy is already partially trained.

        If you meant to train a new policy with different settings, you must
        build a new model.

        If you meant to refine a previously trained policy, turn off this
        warning by passing `add_to_existing_cuts = true` as a keyword argument
        to `SDDP.train`.

        In a future release, this warning may turn into an error.
        """)
    end
    if forward_pass_resampling_probability !== nothing
        forward_pass = SDDP.RiskAdjustedForwardPass(
            forward_pass = forward_pass,
            risk_measure = risk_measure,
            resampling_probability = forward_pass_resampling_probability,
        )
    end

    log_file_handle = open(log_file, "a")
    log = SDDP.Log[]

    if print_level > 0
        SDDP.print_helper(print_banner, log_file_handle)
        SDDP.print_helper(
            SDDP.print_problem_statistics,
            log_file_handle,
            model,
            model.most_recent_training_results !== nothing,
            parallel_scheme,
            risk_measure,
            sampling_scheme,
        )
    end
    if run_numerical_stability_report
        report = sprint(
            io -> SDDP.numerical_stability_report(
                io,
                model,
                print = print_level > 0,
            ),
        )
        SDDP.print_helper(print, log_file_handle, report)
    end

    if print_level > 1
        print_helper(print_parameters, log_file_handle, model.ext[:algo_params], model.ext[:problem_params], model.ext[:applied_solver], model.ext[:ar_process])
    end

    if print_level > 0
        SDDP.print_helper(print_iteration_header, log_file_handle)
    end
    # Convert the vector to an AbstractStoppingRule. Otherwise if the user gives
    # something like stopping_rules = [SDDP.IterationLimit(100)], the vector
    # will be concretely typed and we can't add a TimeLimit.
    stopping_rules = convert(Vector{SDDP.AbstractStoppingRule}, stopping_rules)
    if isempty(stopping_rules)
        push!(stopping_rules, SDDP.SimulationStoppingRule())
    end
    # Add the limits as stopping rules. An IterationLimit or TimeLimit may
    # already exist in stopping_rules, but that doesn't matter.
    if iteration_limit !== nothing
        push!(stopping_rules, SDDP.IterationLimit(iteration_limit))
    end
    if time_limit !== nothing
        push!(stopping_rules, SDDP.TimeLimit(time_limit))
    end
    # Update the nodes with the selected cut type (SINGLE_CUT or MULTI_CUT)
    # and the cut deletion minimum.
    if cut_deletion_minimum < 0
        cut_deletion_minimum = typemax(Int)
    end
    for (_, node) in model.nodes
        node.bellman_function.cut_type = cut_type
        node.bellman_function.global_theta.deletion_minimum =
            cut_deletion_minimum
        for oracle in node.bellman_function.local_thetas
            oracle.deletion_minimum = cut_deletion_minimum
        end
    end
    dashboard_callback = if dashboard
        SDDP.launch_dashboard()
    else
        (::Any, ::Any) -> nothing
    end

    options = LogLinearSDDP.Options(
        model,
        model.initial_root_state;
        sampling_scheme,
        backward_sampling_scheme,
        risk_measures = risk_measure,
        cycle_discretization_delta,
        refine_at_similar_nodes,
        stopping_rules,
        dashboard_callback,
        print_level,
        start_time = time(),
        log,
        log_file_handle,
        log_frequency = log_frequency_f,
        forward_pass,
        duality_handler,
        forward_pass_callback,
        # post_iteration_callback,
    )
    status = :not_solved
    try
        TimerOutputs.@timeit model.timer_output "loop" begin
            status = master_loop(parallel_scheme, model, options)
        end
    catch ex
        if isa(ex, InterruptException)
            status = :interrupted
            interrupt(parallel_scheme)
        else
            close(log_file_handle)
            rethrow(ex)
        end
    finally
        # And close the dashboard callback if necessary.
        dashboard_callback(nothing, true)
    end
    training_results = SDDP.TrainingResults(status, log)
    model.most_recent_training_results = training_results
    if print_level > 0
        SDDP.log_iteration(options; force_if_needed = true)
        SDDP.print_helper(SDDP.print_footer, log_file_handle, training_results)
        if print_level > 1
            # SDDP.print_helper(TimerOutputs.print_timer, log_file_handle, model.timer_output)
            # Annoyingly, TimerOutputs doesn't end the print section with `\n`, so we do it here.
            TimerOutputs.print_timer(log_file_handle, model.timer_output, allocations=true)
            TimerOutputs.print_timer(stdout, model.timer_output, allocations=true)

            SDDP.print_helper(println, log_file_handle)
        end
    end
    close(log_file_handle)
    return
end


"""
    SDDP.train(model::PolicyGraph; kwargs...)

Train the policy for `model`.

## Keyword arguments: see SDDP.jl

"""
function train_loglinear(
    model::SDDP.PolicyGraph,
    algo_params::LogLinearSDDP.AlgoParams,
    problem_params::LogLinearSDDP.ProblemParams,
    applied_solver::LogLinearSDDP.AppliedSolver,
    ar_process::LogLinearSDDP.AutoregressiveProcess,
)

    # Reset the TimerOutput.
    TimerOutputs.reset_timer!(model.timer_output)
    #TODO: TimerOutputs.reset_timer!(SDDP_TIMER)
    
    # Store autoregressive_data, problem_params and algo_params in model.ext
    model.ext[:algo_params] = algo_params
    model.ext[:problem_params] = problem_params
    model.ext[:ar_process] = ar_process
    model.ext[:applied_solver] = applied_solver

    # Compute and store cut exponents
    TimerOutputs.@timeit model.timer_output "compute_cut_exponents" begin
        model.ext[:cut_exponents] = compute_cut_exponents(problem_params, ar_process)    
    end

    # Initialize the process state
    TimerOutputs.@timeit model.timer_output "initialize_process" begin
        initialize_process_state(model, ar_process)
    end

    # Reset Bellman functions
    reset_bellman_function(model)

    # Set solvers
    set_solver_for_model(model, algo_params, applied_solver)

    # Use algo_params to define parameters for train() function that is based on SDDP.train()
    train(
        model,
        print_level = algo_params.print_level,
        log_file = algo_params.log_file,
        log_frequency = algo_params.log_frequency,
        run_numerical_stability_report = algo_params.run_numerical_stability_report,
        stopping_rules = algo_params.stopping_rules,
        cut_type = SDDP.SINGLE_CUT,
        refine_at_similar_nodes = false, #TODO
        cut_deletion_minimum = 0,
        backward_sampling_scheme = CompleteSampler(),
        duality_handler = ContinuousConicDuality(),
    )

end