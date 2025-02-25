# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Copyright (c) 2023 Christian Fuellner <christian.fuellner@kit.edu>

# Note that this code reuses functions from SDDP.jl by Oscar Dowson,
# which are licensed under the Mozilla Public License, Version 2.0 as well. 
# Copyright (c) 2017-2023: Oscar Dowson and SDDP.jl contributors.
################################################################################

function _add_cut(
    V::LogLinearSDDP.ConvexApproximation,
    θᵏ::Float64, #actually not required
    πᵏ::Dict{Symbol,Float64},
    xᵏ::Dict{Symbol,Float64},
    αᵏ::Array{Float64,2},
    uᵏ::Float64,
    iteration::Int64,
    obj_y::Union{Nothing,NTuple{N,Float64}},
    belief_y::Union{Nothing,Dict{T,Float64}};
    cut_selection::Bool = false,
) where {N,T}

    model = SDDP.get_policy_graph(JuMP.owner_model(V.theta))

    for (key, x) in xᵏ
        θᵏ -= πᵏ[key] * x
    end
    SDDP._dynamic_range_warning(θᵏ, πᵏ)
    TimerOutputs.@timeit model.timer_output "construct_cut_object" begin
       cut = LogLinearSDDP.Cut(πᵏ, αᵏ, uᵏ, θᵏ - uᵏ, xᵏ, nothing, nothing, obj_y, belief_y, 1, iteration)
    end
    model.ext[:total_cuts] += 1

    TimerOutputs.@timeit model.timer_output "add_cut_to_model" begin
        _add_cut_constraint_to_model(V, cut)
    end
    model.ext[:active_cuts] += 1

    if cut_selection
        TimerOutputs.@timeit model.timer_output "cut_selection" begin
            _cut_selection_update(V, cut, xᵏ)
        end
    else
        push!(V.cuts, cut)
    end
    return
end


function _add_cut_constraint_to_model(V::LogLinearSDDP.ConvexApproximation, cut::LogLinearSDDP.Cut)
    model = JuMP.owner_model(V.theta)
    yᵀμ = JuMP.AffExpr(0.0)
    if V.objective_states !== nothing
        for (y, μ) in zip(cut.obj_y, V.objective_states)
            JuMP.add_to_expression!(yᵀμ, y, μ)
        end
    end
    if V.belief_states !== nothing
        for (k, μ) in V.belief_states
            JuMP.add_to_expression!(yᵀμ, cut.belief_y[k], μ)
        end
    end
    expr = JuMP.@expression(
        model,
        V.theta + yᵀμ - sum(cut.coefficients[i] * x for (i, x) in V.states)
    )

     # CHANGES TO SDDP.jl
    ####################################################################################
    cut.cut_intercept_variable = JuMP.@variable(model)
    JuMP.fix(cut.cut_intercept_variable, 0)
    # this variable will be fixed by function evaluate_cut_intercept to a meaningful value
    # before any subproblem is solved

    cut.constraint_ref = if JuMP.objective_sense(model) == MOI.MIN_SENSE
        JuMP.@constraint(model, expr >= cut.cut_intercept_variable)
    else
        JuMP.@constraint(model, expr <= cut.cut_intercept_variable)
    end
    
    node = SDDP.get_node(model)
    push!(node.ext[:cut_cons], cut.constraint_ref)
    ####################################################################################

    return
end

function _add_cut_constraint_to_model2(V::LogLinearSDDP.ConvexApproximation, cut::LogLinearSDDP.Cut)
    model = JuMP.owner_model(V.theta)
    yᵀμ = JuMP.AffExpr(0.0)
    if V.objective_states !== nothing
        for (y, μ) in zip(cut.obj_y, V.objective_states)
            JuMP.add_to_expression!(yᵀμ, y, μ)
        end
    end
    if V.belief_states !== nothing
        for (k, μ) in V.belief_states
            JuMP.add_to_expression!(yᵀμ, cut.belief_y[k], μ)
        end
    end
    expr = JuMP.@expression(
        model,
        V.theta + yᵀμ - sum(cut.coefficients[i] * x for (i, x) in V.states)
    )

     # CHANGES TO SDDP.jl
    ####################################################################################
    cut.constraint_ref = if JuMP.objective_sense(model) == MOI.MIN_SENSE
        JuMP.@constraint(model, expr >= 0)
    else
        JuMP.@constraint(model, expr <= 0)
    end
    
    node = SDDP.get_node(model)
    push!(node.ext[:cut_cons], cut.constraint_ref)
    ####################################################################################

    return
end

function refine_bellman_function(
    model::SDDP.PolicyGraph{T},
    node::SDDP.Node{T},
    bellman_function::LogLinearSDDP.BellmanFunction,
    risk_measure::SDDP.AbstractRiskMeasure,
    outgoing_state::Dict{Symbol,Float64},
    dual_variables::Vector{Dict{Symbol,Float64}},
    noise_supports::Vector,
    nominal_probability::Vector{Float64},
    objective_realizations::Vector{Float64}, #actually not required
    intercept_factors::Vector{Array{Float64,2}},
    stochastic_intercepts_tight::Vector{Float64},
) where {T}
    # Sanity checks.
    @assert length(dual_variables) ==
            length(noise_supports) ==
            length(nominal_probability) ==
            length(objective_realizations) ==
            length(intercept_factors) ==
            length(stochastic_intercepts_tight)

    # Preliminaries that are common to all cut types.
    risk_adjusted_probability = similar(nominal_probability)
    offset = SDDP.adjust_probability(
        risk_measure,
        risk_adjusted_probability,
        nominal_probability,
        noise_supports,
        objective_realizations,
        model.objective_sense == MOI.MIN_SENSE,
    )
    # The meat of the function.
    if bellman_function.cut_type == SDDP.SINGLE_CUT
        return _add_average_cut(
            node,
            outgoing_state,
            risk_adjusted_probability,
            objective_realizations, #actually not required
            dual_variables,
            offset,
            intercept_factors,
            stochastic_intercepts_tight,
        )
    else  # Add a multi-cut
        @assert bellman_function.cut_type == SDDP.MULTI_CUT
        @error("Multi-cut approach is not supported yet.")
    end
end


function _add_average_cut(
    node::SDDP.Node,
    outgoing_state::Dict{Symbol,Float64},
    risk_adjusted_probability::Vector{Float64},
    objective_realizations::Vector{Float64}, #actually not required
    dual_variables::Vector{Dict{Symbol,Float64}},
    offset::Float64,
    intercept_factors::Vector{Array{Float64,2}},
    stochastic_intercepts_tight::Vector{Float64},
)
    N = length(risk_adjusted_probability)
    @assert N == length(objective_realizations) == length(dual_variables) == length(intercept_factors)

    # CHANGES TO SDDP.jl
    ####################################################################################
    model = SDDP.get_policy_graph(node.subproblem)
    t = node.index+1
    T = model.ext[:problem_params].number_of_stages
    L = model.ext[:ar_process].parameters[t].dimension
    αᵏ = zeros(T-t+1, L)    
    
    # Calculate the expected intercept and dual variables with respect to the
    # risk-adjusted probability distribution.
    
    πᵏ = Dict(key => 0.0 for key in keys(outgoing_state))
    θᵏ = offset
    uᵏ = 0.0

    for i in 1:length(objective_realizations)
        p = risk_adjusted_probability[i]
        θᵏ += p * objective_realizations[i]
        uᵏ += p * stochastic_intercepts_tight[i]
        for (key, dual) in dual_variables[i]
            πᵏ[key] += p * dual
        end
    end

    TimerOutputs.@timeit model.timer_output "aggregate_cut_info" begin
        aggregate_alpha4!(αᵏ, model, intercept_factors, t, T, risk_adjusted_probability)
    end
 
    # aggregate_alpha!(αᵏ, model, intercept_factors, t, T, risk_adjusted_probability) 
    # println(αᵏ)
    # αᵏ = zeros(T-t+1, L)    
    # aggregate_alpha2!(αᵏ, model, intercept_factors, t, T, risk_adjusted_probability) 
    # println(αᵏ)
    # αᵏ = zeros(T-t+1, L)    
    # aggregate_alpha3!(αᵏ, model, intercept_factors, t, T, risk_adjusted_probability) 
    # println(αᵏ)
    # αᵏ = zeros(T-t+1, L)    
    # aggregate_alpha4!(αᵏ, model, intercept_factors, t, T, risk_adjusted_probability) 
    # println(αᵏ)

    #Infiltrator.@infiltrate 

    # BenchmarkTools.@btime aggregate_alpha!($αᵏ, $model, $intercept_factors, $t, $T, $risk_adjusted_probability) 
    # αᵏ = zeros(T-t+1, L)    
    # BenchmarkTools.@btime aggregate_alpha2!($αᵏ, $model, $intercept_factors, $t, $T, $risk_adjusted_probability) 
    # αᵏ = zeros(T-t+1, L)    
    # BenchmarkTools.@btime aggregate_alpha3!($αᵏ, $model, $intercept_factors, $t, $T, $risk_adjusted_probability) 
    # αᵏ = zeros(T-t+1, L)    
    # BenchmarkTools.@btime aggregate_alpha4!($αᵏ, $model, $intercept_factors, $t, $T, $risk_adjusted_probability) 
    # Infiltrator.@infiltrate

    ####################################################################################

    # Now add the average-cut to the subproblem. We include the objective-state
    # component μᵀy and the belief state (if it exists).
    obj_y =
        node.objective_state === nothing ? nothing : node.objective_state.state
    belief_y =
        node.belief_state === nothing ? nothing : node.belief_state.belie
    _add_cut(
        node.bellman_function.global_theta,
        θᵏ,
        πᵏ,
        outgoing_state,
        αᵏ,
        uᵏ,
        model.ext[:iteration],
        obj_y,
        belief_y;
        model.ext[:algo_params].cut_selection,
    )
    return (
        theta = θᵏ,
        pi = πᵏ,
        x = outgoing_state,
        obj_y = obj_y,
        belief_y = belief_y,
        αᵏ = αᵏ,
    )
end

"""
Internal function: calculate the height of `cut` evaluated at `state`.
"""
function _eval_height(cut::LogLinearSDDP.Cut, sampled_state::LogLinearSDDP.SampledState)
    height = cut.deterministic_intercept + cut.stochastic_intercept_tight
    for (key, value) in cut.coefficients
        height += value * sampled_state.state[key]
    end
    return height
end

"""
Internal function: check if the candidate point dominates the incumbent.
"""
function _dominates(candidate, incumbent, minimization::Bool)
    return minimization ? candidate >= incumbent : candidate <= incumbent
end

function _cut_selection_update(
    V::LogLinearSDDP.ConvexApproximation,
    cut::LogLinearSDDP.Cut,
    state::Dict{Symbol,Float64},
)
    model = JuMP.owner_model(V.theta)
    policy_graph = SDDP.get_policy_graph(model)
    is_minimization = JuMP.objective_sense(model) == MOI.MIN_SENSE
    sampled_state = LogLinearSDDP.SampledState(state, cut.obj_y, cut.belief_y, cut, NaN)
    sampled_state.best_objective = _eval_height(cut, sampled_state)
    # Loop through previously sampled states and compare the height of the most
    # recent cut against the current best. If this new cut is an improvement,
    # store this one instead.
    for old_state in V.sampled_states
        # Only compute cut selection at same points in concave space.
        if old_state.obj_y != cut.obj_y || old_state.belief_y != cut.belief_y
            continue
        end
        height = _eval_height(cut, old_state)
        if _dominates(height, old_state.best_objective, is_minimization)
            old_state.dominating_cut.non_dominated_count -= 1
            cut.non_dominated_count += 1
            old_state.dominating_cut = cut
            old_state.best_objective = height
        end
    end
    push!(V.sampled_states, sampled_state)
    # Now loop through previously discovered cuts and compare their height at
    # `sampled_state`. If a cut is an improvement, add it to a queue to be
    # added.
    for old_cut in V.cuts
        if old_cut.constraint_ref !== nothing
            # We only care about cuts not currently in the model.
            continue
        elseif old_cut.obj_y != sampled_state.obj_y
            # Only compute cut selection at same points in objective space.
            continue
        elseif old_cut.belief_y != sampled_state.belief_y
            # Only compute cut selection at same points in belief space.
            continue
        end
        height = _eval_height(old_cut, sampled_state)
        if _dominates(height, sampled_state.best_objective, is_minimization)
            sampled_state.dominating_cut.non_dominated_count -= 1
            old_cut.non_dominated_count += 1
            sampled_state.dominating_cut = old_cut
            sampled_state.best_objective = height
            _add_cut_constraint_to_model(V, old_cut)
            policy_graph.ext[:active_cuts] += 1
        end
    end
    push!(V.cuts, cut)
    # Delete cuts that need to be deleted.
    for cut in V.cuts
        if cut.non_dominated_count < 1
            if cut.constraint_ref !== nothing
                push!(V.cuts_to_be_deleted, cut)
            end
        end
    end

    if length(V.cuts_to_be_deleted) >= V.deletion_minimum
        for cut in V.cuts_to_be_deleted
            JuMP.delete(model, cut.constraint_ref)
            cut.constraint_ref = nothing
            cut.non_dominated_count = 0
            policy_graph.ext[:active_cuts] -= 1
        end
    end
    empty!(V.cuts_to_be_deleted)
    return
end

function aggregate_alpha!(
    αᵏ::Array{Float64,2},
    model::SDDP.PolicyGraph,
    intercept_factors::Vector{Array{Float64,2}},
    t::Int,
    T::Int,
    risk_adjusted_probability::Vector{Float64},
)
    for i in 1:length(risk_adjusted_probability)
        p = risk_adjusted_probability[i]
        for τ in t:T
            L_τ = model.ext[:ar_process].parameters[τ].dimension
            for ℓ in 1:L_τ
                αᵏ[τ-t+1,ℓ] += p * intercept_factors[i][τ-t+1,ℓ]
            end
        end
    end

    return
end

function aggregate_alpha2!(
    αᵏ::Array{Float64,2},
    model::SDDP.PolicyGraph,
    intercept_factors::Vector{Array{Float64,2}},
    t::Int,
    T::Int,
    risk_adjusted_probability::Vector{Float64},
)
    for τ in t:T
        L_τ = model.ext[:ar_process].parameters[τ].dimension
        for ℓ in 1:L_τ
            for i in 1:length(risk_adjusted_probability)
                p = risk_adjusted_probability[i]
                αᵏ[τ-t+1,ℓ] += p * intercept_factors[i][τ-t+1,ℓ]
            end
        end
    end
    
    return
end

function aggregate_alpha3!(
    αᵏ::Array{Float64,2},
    model::SDDP.PolicyGraph,
    intercept_factors::Vector{Array{Float64,2}},
    t::Int,
    T::Int,
    risk_adjusted_probability::Vector{Float64},
)
    for τ in t:T
        L_τ = model.ext[:ar_process].parameters[τ].dimension
        for ℓ in 1:L_τ
            αᵏ[τ-t+1,ℓ] = sum(risk_adjusted_probability[i] * intercept_factors[i][τ-t+1,ℓ] for i in eachindex(risk_adjusted_probability))
        end
    end

    return
end

function aggregate_alpha4!(
    αᵏ::Array{Float64,2},
    model::SDDP.PolicyGraph,
    intercept_factors::Vector{Array{Float64,2}},
    t::Int,
    T::Int,
    risk_adjusted_probability::Vector{Float64},
)
    Tullio.@tullio αᵏ[τ-t+1,ℓ] = sum(risk_adjusted_probability[i] * intercept_factors[i][τ-t+1,ℓ] for i in eachindex(risk_adjusted_probability)) (τ in t:T, ℓ in 1:4)

    return
end