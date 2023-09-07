mutable struct ConvexApproximation
    theta::JuMP.VariableRef
    states::Dict{Symbol,JuMP.VariableRef}
    objective_states::Union{Nothing,NTuple{N,JuMP.VariableRef} where {N}}
    belief_states::Union{Nothing,Dict{T,JuMP.VariableRef} where {T}}
    # Storage for cut selection
    cuts::Vector{LogLinearSDDP.Cut}
    sampled_states::Vector{SDDP.SampledState}
    cuts_to_be_deleted::Vector{LogLinearSDDP.Cut}
    deletion_minimum::Int

    function ConvexApproximation(
        theta::JuMP.VariableRef,
        states::Dict{Symbol,JuMP.VariableRef},
        objective_states,
        belief_states,
        deletion_minimum::Int,
    )
        return new(
            theta,
            states,
            objective_states,
            belief_states,
            LogLinearSDDP.Cut[],
            SDDP.SampledState[],
            LogLinearSDDP.Cut[],
            deletion_minimum,
        )
    end
end


function _add_cut(
    V::LogLinearSDDP.ConvexApproximation,
    θᵏ::Float64, #actually not required
    πᵏ::Dict{Symbol,Float64},
    xᵏ::Dict{Symbol,Float64},
    αᵏ::Array{Float64,2},
    iteration::Int64,
    obj_y::Union{Nothing,NTuple{N,Float64}},
    belief_y::Union{Nothing,Dict{T,Float64}};
    cut_selection::Bool = false,
) where {N,T}
    for (key, x) in xᵏ
        θᵏ -= πᵏ[key] * x
    end
    SDDP._dynamic_range_warning(θᵏ, πᵏ)
    cut = LogLinearSDDP.Cut(πᵏ, θᵏ, αᵏ, xᵏ, nothing, nothing, obj_y, belief_y, 1, iteration)
    _add_cut_constraint_to_model(V, cut)
    if cut_selection
        @error("Cut selection is not supported yet.")
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
    ####################################################################################

    return
end


function refine_bellman_function(
    model::SDDP.PolicyGraph{T},
    node::SDDP.Node{T},
    bellman_function::SDDP.BellmanFunction,
    risk_measure::SDDP.AbstractRiskMeasure,
    outgoing_state::Dict{Symbol,Float64},
    dual_variables::Vector{Dict{Symbol,Float64}},
    noise_supports::Vector,
    nominal_probability::Vector{Float64},
    objective_realizations::Vector{Float64}, #actually not required
    intercept_factors::Vector{Array{Float64,2}},
) where {T}
    # Sanity checks.
    @assert length(dual_variables) ==
            length(noise_supports) ==
            length(nominal_probability) ==
            length(objective_realizations) ==
            length(intercept_factors)

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
    if bellman_function.cut_type == SINGLE_CUT
        return _add_average_cut(
            node,
            outgoing_state,
            risk_adjusted_probability,
            objective_realizations, #actually not required
            dual_variables,
            offset,
            intercept_factors,
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
)
    N = length(risk_adjusted_probability)
    @assert N == length(objective_realizations) == length(dual_variables) == length(intercept_factors)

    # CHANGES TO SDDP.jl
    ####################################################################################
    model = SDDP.get_policy_graph(node.subproblem)
    t = node.node_index+1
    T = model.ext[:problem_params].number_of_stages
    L = model.ext[:autoregressive_data].ar_data[t].ar_dimension
    αᵏ = Array{Float64,2}(undef, T-t+1, L)    
    
    # Calculate the expected intercept and dual variables with respect to the
    # risk-adjusted probability distribution.
    πᵏ = Dict(key => 0.0 for key in keys(outgoing_state))
    θᵏ = offset

    for i in 1:length(objective_realizations)
        p = risk_adjusted_probability[i]
        θᵏ += p * objective_realizations[i]
        for (key, dual) in dual_variables[i]
            πᵏ[key] += p * dual
        end

        for τ in t:T
            L_τ = model.ext[:autoregressive_data].ar_data[τ].ar_dimension
            for ℓ in 1:L_τ
                αᵏ[τ-t+1,ℓ] += p * intercept_factors[i][τ-t+1,ℓ]
            end
        end
    end

    # Note that θᵏ is actually not required for the modified cut formulas. However, we compute it for 
    # possible checks.
    ####################################################################################

    # Now add the average-cut to the subproblem. We include the objective-state
    # component μᵀy and the belief state (if it exists).
    obj_y =
        node.objective_state === nothing ? nothing : node.objective_state.state
    belief_y =
        node.belief_state === nothing ? nothing : node.belief_state.belief
    _add_cut(
        node.bellman_function.global_theta,
        θᵏ,
        πᵏ,
        outgoing_state,
        αᵏ,
        model.ext[:iteration],
        obj_y,
        belief_y,
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