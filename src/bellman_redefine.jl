# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Copyright (c) 2025 Christian Fuellner <christian.fuellner@kit.edu>

# Note that this code reuses functions from SDDP.jl by Oscar Dowson,
# which are licensed under the Mozilla Public License, Version 2.0 as well. 
# Copyright (c) 2017-2025: Oscar Dowson and SDDP.jl contributors.
################################################################################

mutable struct ConvexApproximation
    theta::JuMP.VariableRef
    states::Dict{Symbol,JuMP.VariableRef}
    objective_states::Union{Nothing,NTuple{N,JuMP.VariableRef} where {N}}
    belief_states::Union{Nothing,Dict{T,JuMP.VariableRef} where {T}}
    # Storage for cut selection
    cuts::Vector{LogLinearSDDP.Cut}
    sampled_states::Vector{SDDP.SampledState}
    cuts_to_be_deleted::Vector{LogLinearSDDP.Cut}
    deletion_minimum::Int64

    function ConvexApproximation(
        theta::JuMP.VariableRef,
        states::Dict{Symbol,JuMP.VariableRef},
        objective_states,
        belief_states,
        deletion_minimum::Int64,
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


"""
    BellmanFunction

A representation of the value function. SDDP.jl uses the following unique
representation of the value function that is undocumented in the literature.
For more details, see SDDP.jl documentation.
"""
mutable struct BellmanFunction
    cut_type::SDDP.CutType
    global_theta::LogLinearSDDP.ConvexApproximation
    local_thetas::Vector{ LogLinearSDDP.ConvexApproximation}
    # Cuts defining the dual representation of the risk measure.
    risk_set_cuts::Set{Vector{Float64}}
end

"""
    BellmanFunction(;
        lower_bound = -Inf,
        upper_bound = Inf,
        deletion_minimum::Int64 = 1,
        cut_type::CutType = SDDP.MULTI_CUT,
    )
"""
function BellmanFunction(;
    lower_bound = -Inf,
    upper_bound = Inf,
    deletion_minimum::Int64 = 1,
    cut_type::SDDP.CutType = SDDP.MULTI_CUT,
)
    return SDDP.InstanceFactory{LogLinearSDDP.BellmanFunction}(
        lower_bound = lower_bound,
        upper_bound = upper_bound,
        deletion_minimum = deletion_minimum,
        cut_type = cut_type,
    )
end

function bellman_term(bellman_function::LogLinearSDDP.BellmanFunction)
    return bellman_function.global_theta.theta
end

function initialize_bellman_function(
    factory::SDDP.InstanceFactory{LogLinearSDDP.BellmanFunction},
    model::SDDP.PolicyGraph{T},
    node::SDDP.Node{T},
) where {T}
    lower_bound, upper_bound, deletion_minimum, cut_type =
        -Inf, Inf, 0, SDDP.SINGLE_CUT
    if length(factory.args) > 0
        error(
            "Positional arguments $(factory.args) ignored in BellmanFunction.",
        )
    end
    for (kw, value) in factory.kwargs
        if kw == :lower_bound
            lower_bound = value
        elseif kw == :upper_bound
            upper_bound = value
        elseif kw == :deletion_minimum
            deletion_minimum = value
        elseif kw == :cut_type
            cut_type = value
        else
            error(
                "Keyword $(kw) not recognised as argument to BellmanFunction.",
            )
        end
    end
    if lower_bound == -Inf && upper_bound == Inf
        error("You must specify a finite bound on the cost-to-go term.")
    end
    if length(node.children) == 0
        lower_bound = upper_bound = 0.0
    end
    Θᴳ = JuMP.@variable(node.subproblem)
    lower_bound > -Inf && JuMP.set_lower_bound(Θᴳ, lower_bound)
    upper_bound < Inf && JuMP.set_upper_bound(Θᴳ, upper_bound)
    # Initialize bounds for the objective states. If objective_state==nothing,
    # this check will be skipped by dispatch.
    SDDP._add_initial_bounds(node.objective_state, Θᴳ)
    x′ = Dict(key => var.out for (key, var) in node.states)
    obj_μ = node.objective_state !== nothing ? node.objective_state.μ : nothing
    belief_μ = node.belief_state !== nothing ? node.belief_state.μ : nothing
    return LogLinearSDDP.BellmanFunction(
        cut_type,
        LogLinearSDDP.ConvexApproximation(Θᴳ, x′, obj_μ, belief_μ, deletion_minimum),
        LogLinearSDDP.ConvexApproximation[],
        Set{Vector{Float64}}(),
    )
end

"""
Replace the bellman_functions created by SDDP.BellmanFunction with LogLinearSDDP.BellmanFunction
to be able to store our slightly altered cut objects.
"""
function reset_bellman_function(
    model::SDDP.PolicyGraph,
    )

    for (_, node) in model.nodes
        # Get existing bounds
        θ = SDDP.bellman_term(node.bellman_function)
        lower_bound = JuMP.has_lower_bound(θ) ? JuMP.lower_bound(θ) : -Inf
        upper_bound = JuMP.has_upper_bound(θ) ? JuMP.upper_bound(θ) : Inf

        # Get parameters of current Bellman function
        cut_type = node.bellman_function.cut_type
        cut_deletion_minimum = node.bellman_function.global_theta.deletion_minimum 
        
        # Create a new bellman function
        bellman_function = LogLinearSDDP.BellmanFunction(
                lower_bound = lower_bound,
                upper_bound = upper_bound,
        )

        # Initialize the bellman function
        node.bellman_function = initialize_bellman_function(bellman_function, model, node)

        # Initialize parameters for new bellman function
        node.bellman_function.cut_type = cut_type
        node.bellman_function.global_theta.deletion_minimum =
            cut_deletion_minimum
        for oracle in node.bellman_function.local_thetas
            oracle.deletion_minimum = cut_deletion_minimum
        end

    end
end


