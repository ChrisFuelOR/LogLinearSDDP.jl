# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Copyright (c) 2023 Christian Fuellner <christian.fuellner@kit.edu>

# Note that this code reuses functions from SDDP.jl by Oscar Dowson,
# which are licensed under the Mozilla Public License, Version 2.0 as well. 
# Copyright (c) 2017-2023: Oscar Dowson and SDDP.jl contributors.
################################################################################

struct ContinuousConicDuality <: SDDP.AbstractDualityHandler end

function get_dual_solution(node::SDDP.Node, Nothing)
    return JuMP.objective_value(node.subproblem), Dict{Symbol,Float64}(), nothing, -Inf
end


function prepare_backward_pass(
    model::SDDP.PolicyGraph,
    duality_handler::SDDP.AbstractDualityHandler,
    options::LogLinearSDDP.Options,
)
    undo = Function[]
    for (_, node) in model.nodes
        push!(undo, prepare_backward_pass(node, duality_handler, options))
    end
    function undo_relax()
        for f in undo
            f()
        end
        return
    end
    return undo_relax
end


function get_dual_solution(node::SDDP.Node, ::ContinuousConicDuality)
    if JuMP.dual_status(node.subproblem) != JuMP.MOI.FEASIBLE_POINT
        # Attempt to recover by resetting the optimizer and re-solving.
        if JuMP.mode(node.subproblem) != JuMP.DIRECT
            MOI.Utilities.reset_optimizer(node.subproblem)
            JuMP.optimize!(node.subproblem)
        end
    end
    if JuMP.dual_status(node.subproblem) != JuMP.MOI.FEASIBLE_POINT
        SDDP.write_subproblem_to_file(
            node,
            "subproblem.mof.json",
            throw_error = true,
        )
    end
    # Note: due to JuMP's dual convention, we need to flip the sign for
    # maximization problems.
    dual_sign = JuMP.objective_sense(node.subproblem) == MOI.MIN_SENSE ? 1 : -1

    # CHANGES TO SDDP.jl
    ####################################################################################
    # Here, we need optimal solutions for much more different dual variables, as this
    # is required for the more complicated cut formulas.

    # The multiplier for the copy constraint can be used to define the cut gradient later on.
    λ = Dict{Symbol,Float64}(
        name => dual_sign * JuMP.dual(JuMP.FixRef(state.in)) for
        (name, state) in node.states
    )

    model = SDDP.get_policy_graph(node.subproblem)
    TimerOutputs.@timeit model.timer_output "compute_alphas" begin
        α = get_alphas(node)
    end

    # Evaluate the "stochastic" part of the intercept for the current noise in the backward pass
    TimerOutputs.@timeit model.timer_output "cut_intercept_tight" begin
        stochastic_intercept_tight = evaluate_cut_intercept_tight(node, α)
    end

    return JuMP.objective_value(node.subproblem), λ, α, stochastic_intercept_tight
end


function _relax_integrality(node::SDDP.Node)
    if !node.has_integrality
        return () -> nothing
    end
    return JuMP.relax_integrality(node.subproblem)
end


function prepare_backward_pass(node::SDDP.Node, ::ContinuousConicDuality, ::LogLinearSDDP.Options)
    return _relax_integrality(node)
end


function get_existing_cuts_factors1(cuts::Vector{LogLinearSDDP.Cut}, model::JuMP.Model)

    cut_array = Vector{Array{Float64,2}}(undef, length(cuts))

    for cut_index in eachindex(cuts)
        # Get optimal dual value of cut constraint and alpha value for given cut to update the factor
        cut_array[cut_index] = JuMP.dual(cuts[cut_index].constraint_ref) * cuts[cut_index].intercept_factors
        #println(JuMP.dual(cuts[cut_index].constraint_ref))
    end
    #println()

    return sum(cut_array)
end

function get_existing_cuts_factors7(cuts::Vector{LogLinearSDDP.Cut}, node::SDDP.Node)

    dual_solution = zeros(length(cuts))
    Gurobi.GRBgetdblattrarray(JuMP.backend(node.subproblem), "Pi", 9, length(dual_solution), dual_solution)
    non_zero_dual_indices = findall(>(0), dual_solution)
    cut_array = Vector{Array{Float64,2}}()   
    
    if length(non_zero_dual_indices) > 0
        for cut_index in non_zero_dual_indices
            # Get optimal dual value of cut constraint and alpha value for given cut to update the factor
            push!(cut_array, dual_solution[cut_index] * cuts[cut_index].intercept_factors)
        end
        return sum(cut_array)

    else
        return zeros(size(cuts[1].intercept_factors))
    end
end


function compute_alpha_t!(α::Array{Float64,2}, ar_process_stage::LogLinearSDDP.AutoregressiveProcessStage, current_independent_noise_term::Any, coupling_constraints::Vector{JuMP.ConstraintRef}, L::Int64)

    for ℓ in 1:L
        μ = JuMP.dual(coupling_constraints[ℓ])
        α[1,ℓ] = μ * exp(ar_process_stage.intercept[ℓ]) * exp(current_independent_noise_term[ℓ] * ar_process_stage.psi[ℓ])
    end
end

function compute_alpha_t6!(α::Array{Float64,2}, node::SDDP.Node, ar_process_stage::LogLinearSDDP.AutoregressiveProcessStage, current_independent_noise_term::Vector{Float64}, coupling_constraints::Vector{JuMP.ConstraintRef}, L::Int64)

    indices = JuMP.index.(coupling_constraints)
    dual_solution = zeros(length(indices))
    Gurobi.GRBgetdblattrarray(JuMP.backend(node.subproblem), "Pi", 0, length(dual_solution), dual_solution)

    for ℓ in 1:L
        α[1,ℓ] = dual_solution[ℓ] * exp(ar_process_stage.intercept[ℓ] + current_independent_noise_term[ℓ] * ar_process_stage.psi[ℓ])
    end
end

   
function compute_alpha_tau3!(α::Array{Float64,2}, cut_factors::Array{Float64,2}, cut_exponents::Any, ar_process_stage::LogLinearSDDP.AutoregressiveProcessStage, ar_parameters::Any, current_independent_noise_term::Any, t::Int64, T::Int64, L_t::Int64)

    for τ in t+1:T 
        L_τ = ar_parameters[τ].dimension
        for ℓ in 1:L_τ
            sum_val = 0.0
            for ν in 1:L_t
                sum_val += cut_exponents[τ,ℓ,ν,1] * (ar_process_stage.intercept[ν] + current_independent_noise_term[ν] * ar_process_stage.psi[ν])
            end
            α[τ-t+1,ℓ] = cut_factors[τ-t,ℓ] * exp(sum_val)
        end
    end
end

function compute_alpha_tau13!(α::Array{Float64,2}, cut_factors::Array{Float64,2}, cut_exponents::Array{Float64,4}, intercept::Vector{Float64}, psi::Vector{Float64}, current_independent_noise_term::Any, t::Int64, T::Int64, L_t::Int64)

    @turbo for ℓ in 1:4
        for τ in t+1:T 
            α[τ-t+1,ℓ] = cut_factors[τ-t,ℓ] * exp(cut_exponents[τ,ℓ,ℓ,1] * (intercept[ℓ] + current_independent_noise_term[ℓ] * psi[ℓ]))
        end
    end
end


function get_alphas(node::SDDP.Node)

    # We also need the dual variables for all coupling constraints.
    # In order to identify the coupling constraints, we should specify them in the problem definition.
    # Moreover, we need the dual variables for all existing cut constraints.
    model = SDDP.get_policy_graph(node.subproblem)
    t = node.index
    T = model.ext[:problem_params].number_of_stages
    ar_process = model.ext[:ar_process]
    ar_process_stage = ar_process.parameters[t]    
    L = get_max_dimension(ar_process)
    L_t = ar_process_stage.dimension
    α = Array{Float64,2}(undef, T-t+1, L)

    current_independent_noise_term = node.ext[:current_independent_noise_term]

    # Case τ = t 
    TimerOutputs.@timeit model.timer_output "alpha_t_new" begin
        compute_alpha_t!(α, ar_process_stage, current_independent_noise_term, node.subproblem.ext[:coupling_constraints], L_t)
    end   

    # Get cut constraint duals and compute first factor
    if t < T
       TimerOutputs.@timeit model.timer_output "existing_cut_factor_7" begin
           cut_factors = get_existing_cuts_factors7(node.bellman_function.global_theta.cuts, node)
        end
        
        # Case τ > t
        TimerOutputs.@timeit model.timer_output "alpha_tau_new" begin
            compute_alpha_tau13!(α, cut_factors, model.ext[:cut_exponents][t+1],  ar_process_stage.intercept, ar_process_stage.psi, current_independent_noise_term, t, T, L_t)
        end
   end

    return α

end

duality_log_key(::ContinuousConicDuality) = " "

