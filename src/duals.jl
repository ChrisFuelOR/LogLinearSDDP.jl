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

function get_existing_cuts_factors3(cuts::Vector{LogLinearSDDP.Cut}, model::JuMP.Model)

    cut_array = Vector{Array{Float64,2}}(undef, length(cuts))

    for cut_index in eachindex(cuts)
        # Get optimal dual value of cut constraint and alpha value for given cut to update the factor
        cut_array[cut_index] = MOI.get(JuMP.backend(model), MOI.ConstraintDual(), JuMP.index(cuts[cut_index].constraint_ref)) * cuts[cut_index].intercept_factors
    end

    return sum(cut_array)
end

function get_existing_cuts_factors4(cuts::Vector{LogLinearSDDP.Cut}, node::SDDP.Node)

    indices = JuMP.index.(node.ext[:cut_cons])
    dual_solution = zeros(length(indices))
    dual_solution .= MOI.get(JuMP.backend(node.subproblem), MOI.ConstraintDual(), indices)
    cut_array = Vector{Array{Float64,2}}(undef, length(cuts))

    for cut_index in eachindex(cuts)
        # Get optimal dual value of cut constraint and alpha value for given cut to update the factor
        cut_array[cut_index] = dual_solution[cut_index] * cuts[cut_index].intercept_factors
    end

    return sum(cut_array)
end

function get_existing_cuts_factors5(cuts::Vector{LogLinearSDDP.Cut}, node::SDDP.Node)

    dual_solution = zeros(length(cuts))
    Gurobi.GRBgetdblattrarray(JuMP.backend(node.subproblem), "Pi", 9, length(dual_solution), dual_solution)
    cut_array = Vector{Array{Float64,2}}(undef, length(cuts))   

    for cut_index in eachindex(cuts)
        # Get optimal dual value of cut constraint and alpha value for given cut to update the factor
        cut_array[cut_index] = dual_solution[cut_index] * cuts[cut_index].intercept_factors
    end
    
    return sum(cut_array)
end

function get_existing_cuts_factors6(cuts::Vector{LogLinearSDDP.Cut}, node::SDDP.Node)
    
    cut_array = Vector{Array{Float64,2}}(undef, length(cuts))   

    for cut_index in eachindex(cuts)
        # Get optimal dual value of cut constraint and alpha value for given cut to update the factor
        dual_value = Ref{Cdouble}()
        cut_array[cut_index] = Gurobi.GRBgetdblattrelement(JuMP.backend(node.subproblem), "Pi", 9+cut_index, dual_value) * cuts[cut_index].intercept_factors
    end

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

function get_existing_cuts_factors2(cuts::Vector{LogLinearSDDP.Cut})

    cut_array = Vector{Array{Float64,2}}(undef, length(cuts))

    @batch minbatch=100 for cut_index in eachindex(cuts)
        # Get optimal dual value of cut constraint and alpha value for given cut to update the factor
        cut_array[cut_index] = JuMP.dual(cuts[cut_index].constraint_ref) * cuts[cut_index].intercept_factors
    end

    return sum(cut_array)
end

function compute_alpha_t!(α::Array{Float64,2}, ar_process_stage::LogLinearSDDP.AutoregressiveProcessStage, current_independent_noise_term::Any, coupling_constraints::Vector{JuMP.ConstraintRef}, L::Int64)

    for ℓ in 1:L
        μ = JuMP.dual(coupling_constraints[ℓ])
        α[1,ℓ] = μ * exp(ar_process_stage.intercept[ℓ]) * exp(current_independent_noise_term[ℓ] * ar_process_stage.psi[ℓ])
    end
end

function compute_alpha_t2!(α::Array{Float64,2}, ar_process_stage::LogLinearSDDP.AutoregressiveProcessStage, current_independent_noise_term::Any, coupling_constraints::Vector{JuMP.ConstraintRef}, L::Int64)

    expo = ar_process_stage.intercept .+ ar_process_stage.psi .* current_independent_noise_term

    for ℓ in 1:L
        α[1,ℓ] = JuMP.dual(coupling_constraints[ℓ]) * exp(expo[ℓ])
    end
end

function compute_alpha_t3!(α::Array{Float64,2}, node::SDDP.Node, ar_process_stage::LogLinearSDDP.AutoregressiveProcessStage, current_independent_noise_term::Any, coupling_constraints::Vector{JuMP.ConstraintRef}, L::Int64)

    indices = JuMP.index.(coupling_constraints)
    dual_solution = zeros(length(indices))
    dual_solution .= MOI.get(JuMP.backend(node.subproblem), MOI.ConstraintDual(), indices)

    for ℓ in 1:L
        α[1,ℓ] = dual_solution[ℓ] * exp(ar_process_stage.intercept[ℓ]) * exp(current_independent_noise_term[ℓ] * ar_process_stage.psi[ℓ])
    end
end

function compute_alpha_t4!(α::Array{Float64,2}, node::SDDP.Node, ar_process_stage::LogLinearSDDP.AutoregressiveProcessStage, current_independent_noise_term::Vector{Float64}, coupling_constraints::Vector{JuMP.ConstraintRef}, L::Int64)

    indices = JuMP.index.(coupling_constraints)
    dual_solution = zeros(length(indices))
    dual_solution .= MOI.get(JuMP.backend(node.subproblem), MOI.ConstraintDual(), indices)

    for ℓ in 1:L
        α[1,ℓ] = dual_solution[ℓ] * exp(ar_process_stage.intercept[ℓ]) * exp(current_independent_noise_term[ℓ] * ar_process_stage.psi[ℓ])
    end
end

function compute_alpha_t5!(α::Array{Float64,2}, node::SDDP.Node, intercept::Vector{Float64}, psi::Vector{Float64}, current_independent_noise_term::Vector{Float64}, coupling_constraints::Vector{JuMP.ConstraintRef}, L::Int64)

    indices = JuMP.index.(coupling_constraints)
    dual_solution = zeros(length(indices))
    dual_solution .= MOI.get(JuMP.backend(node.subproblem), MOI.ConstraintDual(), indices)

    @turbo for ℓ in 1:L
        α[1,ℓ] = dual_solution[ℓ] * exp(intercept[ℓ] + current_independent_noise_term[ℓ] * psi[ℓ])
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

   
function compute_alpha_t7!(α::Array{Float64,2}, ar_process_stage::LogLinearSDDP.AutoregressiveProcessStage, current_independent_noise_term::Any, coupling_constraints::Vector{JuMP.ConstraintRef}, L::Int64)

    dual_solution = JuMP.dual.(coupling_constraints)
    for ℓ in 1:L
        α[1,ℓ] = dual_solution[ℓ] * exp(ar_process_stage.intercept[ℓ]) * exp(current_independent_noise_term[ℓ] * ar_process_stage.psi[ℓ])
    end
end

function compute_alpha_tau!(α::Array{Float64,2}, cut_factors::Array{Float64,2}, cut_exponents::Any, ar_process_stage::LogLinearSDDP.AutoregressiveProcessStage, ar_parameters::Any, current_independent_noise_term::Any, t::Int64, T::Int64, L_t::Int64)

    for τ in t+1:T 
        L_τ = ar_parameters[τ].dimension
        for ℓ in 1:L_τ
            α[τ-t+1,ℓ] = cut_factors[τ-t,ℓ] * prod(exp(ar_process_stage.intercept[ν] * cut_exponents[τ,ℓ,ν,1]) * exp(current_independent_noise_term[ν] * cut_exponents[τ,ℓ,ν,1] * ar_process_stage.psi[ν]) for ν in 1:L_t)
        end
    end
end

function compute_alpha_tau2!(α::Array{Float64,2}, cut_factors::Array{Float64,2}, cut_exponents::Any, ar_process_stage::LogLinearSDDP.AutoregressiveProcessStage, ar_parameters::Any, current_independent_noise_term::Any, t::Int64, T::Int64, L_t::Int64)

    for τ in t+1:T 
        L_τ = ar_parameters[τ].dimension
        for ℓ in 1:L_τ
            α[τ-t+1,ℓ] = cut_factors[τ-t,ℓ] * exp(sum(ar_process_stage.intercept[ν] * cut_exponents[τ,ℓ,ν,1] + current_independent_noise_term[ν] * cut_exponents[τ,ℓ,ν,1] * ar_process_stage.psi[ν] for ν in 1:L_t))
        end
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

function compute_alpha_tau4!(α::Array{Float64,2}, cut_factors::Array{Float64,2}, cut_exponents::Any, ar_process_stage::LogLinearSDDP.AutoregressiveProcessStage, ar_parameters::Any, current_independent_noise_term::Any, t::Int64, T::Int64, L_t::Int64)

    beta = zeros(T-(t+1)+1,4)
    aux = zeros(T-(t+1)+1,4)
    Tullio.@tullio aux[τ-t+1,ℓ] = sum(cut_exponents[τ,ℓ,ν,1] * (ar_process_stage.intercept[ν] + current_independent_noise_term[ν] * ar_process_stage.psi[ν]) for ν in 1:L_t) (τ in t+1:T, ℓ in 1:4)
    Tullio.@tullio beta[τ-t,ℓ] = cut_factors[τ-t,ℓ] * exp(aux[τ-t,ℓ]) (τ in t+1:T)
    α[2:end,:] = beta
end

function compute_alpha_tau5!(α::Array{Float64,2}, cut_factors::Array{Float64,2}, cut_exponents::Any, ar_process_stage::LogLinearSDDP.AutoregressiveProcessStage, ar_parameters::Any, current_independent_noise_term::Any, t::Int64, T::Int64, L_t::Int64)

    aux = zeros(T-(t+1)+1,4)
    Tullio.@tullio aux[τ-t+1,ℓ] = sum(cut_exponents[τ,ℓ,ν,1] * (ar_process_stage.intercept[ν] + current_independent_noise_term[ν] * ar_process_stage.psi[ν]) for ν in 1:L_t) (τ in t+1:T, ℓ in 1:4)

    for τ in t+1:T 
        L_τ = ar_parameters[τ].dimension
        for ℓ in 1:L_τ
            α[τ-t+1,ℓ] = cut_factors[τ-t,ℓ] * exp(aux[τ-t,ℓ])
        end
    end
end

function compute_alpha_tau6!(α::Array{Float64,2}, cut_factors::Array{Float64,2}, cut_exponents::Any, ar_process_stage::LogLinearSDDP.AutoregressiveProcessStage, ar_parameters::Any, current_independent_noise_term::Any, t::Int64, T::Int64, L_t::Int64)

    β = @view α[2:end,:]
    Tullio.@tullio β[τ-t,ℓ] = cut_factors[τ-t,ℓ] * exp(sum(cut_exponents[τ,ℓ,ν,1] * (ar_process_stage.intercept[ν] + current_independent_noise_term[ν] * ar_process_stage.psi[ν]) for ν in 1:L_t)) (τ in t+1:T, ℓ in 1:4)
end

function compute_alpha_tau7!(α::Array{Float64,2}, cut_factors::Array{Float64,2}, cut_exponents::Any, ar_process_stage::LogLinearSDDP.AutoregressiveProcessStage, ar_parameters::Any, current_independent_noise_term::Any, t::Int64, T::Int64, L_t::Int64)

    β = @view α[2:end,:]
    Tullio.@tullio β[τ-t+1,ℓ] = sum(cut_exponents[τ,ℓ,ν,1] * (ar_process_stage.intercept[ν] + current_independent_noise_term[ν] * ar_process_stage.psi[ν]) for ν in 1:L_t) (τ in t+1:T, ℓ in 1:4)
    β = cut_factors .* exp.(β)
end

function compute_alpha_tau8!(α::Array{Float64,2}, cut_factors::Array{Float64,2}, cut_exponents::Any, ar_process_stage::LogLinearSDDP.AutoregressiveProcessStage, ar_parameters::Any, current_independent_noise_term::Any, t::Int64, T::Int64, L_t::Int64)

    for τ in t+1:T 
        L_τ = ar_parameters[τ].dimension
        for ℓ in 1:L_τ
            α[ℓ,τ-t+1] = cut_factors[τ-t,ℓ] * exp(sum(ar_process_stage.intercept[ν] * cut_exponents[ν,ℓ,τ,1] + current_independent_noise_term[ν] * cut_exponents[ν,ℓ,τ,1] * ar_process_stage.psi[ν] for ν in 1:L_t))
        end
    end
end

function compute_alpha_tau9!(α::Array{Float64,2}, cut_factors::Array{Float64,2}, cut_exponents::Any, ar_process_stage::LogLinearSDDP.AutoregressiveProcessStage, ar_parameters::Any, current_independent_noise_term::Any, t::Int64, T::Int64, L_t::Int64)

    for τ in t+1:T 
        L_τ = ar_parameters[τ].dimension
        for ℓ in 1:L_τ
            α[τ-t+1,ℓ] = cut_factors[τ-t,ℓ] * exp(cut_exponents[τ,ℓ,ℓ,1] * ar_process_stage.intercept[ℓ] + cut_exponents[τ,ℓ,ℓ,1] * current_independent_noise_term[ℓ] * ar_process_stage.psi[ℓ])
        end
    end
end

function compute_alpha_tau10!(α::Array{Float64,2}, cut_factors::Array{Float64,2}, cut_exponents::Any, ar_process_stage::LogLinearSDDP.AutoregressiveProcessStage, ar_parameters::Any, current_independent_noise_term::Any, t::Int64, T::Int64, L_t::Int64)
    
    β = @view α[2:end,:]
    Tullio.@tullio β[τ-t+1,ℓ] = cut_exponents[τ,ℓ,ℓ,1] * (ar_process_stage.intercept[ℓ] + current_independent_noise_term[ℓ] * ar_process_stage.psi[ℓ]) (τ in t+1:T, ℓ in 1:4)
    β = cut_factors .* exp.(β)
end

function compute_alpha_tau11!(α::Array{Float64,2}, cut_factors::Array{Float64,2}, cut_exponents::Any, ar_process_stage::LogLinearSDDP.AutoregressiveProcessStage, ar_parameters::Any, current_independent_noise_term::Any, t::Int64, T::Int64, L_t::Int64)

    for ℓ in 1:4
        expo_factor = ar_process_stage.intercept[ℓ] + current_independent_noise_term[ℓ] * ar_process_stage.psi[ℓ]
        for τ in t+1:T 
            α[τ-t+1,ℓ] = cut_factors[τ-t,ℓ] * exp(cut_exponents[τ,ℓ,ℓ,1] * expo_factor)
        end
    end
end

function compute_alpha_tau12!(α::Array{Float64,2}, cut_factors::Array{Float64,2}, cut_exponents::Array{Float64,4}, intercept::Vector{Float64}, psi::Vector{Float64}, current_independent_noise_term::Any, t::Int64, T::Int64, L_t::Int64)

    for ℓ in 1:4
        for τ in t+1:T 
            α[τ-t+1,ℓ] = cut_factors[τ-t,ℓ] * exp(cut_exponents[τ,ℓ,ℓ,1] * (intercept[ℓ] + current_independent_noise_term[ℓ] * psi[ℓ]))
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

function compute_alpha_tau11b!(α::Array{Float64,2}, cut_factors::Array{Float64,2}, cut_exponents::Any, ar_process_stage::LogLinearSDDP.AutoregressiveProcessStage, ar_parameters::Any, current_independent_noise_term::Any, t::Int64, T::Int64, L_t::Int64)

    @inbounds for ℓ in 1:4
        expo_factor = ar_process_stage.intercept[ℓ] + current_independent_noise_term[ℓ] * ar_process_stage.psi[ℓ]
        for τ in t+1:T 
            α[τ-t+1,ℓ] = cut_factors[τ-t,ℓ] * exp(cut_exponents[τ,ℓ,ℓ,1] * expo_factor)
        end
    end
end

function compute_alpha_tau2b!(α::Array{Float64,2}, cut_factors::Array{Float64,2}, cut_exponents::Any, ar_process_stage::LogLinearSDDP.AutoregressiveProcessStage, ar_parameters::Any, current_independent_noise_term::Any, t::Int64, T::Int64, L_t::Int64)

    @inbounds for τ in t+1:T 
        for ℓ in 1:4
            expo_factor = 0.0
            for ν in 1:4
                expo_factor += ar_process_stage.intercept[ν] * cut_exponents[τ,ℓ,ν,1] + current_independent_noise_term[ν] * cut_exponents[τ,ℓ,ν,1] * ar_process_stage.psi[ν]
            end
            α[τ-t+1,ℓ] = cut_factors[τ-t,ℓ] * exp(expo_factor)
        end
    end
end

function compute_alpha_tau2c!(α::Array{Float64,2}, cut_factors::Array{Float64,2}, cut_exponents::Any, ar_process_stage::LogLinearSDDP.AutoregressiveProcessStage, ar_parameters::Any, current_independent_noise_term::Any, t::Int64, T::Int64, L_t::Int64)

    #TODO

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
        #compute_alpha_t6!($α, $node, $ar_process_stage, collect($current_independent_noise_term), $node.subproblem.ext[:coupling_constraints], $L_t)
    end   

    #BenchmarkTools.@btime compute_alpha_t!($α, $ar_process_stage, $current_independent_noise_term, $node.subproblem.ext[:coupling_constraints], $L_t) 
    #BenchmarkTools.@btime compute_alpha_t2!($α, $ar_process_stage, $current_independent_noise_term, $node.subproblem.ext[:coupling_constraints], $L_t) 
    #BenchmarkTools.@btime compute_alpha_t3!($α, $node, $ar_process_stage, $current_independent_noise_term, $node.subproblem.ext[:coupling_constraints], $L_t) 
    #BenchmarkTools.@btime compute_alpha_t4!($α, $node, $ar_process_stage, collect($current_independent_noise_term), $node.subproblem.ext[:coupling_constraints], $L_t)
    #BenchmarkTools.@btime compute_alpha_t5!($α, $node, $ar_process_stage.intercept, $ar_process_stage.psi, collect($current_independent_noise_term), $node.subproblem.ext[:coupling_constraints], $L_t)
    #BenchmarkTools.@btime compute_alpha_t6!($α, $node, $ar_process_stage, collect($current_independent_noise_term), $node.subproblem.ext[:coupling_constraints], $L_t)
    #BenchmarkTools.@btime compute_alpha_t7!($α, $ar_process_stage, $current_independent_noise_term, $node.subproblem.ext[:coupling_constraints], $L_t) 

    # Get cut constraint duals and compute first factor
    if t < T
        # TimerOutputs.@timeit model.timer_output "existing_cut_factor_1" begin
        #    cut_factors = get_existing_cuts_factors1(node.bellman_function.global_theta.cuts, node.subproblem)
        # end
        # TimerOutputs.@timeit model.timer_output "existing_cut_factor_3" begin
        #    cut_factors = get_existing_cuts_factors3(node.bellman_function.global_theta.cuts, node.subproblem)
        # end
        # TimerOutputs.@timeit model.timer_output "existing_cut_factor_4" begin
        #    cut_factors = get_existing_cuts_factors4(node.bellman_function.global_theta.cuts, node)
        # end
        # TimerOutputs.@timeit model.timer_output "existing_cut_factor_5" begin
        #    cut_factors = BenchmarkTools.@btime get_existing_cuts_factors5($node.bellman_function.global_theta.cuts, $node)
        # end
        # TimerOutputs.@timeit model.timer_output "existing_cut_factor_6" begin
        #    cut_factors = get_existing_cuts_factors6(node.bellman_function.global_theta.cuts, node)
        # end
        TimerOutputs.@timeit model.timer_output "existing_cut_factor_7" begin
           cut_factors = get_existing_cuts_factors7(node.bellman_function.global_theta.cuts, node)
        end
        
        # cut_factors = BenchmarkTools.@btime get_existing_cuts_factors1($node.bellman_function.global_theta.cuts, $node.subproblem)
        # cut_factors = BenchmarkTools.@btime get_existing_cuts_factors3($node.bellman_function.global_theta.cuts, $node.subproblem)
        # cut_factors = BenchmarkTools.@btime get_existing_cuts_factors4($node.bellman_function.global_theta.cuts, $node)
        # cut_factors = BenchmarkTools.@btime get_existing_cuts_factors5($node.bellman_function.global_theta.cuts, $node)
        # cut_factors = BenchmarkTools.@btime get_existing_cuts_factors6($node.bellman_function.global_theta.cuts, $node)
        # cut_factors = BenchmarkTools.@btime get_existing_cuts_factors7($node.bellman_function.global_theta.cuts, $node)
        
        # Case τ > t
        TimerOutputs.@timeit model.timer_output "alpha_tau_new" begin
            #compute_alpha_tau11!(α, cut_factors, model.ext[:cut_exponents][t+1], ar_process_stage, ar_process.parameters, current_independent_noise_term, t, T, L_t)
            compute_alpha_tau13!(α, cut_factors, model.ext[:cut_exponents][t+1],  ar_process_stage.intercept, ar_process_stage.psi, current_independent_noise_term, t, T, L_t)
        end

        # α = Array{Float64,2}(undef, T-t+1, L)
        # # Case τ > t
        # compute_alpha_t!(α, ar_process_stage, current_independent_noise_term, node.subproblem.ext[:coupling_constraints], L_t)
        # TimerOutputs.@timeit model.timer_output "alpha_tau_new" begin
        #     compute_alpha_tau!(α, cut_factors, model.ext[:cut_exponents][t+1], ar_process_stage, ar_process.parameters, current_independent_noise_term, t, T, L_t)
        # end
        # println(α)

        # α = Array{Float64,2}(undef, T-t+1, L)
        # # Case τ > t
        # compute_alpha_t!(α, ar_process_stage, current_independent_noise_term, node.subproblem.ext[:coupling_constraints], L_t)
        # TimerOutputs.@timeit model.timer_output "alpha_tau_new" begin
        #     compute_alpha_tau2!(α, cut_factors, model.ext[:cut_exponents][t+1], ar_process_stage, ar_process.parameters, current_independent_noise_term, t, T, L_t)
        # end
        # println(α)

        # α = Array{Float64,2}(undef, T-t+1, L)
        # # Case τ > t
        # compute_alpha_t!(α, ar_process_stage, current_independent_noise_term, node.subproblem.ext[:coupling_constraints], L_t)
        # TimerOutputs.@timeit model.timer_output "alpha_tau_new" begin
        #     compute_alpha_tau3!(α, cut_factors, model.ext[:cut_exponents][t+1], ar_process_stage, ar_process.parameters, current_independent_noise_term, t, T, L_t)
        # end
        # println(α)

        # α = Array{Float64,2}(undef, T-t+1, L)
        # # Case τ > t
        # compute_alpha_t!(α, ar_process_stage, current_independent_noise_term, node.subproblem.ext[:coupling_constraints], L_t)
        # TimerOutputs.@timeit model.timer_output "alpha_tau_new" begin
        #     compute_alpha_tau9!(α, cut_factors, model.ext[:cut_exponents][t+1], ar_process_stage, ar_process.parameters, current_independent_noise_term, t, T, L_t)
        # end
        # println(α)

        # α = Array{Float64,2}(undef, T-t+1, L)
        # # Case τ > t
        # compute_alpha_t!(α, ar_process_stage, current_independent_noise_term, node.subproblem.ext[:coupling_constraints], L_t)
        # TimerOutputs.@timeit model.timer_output "alpha_tau_new" begin
        #     compute_alpha_tau10!(α, cut_factors, model.ext[:cut_exponents][t+1], ar_process_stage, ar_process.parameters, current_independent_noise_term, t, T, L_t)
        # end
        # println(α)
        # println()

        # α = Array{Float64,2}(undef, T-t+1, L)
        # # Case τ > t
        # compute_alpha_t!(α, ar_process_stage, current_independent_noise_term, node.subproblem.ext[:coupling_constraints], L_t)
        # TimerOutputs.@timeit model.timer_output "alpha_tau_new" begin
        #     compute_alpha_tau11!(α, cut_factors, model.ext[:cut_exponents][t+1], ar_process_stage, ar_process.parameters, current_independent_noise_term, t, T, L_t)
        # end
        # println(α)
        # println()

        # α = Array{Float64,2}(undef, T-t+1, L)
        # # Case τ > t
        # compute_alpha_t!(α, ar_process_stage, current_independent_noise_term, node.subproblem.ext[:coupling_constraints], L_t)
        # TimerOutputs.@timeit model.timer_output "alpha_tau_new" begin
        #     compute_alpha_tau12!(α, cut_factors, model.ext[:cut_exponents][t+1], ar_process_stage.intercept, ar_process_stage.psi, current_independent_noise_term, t, T, L_t)
        # end
        # println(α)
        # println()


        # TimerOutputs.@timeit model.timer_output "alpha_tau_new4" begin
        #     compute_alpha_tau7!(α, cut_factors, model.ext[:cut_exponents][t+1], ar_process_stage, ar_process.parameters, current_independent_noise_term, t, T, L_t)
        # end

        #β = permutedims(α, [2, 1])
        #expo = permutedims(model.ext[:cut_exponents][t+1], [3, 2, 1, 4])

        # TimerOutputs.@timeit model.timer_output "alpha_tau_new4" begin
        #     compute_alpha_tau7!(α, cut_factors, model.ext[:cut_exponents][t+1], ar_process_stage, ar_process.parameters, current_independent_noise_term, t, T, L_t)
        # end
        
        #Infiltrator.@infiltrate
        #BenchmarkTools.@btime compute_alpha_tau!($α, $cut_factors, $model.ext[:cut_exponents][$t+1], $ar_process_stage, $ar_process.parameters, $current_independent_noise_term, $t, $T, $L_t)
        #BenchmarkTools.@btime compute_alpha_tau2!($α, $cut_factors, $model.ext[:cut_exponents][$t+1], $ar_process_stage, $ar_process.parameters, $current_independent_noise_term, $t, $T, $L_t)
        #BenchmarkTools.@btime compute_alpha_tau2b!($α, $cut_factors, $model.ext[:cut_exponents][$t+1], $ar_process_stage, $ar_process.parameters, $current_independent_noise_term, $t, $T, $L_t)
        #BenchmarkTools.@btime compute_alpha_tau3!($α, $cut_factors, $model.ext[:cut_exponents][$t+1], $ar_process_stage, $ar_process.parameters, $current_independent_noise_term, $t, $T, $L_t)
        #BenchmarkTools.@btime compute_alpha_tau4!($α, $cut_factors, $model.ext[:cut_exponents][$t+1], $ar_process_stage, $ar_process.parameters, $current_independent_noise_term, $t, $T, $L_t)
        #BenchmarkTools.@btime compute_alpha_tau5!($α, $cut_factors, $model.ext[:cut_exponents][$t+1], $ar_process_stage, $ar_process.parameters, $current_independent_noise_term, $t, $T, $L_t)
        #BenchmarkTools.@btime compute_alpha_tau6!($α, $cut_factors, $model.ext[:cut_exponents][$t+1], $ar_process_stage, $ar_process.parameters, $current_independent_noise_term, $t, $T, $L_t)
        #BenchmarkTools.@btime compute_alpha_tau7!($α, $cut_factors, $model.ext[:cut_exponents][$t+1], $ar_process_stage, $ar_process.parameters, $current_independent_noise_term, $t, $T, $L_t)
        #BenchmarkTools.@btime compute_alpha_tau8!($β, $cut_factors, $expo, $ar_process_stage, $ar_process.parameters, $current_independent_noise_term, $t, $T, $L_t)
        #BenchmarkTools.@btime compute_alpha_tau9!($α, $cut_factors, $model.ext[:cut_exponents][$t+1], $ar_process_stage, $ar_process.parameters, $current_independent_noise_term, $t, $T, $L_t)
        #BenchmarkTools.@btime compute_alpha_tau10!($α, $cut_factors, $model.ext[:cut_exponents][$t+1], $ar_process_stage, $ar_process.parameters, $current_independent_noise_term, $t, $T, $L_t)
        #BenchmarkTools.@btime compute_alpha_tau11!($α, $cut_factors, $model.ext[:cut_exponents][$t+1], $ar_process_stage, $ar_process.parameters, $current_independent_noise_term, $t, $T, $L_t)
        #BenchmarkTools.@btime compute_alpha_tau12!($α, $cut_factors, $model.ext[:cut_exponents][$t+1], $ar_process_stage.intercept, $ar_process_stage.psi, $current_independent_noise_term, $t, $T, $L_t)
        #BenchmarkTools.@btime compute_alpha_tau13!($α, $cut_factors, $model.ext[:cut_exponents][$t+1], $ar_process_stage.intercept, $ar_process_stage.psi, $current_independent_noise_term, $t, $T, $L_t)
        #BenchmarkTools.@btime compute_alpha_tau11b!($α, $cut_factors, $model.ext[:cut_exponents][$t+1], $ar_process_stage, $ar_process.parameters, $current_independent_noise_term, $t, $T, $L_t)
        
        #α = permutedims(β, [2, 1])
        #Infiltrator.@infiltrate
    end

    return α

end

duality_log_key(::ContinuousConicDuality) = " "

