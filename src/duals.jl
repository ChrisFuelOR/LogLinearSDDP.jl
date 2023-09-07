struct ContinuousConicDuality <: SDDP.AbstractDualityHandler end

function get_dual_solution(node::SDDP.Node, noise_index::Int64, ::ContinuousConicDuality)
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

    TimerOutputs.@timeit model.timer_output "compute_alphas" begin
        # We also need the dual variables for all coupling constraints.
        # In order to identify the coupling constraints, we should specify them in the problem definition.
        # Moreover, we need the dual variables for all existing cut constraints.
        model = SDDP.get_policy_graph(node.subproblem)
        t = node.node_index
        T = model.ext[:problem_params].number_of_stages
        current_independent_noise_term = SDDP.get_noise_terms(sampling_scheme, node, node_index)[noise_index].term
        cut_exponents_required = model.ext[:cut_exponents][t+1]
        autoregressive_data_stage = model.ext[:autoregressive_data].ar_data[t]
        
        α = Array{Float64,2}(undef, T-t+1, L)

        for τ in t:T 
            L_τ = autoregressive_data.ar_data[τ].ar_dimension
            for ℓ in 1:L_τ
                if τ == t
                    # Get coupling constraint reference
                    coupling_ref = node.ext[:coupling_constraints][ℓ]
                    π = JuMP.dual(coupling_ref) #TODO: dual_sign  

                    # Compute alpha value
                    α[τ,ℓ] = π * exp(autoregressive_data_stage.ar_intercept[ℓ]) * exp(node.ext[:current_independent_noise_term][ℓ])
                else
                    # Get cut constraint duals and compute first factor
                    factor_1 = get_existing_cuts_factor(node, t, τ, ℓ)

                    # Compute second factor
                    factor_2 = prod(exp(autoregressive_data_stage.ar_intercept[ν] * cut_exponents_required[τ,ℓ,ν,t]) * exp(current_independent_noise_term[ℓ] * cut_exponents_required[τ,ℓ,ν,t]) for ν in 1:L_t)

                    # Compute alpha value
                    α[τ,ℓ] = factor_1 * factor_2
                end
            end
        end
    end

    return objective_value(node.subproblem), λ, α
end


function _relax_integrality(node::SDDP.Node)
    if !node.has_integrality
        return () -> nothing
    end
    return JuMP.relax_integrality(node.subproblem)
end


function prepare_backward_pass(node::SDDP.Node, ::ContinuousConicDuality, ::SDDP.Options)
    return _relax_integrality(node)
end


function get_existing_cuts_factor(node::SDDP.Node, t::Int64, τ::Int64, ℓ::Int64)

    factor = 0

    for cut in node.bellman_function.global_theta.cuts
        # Get optimal dual value of cut constraint and alpha value for given cut to update the factor
        factor = factor + JuMP.dual(cut.cut_constraint) * cut.intercept_factors[τ-t+1,ℓ]
    end

    return factor
end
