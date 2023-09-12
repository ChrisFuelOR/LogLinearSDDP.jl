""" 
Pre-computation of the cut intercept exponents once before the algorithm is started.

For simplicity, we assume constant L for the array. Non-required values will just not be defined.
"""

function compute_cut_exponents(
    problem_params::LogLinearSDDP.ProblemParams,
    autoregressive_data::LogLinearSDDP.AutoregressiveData
)

    T = problem_params.number_of_stages
    L = LogLinearSDDP.get_max_dimension(autoregressive_data)
    p = autoregressive_data.ar_lag_order

    cut_exponents = Vector{Array{Float64,4}}(undef, T)
    
    for t in T:-1:2
        cut_exponents_stage = zeros(T, L, L, p)
        #cut_exponents_stage = Array{Float64,4}(undef, T, L, L, p)
        autoregressive_data_stage = autoregressive_data.ar_data[t]
        L_t = autoregressive_data_stage.ar_dimension

        for τ in t:T

            if τ == t
                for ℓ in 1:L_t
                    for k in t-p:t-1
                        L_k = autoregressive_data.ar_data[k].ar_dimension 
                        for m in 1:L_k
                            cut_exponents_stage[τ,ℓ,m,t-k] = autoregressive_data_stage.ar_coefficients[ℓ,m,t-k]
                        end
                    end
                end
                
            else
                L_τ = autoregressive_data.ar_data[τ].ar_dimension 
                for ℓ in 1:L_τ
                    for k in t-p:t-1
                        L_k = autoregressive_data.ar_data[k].ar_dimension
                        for m in 1:L_k
                            if k == t-p
                                value = 0.0
                                for ν in 1:L_t
                                    value = value + autoregressive_data_stage.ar_coefficients[ν,m,p] * cut_exponents[t+1][τ,ℓ,ν,(t+1)-t] 
                                end
                                cut_exponents_stage[τ,ℓ,m,k] = value
                            else
                                value = cut_exponents[t+1][τ,ℓ,m,t-k]  
                                for ν in 1:L_t
                                    value = value + autoregressive_data_stage.ar_coefficients[ν,m,t-k] * cut_exponents[t+1][τ,ℓ,ν,(t+1)-t] 
                                end
                                cut_exponents_stage[τ,ℓ,m,t-k] = value
                            end
                        end
                    end
                end
                
            end
        end

        cut_exponents[t] = cut_exponents_stage
    end

    #cut_exponents[1] = Array{Float64,4}(undef, T, L, L, p)
    cut_exponents[1] = zeros(T, L, L, p)

    return cut_exponents
end

#TODO: If we sum about more values than that, what happens? Are the added values guaranteed to be 0? Does that match the description in the paper?


""" 
Evaluation of all cut intrcepts for a given problem and the given scenario (process state/history) at hand.
"""

function evaluate_cut_intercepts(
    node::SDDP.Node
)

    # Preparation steps
    subproblem = node.subproblem
    model = SDDP.get_policy_graph(subproblem)
    problem_params = model.ext[:problem_params]
    cut_exponents = model.ext[:cut_exponents]
    process_state = node.ext[:process_state]
    autoregressive_data = model.ext[:autoregressive_data]

    # Get current process state
    process_state = node.ext[:process_state]

    t = node.index #current stage
    cut_exponents_stage = cut_exponents[t]

    # First compute scenario-specific factors
    scenario_factors = compute_scenario_factors(t, process_state, problem_params, cut_exponents_stage, autoregressive_data)

    # Iterate over all cuts and adapt intercept
    for cut in node.bellman_function.global_theta.cuts 
        evaluate_cut_intercept(t, cut, scenario_factors, problem_params, autoregressive_data)
    end

    return

end


""" 
Pre-computation of the scenario-dependent cut intercept factors (scenario factors) for a given problem and a scenario at hand.
Note that this value is the same for all cuts of a given problem for a given scenario, so it should be just computed
once instead of being included in the _evaluat_cut_intercept function call for each scenario.
"""

function compute_scenario_factors(
    t::Int64,
    process_state::Dict{Int64, Vector{Float64}},
    problem_params::LogLinearSDDP.ProblemParams,
    cut_exponents_stage::Array{Float64,4},
    autoregressive_data::LogLinearSDDP.AutoregressiveData,
)

    T = problem_params.number_of_stages
    L = LogLinearSDDP.get_max_dimension(autoregressive_data)
    p = autoregressive_data.ar_lag_order
    scenario_factors = Array{Float64,2}(undef, T, L)

    for τ in t:T
        L_τ = autoregressive_data.ar_data[τ].ar_dimension
        for ℓ in 1:L_τ 
            scenario_factor = 1.0
            
            for k in t-p:t-1
                if k < 1
                    L_k = get_max_dimension(autoregressive_data)
                else
                    L_k = autoregressive_data.ar_data[k].ar_dimension
                end

                for m in 1:L_k
                    scenario_factor = scenario_factor * process_state[k][m] ^ cut_exponents_stage[τ,ℓ,m,t-k] 
                end
            end

            scenario_factors[τ,ℓ] = scenario_factor
        end
    end

    return scenario_factors

end


""" 
Evaluation of the cut intercept for the given cut and the given scenario (process state/history) at hand.
"""

function evaluate_cut_intercept(
    t::Int,
    cut::LogLinearSDDP.Cut,
    scenario_factors::Array{Float64,2},
    problem_params::LogLinearSDDP.ProblemParams,
    autoregressive_data::LogLinearSDDP.AutoregressiveData,
)

    intercept_variable = cut.cut_intercept_variable
    T = problem_params.number_of_stages
    p = autoregressive_data.ar_lag_order

    #Evaluate the intercept
    intercept_value = 0.0
    for τ in t:T
        L_τ = autoregressive_data.ar_data[τ].ar_dimension
        for ℓ in 1:L_τ
            intercept_value = intercept_value + cut.intercept_factors[τ-t+1,ℓ] * scenario_factors[τ,ℓ]
        end
    end

    #Fix the intercept variable
    JuMP.fix(intercept_variable, intercept_value)

    return

end
