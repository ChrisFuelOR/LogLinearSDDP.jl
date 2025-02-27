# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Copyright (c) 2023 Christian Fuellner <christian.fuellner@kit.edu>
################################################################################

""" 
Pre-computation of the cut intercept exponents once before the algorithm is started.

For simplicity, we assume constant L for the array. Non-required values will just not be defined.
"""

function compute_cut_exponents(
    problem_params::LogLinearSDDP.ProblemParams,
    ar_process::LogLinearSDDP.AutoregressiveProcess
)

    T = problem_params.number_of_stages
    L = LogLinearSDDP.get_max_dimension(ar_process)
    p = ar_process.lag_order

    cut_exponents = Vector{Array{Float64,4}}(undef, T)
    
    #println("Θ(t, τ, ℓ, m, k), cut_exponents[t, τ, ℓ, m, t-k), k is stage, t-k is lag.")
    for t in T:-1:2
        cut_exponents_stage = zeros(T, L, L, p)
        ar_process_stage = ar_process.parameters[t]
        L_t = ar_process_stage.dimension

        for τ in t:T

            if τ == t
                for ℓ in 1:L_t
                    for k in t-p:t-1
                        if k <= 1
                            L_k = get_max_dimension(ar_process)
                        else
                            L_k = ar_process.parameters[k].dimension
                        end 
                        for m in 1:L_k
                            cut_exponents_stage[τ,ℓ,m,t-k] = ar_process_stage.coefficients[ℓ,m,t-k]
                            #println("Θ(", t, ",", τ, ",", ℓ, ",", m, ",", k, ") = cut_exponents(", t, ",", τ, ",", ℓ, ",", m, ",", t-k, "): ", cut_exponents_stage[τ,ℓ,m,t-k])
                        end
                    end
                end
                
            else
                L_τ = ar_process.parameters[τ].dimension 
                for ℓ in 1:L_τ
                    for k in t-p:t-1
                        if k <= 1
                            L_k = get_max_dimension(ar_process)
                        else
                            L_k = ar_process.parameters[k].dimension
                        end
                        for m in 1:L_k
                            if k == t-p
                                value = 0.0
                                for ν in 1:L_t
                                    value = value + ar_process_stage.coefficients[ν,m,p] * cut_exponents[t+1][τ,ℓ,ν,(t+1)-t] 
                                end
                                cut_exponents_stage[τ,ℓ,m,p] = value
                                #println("Θ(", t, ",", τ, ",", ℓ, ",", m, ",", k, ") = cut_exponents(", t, ",", τ, ",", ℓ, ",", m, ",", t-k, "): ", cut_exponents_stage[τ,ℓ,m,t-k])
                            else
                                value = cut_exponents[t+1][τ,ℓ,m,t+1-k]  
                                for ν in 1:L_t
                                    value = value + ar_process_stage.coefficients[ν,m,t-k] * cut_exponents[t+1][τ,ℓ,ν,(t+1)-t] 
                                end
                                cut_exponents_stage[τ,ℓ,m,t-k] = value
                                #println("Θ(", t, ",", τ, ",", ℓ, ",", m, ",", k, ") = cut_exponents(", t, ",", τ, ",", ℓ, ",", m, ",", t-k, "): ", cut_exponents_stage[τ,ℓ,m,t-k])
                            end
                        end
                    end
                end
                
            end
        end

        cut_exponents[t] = cut_exponents_stage
    end

    cut_exponents[1] = zeros(T, L, L, p)

    return cut_exponents
end

#TODO: If we sum about more values than that, what happens? Are the added values guaranteed to be 0? Does that match the description in the paper?


""" 
Evaluation of all cut intrcepts for a given problem and the given scenario (process state/history) at hand.
"""

function evaluate_cut_intercepts(
    node::SDDP.Node,
    noise_term::Union{Float64,Any},
)

    # Preparation steps
    subproblem = node.subproblem
    model = SDDP.get_policy_graph(subproblem)
    problem_params = model.ext[:problem_params]
    cut_exponents = model.ext[:cut_exponents]
    process_state = node.ext[:process_state]
    autoregressive_data = model.ext[:ar_process]
    t = node.index
    T = problem_params.number_of_stages

    if !isempty(node.bellman_function.global_theta.cuts)
        # Get exponents for the considered cut
        cut_exponents_stage = cut_exponents[t+1] #current stage + 1 (on stage t, a (t+1)-stage cut is evaluated)

        # Get process state for the considered cut
        process_state_after_realization = update_process_state(model, model.ext[:ar_process].lag_order, t+1, process_state, noise_term, false)

        L = LogLinearSDDP.get_max_dimension(autoregressive_data)
        p = autoregressive_data.lag_order
        ps_array = Array{Float64,2}(undef, L, p)
        for k in 1:p
            for m in 1:4
                ps_array[m,k] = process_state_after_realization[t+1-k][m]
            end
        end

        scenario_factors = ones(T-(t+1)+1, 4)

        # First compute scenario-specific factors
        TimerOutputs.@timeit model.timer_output "scenario_factors" begin
            compute_scenario_factors8!(t+1, ps_array, problem_params, cut_exponents_stage, scenario_factors, autoregressive_data)
        end

        # Iterate over all cuts and adapt intercept
        TimerOutputs.@timeit model.timer_output "adapt_intercepts" begin  
            set_intercepts3(node, node.bellman_function.global_theta.cuts, t+1, scenario_factors, problem_params, autoregressive_data, model.timer_output)
 
        end
    end

    return

end

function set_intercepts(
    cuts::Vector{LogLinearSDDP.Cut},
    t::Int64,
    scenario_factors::Array{Float64,2},
    problem_params::LogLinearSDDP.ProblemParams,
    ar_process::LogLinearSDDP.AutoregressiveProcess,
    timer_output::Any,
    )

    for cut in cuts
        TimerOutputs.@timeit timer_output "compute_value" begin    
            intercept_value = compute_intercept_value2b(t, cut, scenario_factors, problem_params, ar_process)
        end
        
        TimerOutputs.@timeit timer_output "fix_value" begin    
            JuMP.fix(cut.cut_intercept_variable, intercept_value)
        end
    end
end

function set_intercepts3(
    node::SDDP.Node,
    cuts::Vector{LogLinearSDDP.Cut},
    t::Int64,
    scenario_factors::Array{Float64,2},
    problem_params::LogLinearSDDP.ProblemParams,
    ar_process::LogLinearSDDP.AutoregressiveProcess,
    timer_output::Any,
    )

    for cut_index in eachindex(cuts)
        TimerOutputs.@timeit timer_output "intercept_value" begin    
            intercept_value = compute_intercept_value2b(t, cuts[cut_index], scenario_factors, problem_params, ar_process)
        end

        #x = Gurobi.c_column(JuMP.backend(node.subproblem), JuMP.index(cuts[cut_index].cut_intercept_variable))
        #Infiltrator.@infiltrate
        x = Gurobi.c_column(JuMP.backend(node.subproblem), MOI.VariableIndex(167 + cut_index))
        Gurobi.GRBsetdblattrelement(JuMP.backend(node.subproblem), "LB", x, intercept_value) 
        Gurobi.GRBsetdblattrelement(JuMP.backend(node.subproblem), "UB", x, intercept_value) 

    end
end


""" 
Pre-computation of the scenario-dependeGnt cut intercept factors (scenario factors) for a given problem and a scenario at hand.
Note that this value is the same for all cuts of a given problem for a given scenario, so it should be just computed
once instead of being included in the _evaluate_cut_intercept function call for each scenario.
"""

function compute_scenario_factors8!(
    t::Int64,
    ps_array::Array{Float64,2},
    problem_params::LogLinearSDDP.ProblemParams,
    cut_exponents_stage::Array{Float64,4},
    scenario_factors::Array{Float64,2},
    ar_process::LogLinearSDDP.AutoregressiveProcess,
)

    T = problem_params.number_of_stages
    L = LogLinearSDDP.get_max_dimension(ar_process)
    p = ar_process.lag_order
    fill!(scenario_factors, 1.0)

    @turbo for k in t-p:t-1
        for m in 1:4
            for ℓ in 1:L
                for τ in t:T
                   scenario_factors[τ-t+1,ℓ] *= ps_array[m,k-(t-p)+1] ^ cut_exponents_stage[τ,ℓ,m,t-k] 
                end
            end
        end
    end
    return
end

function compute_scenario_factors(
    t::Int64,
    process_state::Dict{Int64, Any},
    problem_params::LogLinearSDDP.ProblemParams,
    cut_exponents_stage::Array{Float64,4},
    ar_process::LogLinearSDDP.AutoregressiveProcess,
)

    T = problem_params.number_of_stages
    L = LogLinearSDDP.get_max_dimension(ar_process)
    p = ar_process.lag_order
    scenario_factors = ones(T-(t-1), L)

    for k in t-p:t-1
        if k <= 1
            L_k = L
        else
            L_k = ar_process.parameters[k].dimension
        end

        for m in 1:L_k
            scenario_factors = scenario_factors .* process_state[k][m] .^ cut_exponents_stage[t:T,1:L,m,t-k] 
        end
    end

    return scenario_factors

end

function compute_intercept_value2b(
    t::Int64,
    cut::LogLinearSDDP.Cut,
    scenario_factors::Array{Float64,2},
    problem_params::LogLinearSDDP.ProblemParams,
    ar_process::LogLinearSDDP.AutoregressiveProcess,
)

    T = problem_params.number_of_stages

    #Evaluate the intercept
    intercept_value = cut.deterministic_intercept
    @turbo for ℓ in 1:4
        for τ in t:T
            intercept_value += cut.intercept_factors[τ-t+1,ℓ] * scenario_factors[τ-t+1,ℓ]
        end
    end

    return intercept_value

end

function compute_stochastic_intercept_value_tight(
    t::Int64,
    T::Int64,
    scenario_factors::Array{Float64,2},
    intercept_factors::Array{Float64,2},
    ar_process::LogLinearSDDP.AutoregressiveProcess,
)

    intercept_value = 0.0
    for τ in t:T
        L_τ = ar_process.parameters[τ].dimension
        for ℓ in 1:L_τ
            intercept_value = intercept_value + intercept_factors[τ-t+1,ℓ] * scenario_factors[τ-t+1,ℓ]
        end
    end

    return intercept_value

end

""" 
Evaluation the cut intercept for the about to be created cut at the state of construction (point of tightness)
"""

function evaluate_cut_intercept_tight(
    node::SDDP.Node,
    intercept_factors::Array{Float64,2},
)

    # Preparation steps
    subproblem = node.subproblem
    model = SDDP.get_policy_graph(subproblem)
    problem_params = model.ext[:problem_params]
    cut_exponents = model.ext[:cut_exponents]
    process_state = node.ext[:process_state]
    ar_process = model.ext[:ar_process]
    t = node.index
    T = problem_params.number_of_stages
    L = LogLinearSDDP.get_max_dimension(ar_process)

    # Get exponents for the considered cut
    cut_exponents_stage = cut_exponents[t]

    # Get process state for the considered cut
    process_state = node.ext[:process_state]

    # First compute scenario-specific factors
    TimerOutputs.@timeit model.timer_output "scenario_factors" begin
        scenario_factors = compute_scenario_factors(t, process_state, problem_params, cut_exponents_stage, ar_process)
    end

    #Evaluate the stochastic part of the intercept
    intercept_value = compute_stochastic_intercept_value_tight(t, T, scenario_factors, intercept_factors, ar_process)

    return intercept_value
end
