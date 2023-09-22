# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Copyright (c) 2023 Christian Fuellner <christian.fuellner@kit.edu>
################################################################################

import SDDP
import Gurobi
import JuMP
import LogLinearSDDP
import Infiltrator
import Revise


function model_definition(ar_process::LogLinearSDDP.AutoregressiveProcess, problem_params::LogLinearSDDP.ProblemParams, algo_params::LogLinearSDDP.AlgoParams)

    model = SDDP.LinearPolicyGraph(
        stages = problem_params.number_of_stages,
        optimizer = Gurobi.Optimizer,
        sense = :Min,
        lower_bound = 0.0,
    ) do sp, t

        JuMP.@variable(sp, 5 <= x <= 15, SDDP.State, initial_value = 10)
        JuMP.@variable(sp, g_t >= 0)
        JuMP.@variable(sp, g_h >= 0)
        JuMP.@variable(sp, s >= 0)
        JuMP.@variable(sp, demand)
        JuMP.@variable(sp, inflow)
        coupling_ref_1 = JuMP.@constraint(sp, water_bal, x.out - x.in + g_h + s == inflow)
        coupling_ref_2 = JuMP.@constraint(sp, demand_bal, g_h + g_t == demand) # actually no coupling constraint
        SDDP.@stageobjective(sp, s + t * g_t)

        # Parameterize inflow and demand
        if t == 1
            realizations = [[0.0, 0.0]]
        else
            realizations = ar_process.parameters[t].eta

            # Store coupling constraint reference to access dual multipliers LATER
            coupling_refs = sp.ext[:coupling_constraints] = Vector{JuMP.ConstraintRef}(undef, ar_process.parameters[t].dimension)
            coupling_refs[1] = coupling_ref_1
            coupling_refs[2] = coupling_ref_2 
        end
        
        SDDP.parameterize(sp, realizations) do ω
            JuMP.fix(inflow, ω[1])
            JuMP.fix(demand, ω[2])
        end
    end

    return model
end

function model_and_train()

    # MAIN MODEL AND RUN PARAMETERS    
    ###########################################################################################################
    applied_solver = LogLinearSDDP.AppliedSolver()
    problem_params = LogLinearSDDP.ProblemParams(3, 3)
    algo_params = LogLinearSDDP.AlgoParams(stopping_rules=[SDDP.IterationLimit(1)], simulation_regime=LogLinearSDDP.Simulation())

    # AUTOREGRESSIVE PROCESS (same definition for all three stages)
    ###########################################################################################################
    lag_order = 2
    dim = 2 # constant in this case

    # AR history
    # define also ξ₋₁, ξ₀ and ξ₁
    ar_history = Dict{Int64,Any}()
    # ar_history[-1] = [0.0, 5.0] #inflow, demand
    ar_history[0] = [4.0, 5.0]  #inflow, demand
    ar_history[1] = [4.0, 5.0]  #inflow, demand
    
    ar_parameters = Dict{Int64, LogLinearSDDP.AutoregressiveProcessStage}()
    
    # Stages 2 and 3
    intercept = [0.0, 0.0]
    coefficients = zeros(lag_order, dim, dim)
    coefficients[1,1,1] = 1/5
    coefficients[2,2,1] = 2/3
    coefficients[2,2,2] = 1/3   
    eta_1 = [-4.0, 3/4, 2.0]
    eta_2 = [-0.5, 0.0, 0.5]
    eta = vec(collect(Iterators.product(eta_1, eta_2)))

    # eta_vec = Vector{Vector{Float64}}(undef,length(eta))
    # for i in eachindex(eta)
    #     eta_vec[i] = collect(eta[i])
    # end
    # eta = eta_vec

    ar_parameters[2] = LogLinearSDDP.AutoregressiveProcessStage(dim, intercept, coefficients, eta)
    ar_parameters[3] = LogLinearSDDP.AutoregressiveProcessStage(dim, intercept, coefficients, eta)

    # All stages
    ar_process = LogLinearSDDP.AutoregressiveProcess(lag_order, ar_parameters, ar_history)

    # CREATE MODEL
    model = model_definition(ar_process, problem_params, algo_params)

    Infiltrator.@infiltrate

    # TRAIN MODEL
    LogLinearSDDP.train_loglinear(model, algo_params, problem_params, applied_solver, ar_process)

    Infiltrator.@infiltrate

    # SIMULATE MODEL
    LogLinearSDDP.simulate_loglinear(model, algo_params, problem_params, algo_params.simulation_regime)

end

model_and_train()