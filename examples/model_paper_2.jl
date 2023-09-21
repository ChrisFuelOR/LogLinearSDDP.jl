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

        JuMP.@variable(sp, x >= 0, SDDP.State, initial_value = 0)
        JuMP.@constraint(sp, ubound, x.out <= 6)
        JuMP.@variable(sp, ξ)

        if t == 1
            JuMP.@variable(sp, w)
            SDDP.@stageobjective(sp, w)
            JuMP.@constraint(sp, -w <= x.out - 4)
            JuMP.@constraint(sp, w >= x.out - 4)

            # Parameterize the RHS
            # realizations = [ar_process.history[1]]
            SDDP.parameterize(sp, [[0.0]]) do ω
                JuMP.fix(ξ, ω[1])
            end

        elseif t == 2
            JuMP.@variable(sp, y >= 0)
            JuMP.@constraint(sp, y <= 6)
            SDDP.@stageobjective(sp, 2*y + x.out)
            coupling_ref = JuMP.@constraint(sp, y - x.out == ξ - x.in)

            # Store coupling constraint reference to access dual multipliers LATER
            coupling_refs = sp.ext[:coupling_constraints] = Vector{JuMP.ConstraintRef}(undef, ar_process.parameters[t].dimension)
            coupling_refs[1] = coupling_ref

            realizations = ar_process.parameters[t].eta
            # Parameterize the RHS
            SDDP.parameterize(sp, realizations) do ω
                JuMP.fix(ξ, ω[1])
            end

        else
            JuMP.@variable(sp, y >= 0)
            JuMP.@constraint(sp, y <= 6)
            SDDP.@stageobjective(sp, y + x.out)
            coupling_ref = JuMP.@constraint(sp, y - x.out == ξ - x.in)

            # Store coupling constraint reference to access dual multipliers LATER
            coupling_refs = sp.ext[:coupling_constraints] = Vector{JuMP.ConstraintRef}(undef, ar_process.parameters[t].dimension)
            coupling_refs[1] = coupling_ref
            
            realizations = ar_process.parameters[t].eta
            # Parameterize the RHS
            SDDP.parameterize(sp, realizations) do ω
                JuMP.fix(ξ, ω[1])
            end
        end

    end

    return model
end

function model_and_train()

    # MAIN MODEL AND RUN PARAMETERS    
    applied_solver = LogLinearSDDP.AppliedSolver()
    problem_params = LogLinearSDDP.ProblemParams(3, 2)
    algo_params = LogLinearSDDP.AlgoParams()

    # AUTOREGRESSIVE PROCESS
    lag_order = 1
    dim = 1 # constant in this case

    # AR history
    # define also ξ₋₁, ξ₀ and ξ₁
    ar_history = Dict{Int64,Any}()
    #ar_history[0] = [3.0]
    ar_history[1] = [3.0] 

    ar_parameters  = Dict{Int64, LogLinearSDDP.AutoregressiveProcessStage}()

    # Stage 2
    intercept = zeros(dim)
    coefficients = 1/4 * ones(lag_order, dim, dim)
    eta = [-1.0, 1.0]
    ar_parameters[2] = LogLinearSDDP.AutoregressiveProcessStage(dim, intercept, coefficients, eta)

    # Stage 3
    intercept = zeros(dim)
    coefficients = 3/2 * ones(lag_order, dim, dim)
    eta = [-2.0, -1.0]
    ar_parameters[3] = LogLinearSDDP.AutoregressiveProcessStage(dim, intercept, coefficients, eta)

    # All stages
    ar_process = LogLinearSDDP.AutoregressiveProcess(lag_order, ar_parameters, ar_history)

    # CREATE MODEL
    model = model_definition(ar_process, problem_params, algo_params)

    Infiltrator.@infiltrate

    # TRAIN MODEL
    LogLinearSDDP.train_loglinear(model, algo_params, problem_params, applied_solver, ar_process)

end

model_and_train()