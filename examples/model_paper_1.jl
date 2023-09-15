import SDDP
import Gurobi
import JuMP
import LogLinearSDDP
import Infiltrator
import Revise


function model_definition(autoregressive_data::LogLinearSDDP.AutoregressiveData, problem_params::LogLinearSDDP.ProblemParams, algo_params::LogLinearSDDP.AlgoParams)

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

        elseif t == 2
            JuMP.@variable(sp, y >= 0)
            JuMP.@constraint(sp, y <= 6)
            SDDP.@stageobjective(sp, 2*y + x.out)
            coupling_ref = JuMP.@constraint(sp, y - x.out == ξ - x.in)

            # Store coupling constraint reference to access dual multipliers LATER
            coupling_refs = sp.ext[:coupling_constraints] = Vector{JuMP.ConstraintRef}(undef, autoregressive_data.ar_data[t].ar_dimension)
            coupling_refs[1] = coupling_ref

        else
            JuMP.@variable(sp, y >= 0)
            JuMP.@constraint(sp, y <= 6)
            SDDP.@stageobjective(sp, y + x.out)
            coupling_ref = JuMP.@constraint(sp, y - x.out == ξ - x.in)

            # Store coupling constraint reference to access dual multipliers LATER
            coupling_refs = sp.ext[:coupling_constraints] = Vector{JuMP.ConstraintRef}(undef, autoregressive_data.ar_data[t].ar_dimension)
            coupling_refs[1] = coupling_ref
            
        end

        # Parameterize the demand
        SDDP.parameterize(sp, vec(autoregressive_data.ar_data[t].ar_eta)) do ω
            JuMP.fix(ξ, ω[1])
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
    data = Vector{LogLinearSDDP.AutoregressiveDataStage}(undef, 3)
    lag_order = 1

    dim_1 = 1
    intercept_1 = zeros(dim_1)
    coefficients_1 = ones(lag_order, dim_1, dim_1)  # shouldn't this be zeros?
    eta_1 = zeros(dim_1, 1)
    data[1] = LogLinearSDDP.AutoregressiveDataStage(intercept_1, coefficients_1, eta_1, dim_1)

    dim_2 = 1
    intercept_2 = zeros(dim_1)
    coefficients_2 = 1/4 * ones(lag_order, dim_1, dim_1) # shouldn't this be zeros?
    eta_2 = [-1 1]
    data[2] = LogLinearSDDP.AutoregressiveDataStage(intercept_2, coefficients_2, eta_2, dim_2)

    dim_3 = 1
    intercept_3 = zeros(dim_1)
    coefficients_3 = 1/4 * ones(lag_order, dim_1, dim_1) # shouldn't this be zeros?
    eta_3 = [-1 1]
    data[3] = LogLinearSDDP.AutoregressiveDataStage(intercept_3, coefficients_3, eta_3, dim_3)

    autoregressive_data = LogLinearSDDP.AutoregressiveData(lag_order, data)

    # define also ξ₀
    user_process_state = Dict{Int64,Vector{Float64}}()
    user_process_state[0] = [3.0]

    # CREATE MODEL
    model = model_definition(autoregressive_data, problem_params, algo_params)

    Infiltrator.@infiltrate

    # TRAIN MODEL
    LogLinearSDDP.train_loglinear(model, algo_params, problem_params, applied_solver, autoregressive_data, user_process_state)

end

model_and_train()