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

        JuMP.@variable(sp, 5 <= x <= 15, SDDP.State, initial_value = 10)
        JuMP.@variable(sp, g_t >= 0)
        JuMP.@variable(sp, g_h >= 0)
        JuMP.@variable(sp, s >= 0)
        JuMP.@variable(sp, demand)
        JuMP.@variable(sp, inflow)
        coupling_ref_1 = JuMP.@constraint(sp, water_bal, x.out - x.in + g_h + s == inflow)
        coupling_ref_2 = JuMP.@constraint(sp, demand_bal, g_h + g_t == demand)
        SDDP.@stageobjective(sp, s + t * g_t)

        # Store coupling constraint reference to access dual multipliers LATER
        coupling_refs = sp.ext[:coupling_constraints] = Vector{JuMP.ConstraintRef}(undef, autoregressive_data.ar_data[t].ar_dimension)
        coupling_refs[1] = coupling_ref_1
        coupling_refs[2] = coupling_ref_2 

        # Parameterize inflow and demand
        realizations = autoregressive_data.ar_data[t].ar_eta
        SDDP.parameterize(sp, realizations) do ω
            JuMP.fix(inflow, ω[1])
            JuMP.fix(demand, ω[2])
        end
    end

    return model
end

function model_and_train()

    # MAIN MODEL AND RUN PARAMETERS    
    applied_solver = LogLinearSDDP.AppliedSolver()
    problem_params = LogLinearSDDP.ProblemParams(3, 3)
    algo_params = LogLinearSDDP.AlgoParams()

    # AUTOREGRESSIVE PROCESS (same definition for all three stages)
    data = Vector{LogLinearSDDP.AutoregressiveDataStage}(undef, problem_params.number_of_stages)
    lag_order = 2
    dim = 2
    
    # Stage 1
    intercept = [0.0, 0.0]
    coefficients = zeros(lag_order, dim, dim)
    eta_1 = [4.0]
    eta_2 = [5.0]
    eta = vec(collect(Iterators.product(eta_1, eta_2)))
    
    data[1] = LogLinearSDDP.AutoregressiveDataStage(dim, intercept, coefficients, eta)

    # Stages 2 and 3
    intercept = [0.0, 0.0]
    coefficients = zeros(lag_order, dim, dim)
    coefficients[1,1,1] = 1/5
    coefficients[1,2,2] = 2/3
    coefficients[2,2,2] = 1/3
    eta_1 = [-4.0, 3/4, 2.0]
    eta_2 = [-0.5, 0.0, 0.5]
    eta = vec(collect(Iterators.product(eta_1, eta_2)))

    data[2] = LogLinearSDDP.AutoregressiveDataStage(dim, intercept, coefficients, eta)
    data[3] = LogLinearSDDP.AutoregressiveDataStage(dim, intercept, coefficients, eta)
    autoregressive_data = LogLinearSDDP.AutoregressiveData(lag_order, data)

    # define also ξ₀ and ξ₋₁
    user_process_state = Dict{Int64,Any}()
    user_process_state[-1] = [0.0, 5.0] #inflow, demand
    user_process_state[0] = [4.0, 5.0]  #inflow, demand

    # CREATE MODEL
    model = model_definition(autoregressive_data, problem_params, algo_params)

    Infiltrator.@infiltrate

    # TRAIN MODEL
    LogLinearSDDP.train_loglinear(model, algo_params, problem_params, applied_solver, autoregressive_data, user_process_state)

end

model_and_train()