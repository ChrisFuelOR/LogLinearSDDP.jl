"""
Function which assigns the correct solver to be used given that we
use GAMS.jl as a framework for solver communication.
"""
function set_solver!(
    subproblem::JuMP.Model,
    algo_params::LogLinearSDDP.AlgoParams,
    applied_solver::LogLinearSDDP.AppliedSolver,
    solver_approach::LogLinearSDDP.GAMS_Solver,
)
    numerical_focus = algo_params.numerical_focus ? 1 : 0
    solver = applied_solver.solver
    tolerance = applied_solver.solver_tol
    time_limit = applied_solver.solver_time
    # TODO: Time limit not implemented yet
    @warn("Time limit not implemented yet for GAMS approach.")

    # Set solver with tolerance
    ############################################################################
    if solver == "CPLEX"
        JuMP.set_optimizer(subproblem, JuMP.optimizer_with_attributes(
            () -> GAMS.Optimizer(),
            "Solver"=>solver,
            "optcr"=>tolerance,
            "numericalemphasis"=>numerical_focus)
        )
    elseif solver == "Gurobi"
        JuMP.set_optimizer(subproblem, JuMP.optimizer_with_attributes(
            () -> GAMS.Optimizer(),
            "Solver"=>solver,
            "optcr"=>tolerance,
            "NumericFocus"=>numerical_focus)
        )
    else
        JuMP.set_optimizer(subproblem, JuMP.optimizer_with_attributes(
            () -> GAMS.Optimizer(),
            "Solver"=>solver,
            "optcr"=>tolerance)
        )
    end

    # Numerical focus warning
    ############################################################################
    if numerical_focus == 1 && !(solver in ["Gurobi", "CPLEX"])
        @warn("Numerical focus only works with Gurobi or CPLEX.")
    end

    # Silence solver
    ############################################################################
    if algo_params.silent
        JuMP.set_silent(subproblem)
    else
        JuMP.unset_silent(subproblem)
    end

    return
end

"""
Function which assigns the correct solvers to be used given that we
use the solvers directly in JuMP.
"""
function set_solver!(
    subproblem::JuMP.Model,
    algo_params::LogLinearSDDP.AlgoParams,
    applied_solver::LogLinearSDDP.AppliedSolver,
    solver_approach::LogLinearSDDP.Direct_Solver,
)
    numerical_focus = algo_params.numerical_focus ? 1 : 0
    solver = applied_solver.solver
    tolerance = applied_solver.solver_tol
    time_limit = applied_solver.solver_time

    # Set solver with tolerance
    ############################################################################
    if solver in ["CPLEX", "BARON"]
        error("Solver can only be used with our GAMS license")
    elseif solver == "Gurobi"
        JuMP.set_optimizer(subproblem, JuMP.optimizer_with_attributes(
            () -> Gurobi.Optimizer(GURB_ENV[]),
            "MIPGap"=>tolerance,
            "TimeLimit"=>time_limit,
            "NumericFocus"=>numerical_focus);
            #bridge_constraints = false
            )
    elseif solver == "SCIP"
        JuMP.set_optimizer(subproblem, JuMP.optimizer_with_attributes(
            SCIP.Optimizer(),
            "display_verblevel"=>0,
            "limits_gap"=>tolerance)
            )
    else
        error("Solver has to set-up first.")
    end

    # Numerical focus warning
    ############################################################################
    if numerical_focus == 1 && !(solver in ["Gurobi", "CPLEX"])
        @warn("Numerical focus only works with Gurobi or CPLEX.")
    end

    # Silence solver
    ############################################################################
    if algo_params.silent
        JuMP.set_silent(subproblem)
    else
        JuMP.unset_silent(subproblem)
    end

    return
end


function set_solver_for_model(
    model::SDDP.PolicyGraph,
    algo_params::LogLinearSDDP.AlgoParams,
    applied_solver::LogLinearSDDP.AppliedSolver,
)

    for node in model.nodes
        set_solver!(node.subproblem, algo_params, applied_solver, algo_params.solver_approach)
    end
end

# JuMP.set_optimizer(subproblem, JuMP.optimizer_with_attributes(() -> GAMS.Optimizer()))