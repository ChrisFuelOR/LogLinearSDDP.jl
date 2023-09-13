""" 
MAIN DOCUMENTATION [SHOULD LATER BE MOVED TO A PROPER DOC FILE]
###########################################################################

This package is intended for using a special version of SDDP for multistage
stochastic linear problems with log-linear autoregressive uncertainty in the
right-hand side.

For simplicity, we assume that this uncertainty in the RHS is the only
uncertainty in the problem, even though in theory also stagewise independent
uncertainty in other problem elements is allowed.

The definition of the autoregressive process has to be provided by the user
by means of the autoregressive_data_stage struct.

For now, this version of SDDP is restricted to problems satisfying the 
following properties
> finitely many stages
> linear constraints and objective
> continuous decision variables
> log-linear autoregressive uncertainty in the RHS
> finite support of the uncertainty
> uncertainty is exogeneous
> expectations considered in the objective (no other risk measures)
> no usage of believe or objective states

Additionally, this version of SDDP is restricted to using a SINGLE_CUT approach.
A MULTI_CUT approach is not supported yet, as it affects the already complicated
cut formulas.

Importantly, this package does not require the user to model an explicit state 
expansion for the given problem to take the history of the AR process into account.
Instead, specific nonlinear cut formulas are used to adapt the cut intercept to 
a scenario at hand.
"""

################################################################################
# DEFINING CUTS
################################################################################

mutable struct Cut
    coefficients::Dict{Symbol,Float64} # gradient
    intercept::Float64 # for the construction scenario (only for checks)
    intercept_factors::Array{Float64,2}
    trial_state::Dict{Symbol,Float64}
    constraint_ref::Union{Nothing,JuMP.ConstraintRef}
    cut_intercept_variable::Union{Nothing,JuMP.VariableRef}
    obj_y::Union{Nothing,NTuple{N,Float64} where {N}}
    belief_y::Union{Nothing,Dict{T,Float64} where {T}}
    non_dominated_count::Int
    iteration::Int64
end

"""
The argument "gradient" is the cut slope vector β.

The argument "intercept_factors" is not a scalar intercept as in standard SDDP, 
    but a matrix of intercept factors for each τ=t,...  ,T and each component ℓ of the autoregressive process. 
    These factors are used to compute (adapt) the intercept of a cut to the scenario at hand.
    This is done by fixing the corresponding cut_intercept_variable.

The third and fourth arguments are the optimal dual multipliers from the cut generation process.
    Note that in this specific case we cannot compute the cut coefficients using copy constraints, but we have
    to explicitly get the dual variables to all existing cuts for our formulas.

The argument "trial state" is the point (incumbent) where the cut is constructed.

The argument "cut_constraint" refers to the cut constraint in the JuMP model.

The argument "cut_intercept_variable" refers to the artificial variable in the JuMP model which is fixed to the
    cut intercept for the specific scenario at hand.

The argument "non_dominated_count" is required for cut selection purposes.

The argument "iteration" stores the iteration number in which the cut was constructed. This is just for logging
    and analyses.
"""


################################################################################
# SIMULATION
################################################################################
# Sampling schemes (similar to the ones in SDDP.jl)
# abstract type AbstractSamplingScheme end

# mutable struct InSampleMonteCarlo <: AbstractSamplingScheme end

# mutable struct OutOfSampleMonteCarlo <: AbstractSamplingScheme
#     number_of_realizations :: Int
#     simulation_seed :: Int

#     function OutOfSampleMonteCarlo(;
#         number_of_realizations = 10,
#         simulation_seed = 121212,
#     )
#         return new(simulation_seed)
#     end
# end
#
#mutable struct HistoricalSample <: AbstractSamplingScheme end

# Simulation regimes
abstract type AbstractSimulationRegime end

mutable struct Simulation <: AbstractSimulationRegime
    sampling_scheme :: SDDP.AbstractSamplingScheme
    number_of_replications :: Int

    function Simulation(;
        sampling_scheme = DynamicSDDiP.InSampleMonteCarlo,
        number_of_replications = 1000,
    )
        return new(sampling_scheme, number_of_replications)
    end
end

mutable struct NoSimulation <: AbstractSimulationRegime end

"""
Simulation means that after training the model we perform a simulation with
    number_of_replications. This can be either an in-sample simulation
    (SDDP.InSampleMonteCarlo) or an out-of-sample simulation
    (SDDP.OutOfSampleMonteCarlo). In the latter case, we have to provide a
    method to generate new scenario trees. Different sampling schemes from
    SDDP.jl such as HistoricalSampling or PSRSampling are not supported yet.
NoSimulation means that we do not perform a simulation after training the model,
    either because we do not want to or because we solve a determinist model.
Default is NoSimulation.
"""

################################################################################
# DEFINING STRUCT FOR CONFIGURATION OF TEST PROBLEM PARAMETERS
################################################################################
"""
Stores some parameters of the test problem that is solved.
This is mainly used for logging purposes.

    number_of_stages:       stage number of the multistage problem
    numer_of_realizations:  number of stagewise independent noise realization of ηₜ
    number_of_lags:         number of lags of the autoregressive noise process
"""

struct ProblemParams
    number_of_stages::Int
    number_of_realizations::Int
    tree_seed::Union{Nothing,Int}

    function ProblemParams(
        number_of_stages,
        number_of_realizations;
        tree_seed = nothing,
    )
        return new(
            number_of_stages,
            number_of_realizations,
            tree_seed
        )
    end
end

################################################################################
# SOLVER HANDLING
################################################################################
struct AppliedSolver
    solver :: Any
    solver_tol :: Float64
    solver_time :: Int64

    function AppliedSolver(;
        solver = "Gurobi",
        solver_tol = 1e-4,
        solver_time = 300,
        )
        return new(
            solver,
            solver_tol,
            solver_time,
            )
    end
end

abstract type AbstractSolverApproach end

mutable struct GAMS_Solver <: AbstractSolverApproach end
mutable struct Direct_Solver <: AbstractSolverApproach end

################################################################################
# DEFINING STRUCT FOR CONFIGURATION OF ALGORITHM PARAMETERS
################################################################################
"""
Note that the parameters from risk_measure to refine_at_similar_nodes are
basic SDDP parameters which are required as we are using some functionality
from the package SDDP.jl. They should not be changed, though, as for different
choices the DynamicSDDiP algorithm will not work.

Note that run_numerical_stability_report is not updated to the modified version
of SDDP yet.
"""

mutable struct AlgoParams
    stopping_rules::Vector{SDDP.AbstractStoppingRule}
    simulation_regime::LogLinearSDDP.AbstractSimulationRegime
    ############################################################################
    print_level::Int
    log_frequency::Int
    log_file::String
    run_numerical_stability_report::Bool
    numerical_focus::Bool
    silent::Bool
    infiltrate_state::Symbol
    seed::Union{Nothing,Int}
    run_description::String
    solver_approach::Union{LogLinearSDDP.GAMS_Solver,LogLinearSDDP.Direct_Solver} # Direct_Solver ≠ Direct mode for solve

    function AlgoParams(;
        stopping_rules = [SDDP.IterationLimit(100)],
        simulation_regime = LogLinearSDDP.NoSimulation(), #TODO
        print_level = 2,
        log_frequency = 1,
        log_file = "LogLinearSDDP.log",
        run_numerical_stability_report = false,
        numerical_focus = false,
        silent = true,
        infiltrate_state = :none,
        seed = nothing,
        run_description = "",
        solver_approach = LogLinearSDDP.Direct_Solver(), #TODO
    )
        return new(
            stopping_rules,
            simulation_regime,
            print_level,
            log_frequency,
            log_file,
            run_numerical_stability_report,
            numerical_focus,
            silent,
            infiltrate_state,
            seed,
            run_description,
            solver_approach
        )
    end
end

################################################################################
# DEFINING STRUCT FOR AUTOREGRESSIVE DATA
################################################################################
"""
Struct containing the data for the log-linear autoregressive process for a given stage.
Note that the process is defined componentwise for each component ℓ.

ar_intercept:       Vector containing the intercepts of the log-linear AR process;
                    one-dimensional with component ℓ;
                    denoted by γ in the paper
ar_coefficients:    Array containing the coefficients of the log-linear AR process;
                    three-dimensional with lag k and components ℓ, m;
                    denoted by ϕ in the paper
ar_eta:             Array containing the stagewise independent realizations of the log-linear AR process;
                    two-dimensional with component ℓ and possible realizations;
                    denoted by η in the paper
ar_dimension:       Int64 which defines the dimension of the random process at the current stage;
                    denoted by L in the paper

The size of ar_eta should match number_of_realizations defined in ProblemParams, so ProblemParams
should be defined in advance.
"""

struct AutoregressiveDataStage
    ar_intercept::Vector{Float64}
    ar_coefficients::Array{Float64,3}
    ar_eta::Array{Float64,2}
    ar_dimension::Int64
end


"""
Struct containing the data for the log-linear autoregressive process for all stages.

Note that we assume that the lag order is the same for all stages and components because otherwise
the cut formulas become way more sophisticated (see paper). In practice, different components and stages
may require different lag orders, for instance in SPAR models. If a stage-component combination requires less
lags than globally defined, we can set the ar_coefficients corresponding to excessive lags to 1, so that 
they do not have any effect.

ar_lag_order:      Int64 which defines the lag order of the random process (for each component and stage);
                   denoted by p in the paper
ar_data:           Vector containing the stage-specific data of the AR process;
                   one-dimensional with component t
"""

struct AutoregressiveData
    ar_lag_order::Int64
    ar_data::Vector{AutoregressiveDataStage}
end




