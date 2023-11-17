# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Copyright (c) 2023 Christian Fuellner <christian.fuellner@kit.edu>
################################################################################

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
"""
Struct for storing information related to a cut.

The argument "gradient" is the cut slope vector β. It can be computed as the dual vector corresponding
    to copy constraints.

The argument "intercept_tight" is the value of the full intercept at the state of construction,
    i.e. where the cut is tight. This is used for checks and to compute other values.

The argument "intercept_factors" is not a scalar intercept as in standard SDDP, 
    but a matrix of intercept factors for each τ=t,...  ,T and each component ℓ of the autoregressive process. 
    These factors are used to compute (adapt) the intercept of a cut to the scenario at hand.
    This is done by fixing the corresponding cut_intercept_variable.

The argument "deterministic_intercept" is used to account for the contribution of deterministic constraints
    to the intercept. Handling them as coupling constraints is unnecessary, e.g. from a memory perspective, 
    but not taking them into account leads to a wrong intercept.

The argument "trial state" is the point (incumbent) where the cut is constructed.

The argument "cut_constraint" refers to the cut constraint in the JuMP model.

The argument "cut_intercept_variable" refers to the artificial variable in the JuMP model which is fixed to the
    cut intercept for the specific scenario at hand.

The argument "non_dominated_count" is required for cut selection purposes.

The argument "iteration" stores the iteration number in which the cut was constructed. This is just for logging
    and analyses.
"""

mutable struct Cut
    coefficients::Dict{Symbol,Float64}
    intercept_factors::Array{Float64,2}
    stochastic_intercept_tight::Float64
    deterministic_intercept::Float64
    trial_state::Dict{Symbol,Float64}
    constraint_ref::Union{Nothing,JuMP.ConstraintRef}
    cut_intercept_variable::Union{Nothing,JuMP.VariableRef}
    obj_y::Union{Nothing,NTuple{N,Float64} where {N}}
    belief_y::Union{Nothing,Dict{T,Float64} where {T}}
    non_dominated_count::Int64
    iteration::Int64
end


################################################################################
# SIMULATION
################################################################################
"""
Different simulation regimes.

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

# Sampling schemes (similar to the ones in SDDP.jl)
# abstract type AbstractSamplingScheme end

# mutable struct InSampleMonteCarlo <: AbstractSamplingScheme end

# mutable struct OutOfSampleMonteCarlo <: SDDP.AbstractSamplingScheme
#     number_of_realizations :: Int
#     simulation_seed :: Int

#     function OutOfSampleMonteCarlo(;
#         number_of_realizations = 10,
#         simulation_seed = 121212,
#     )
#         return new(simulation_seed)
#     end
# end

# mutable struct HistoricalSample <: SDDP.AbstractSamplingScheme end

# Simulation regimes
abstract type AbstractSimulationRegime end

mutable struct Simulation <: AbstractSimulationRegime
    sampling_scheme :: SDDP.AbstractSamplingScheme
    number_of_replications :: Int

    function Simulation(;
        sampling_scheme = SDDP.InSampleMonteCarlo(),
        number_of_replications = 1000,
    )
        return new(sampling_scheme, number_of_replications)
    end
end

mutable struct NoSimulation <: AbstractSimulationRegime end


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
    number_of_stages::Int64
    number_of_realizations::Int64
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
    print_level::Int64
    log_frequency::Int64
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
Struct containing the parameters for the log-linear autoregressive process for a given stage.
Note that the process is defined componentwise for each component ℓ.

dimension:      Int64 which defines the dimension of the random process at the current stage;
                denoted by L in the paper
intercept:      Vector containing the intercepts of the log-linear AR process;
                one-dimensional with component ℓ;
                denoted by γ in the paper
coefficients:   Array containing the coefficients of the log-linear AR process;
                three-dimensional with components ℓ, m and lag k;
                denoted by ϕ in the paper
psi:            Vector containing the pre-factor for eta in the log-linear AR process;
                one-dimensional with component ℓ; 
                denoted by ψ in the paper               
eta:            Vector containing the stagewise independent realizations of the log-linear AR process;
                each element is an object containing different components ℓ (e.g. a vector or a tuple);
                denoted by η in the paper;
                note that this is merely stored for logging purposes if required
probabilities:  Probabilities related to eta (optional);
                note that this is merely stored for logging purposes if required

The size of eta should match number_of_realizations defined in ProblemParams, so ProblemParams
should be defined in advance.
"""

struct AutoregressiveProcessStage
    dimension::Int64
    intercept::Vector{Float64}
    coefficients::Array{Float64,3}
    psi::Vector{Any}
    eta::Vector{Any}
    probabilities::Vector{Float64}

    function AutoregressiveProcessStage(
        dimension,
        intercept,
        coefficients,
        eta;
        psi = ones(length(intercept)),
        probabilities = fill(1 / length(eta), length(eta)),
    )
        return new(
            dimension,
            intercept,
            coefficients,
            psi,
            eta,
            probabilities,
        )
    end
end

"""
Struct containing the data for the log-linear autoregressive process for all stages.

1.) Note that we assume that the lag order is the same for all stages and components because otherwise
the cut formulas become way more sophisticated (see paper). In practice, different components and stages
may require different lag orders, for instance in SPAR models. If a stage-component combination requires less
lags than globally defined, we can set the ar_coefficients corresponding to excessive lags to 1, so that 
they do not have any effect.

2.) Note that we assume the first stage data to be deterministic. Therefore, it should be included in
ar_history instead of ar_data.

lag_order:      Int64 which defines the lag order of the random process (for each component and stage);
                denoted by p in the paper
parameters:     Dict containing the stage-specific data of the AR process. 
                The key is the stage and the value is the actual data struct;
                one-dimensional with component t
history:        Dict containing the historic values of the AR process (including stage 1).
                The key is the stage and the value is a vector of index ℓ.
"""

struct AutoregressiveProcess
    lag_order::Int64
    parameters::Dict{Int64,AutoregressiveProcessStage}
    history::Dict{Int64,Vector{Float64}}
end




""" 
MAIN DOCUMENTATION FOR MODEL DEFINITION [SHOULD LATER BE MOVED TO A PROPER DOC FILE]
###########################################################################

The models should be set up in the following way:

> Define important parameters
 >> AppliedSolver:  defines which solver should be used with which configuration within the algorithm
 >> ProblemParams:  specifies the number of stages and the number of realizations per stage
 >> AlgoParams:     specifies the configuration of the algorithm (note that only a few of the SDDP.jl parameters
                    can be changed here; see above)

> Define the AR process
 >> The AR history should be defined as a dictionary with the time stages as indices and the vectors/tuples of
    historic values as values.
 >> Importantly, the deterministic first stage should be defined as part of the AR history.
 >> The constant lag order should be defined.
 >> For all following stages, intercepts, coefficients and dimensions of the AR process can be defined and stored
    in a AutoregressiveProcessStage struct object.
 >> Importantly, also the stagewise independent realizations are stored there, even if they are only used in
    the model definition (or for logging) later on, but not required in the actual algorithm.
 >> The stagewise independent realizations should be stored in a vector of vectors or a vector of tuples
    (one vector with length equal to the number of different realizations, and each component containing a vector
    or tuple of the multi-dimensional realization). Even a named tuple is possible. Hence, all the suggested
    variants from the SDDP.jl documentation for multi-dimensional noise can be used.                    
 >> The AutoregressiveProcessStage struct objects are stored in an AutoregressiveProcess struct object together
    with the lag order and the AR history.

> Define the policy graph and the subproblems as in SDDP.jl.
 >> You should define a lower bound for minimization problems (upper bound for maximization problems).
 >> Importantly, for the cut generation, the coupling constraint containing the .in-part of the 
    state variables have to be stored in sp.ext[:coupling_constraints] in order to be accessed within
    the cut generation process.
> The realizations (and probabilities) that are given to the parameterize-function in the model
    definition only include the stagewise independent part of the uncertainty.
 >> The first stage data is considered deterministic, so there should be only one realization (for each
    dimension of the uncertainty). This can always be set to 0.
 >> For the following stages t, the realizations can be set to ar_process.parameters[t].eta.

> Before the iteration loop is started, the constant cut exponents Θ are computed once. Note that, compared to the 
  paper, the indexing is a bit different. Precisely, Θ(t,τ,ℓ,m,k) with k a stage translates to cut_exponents_stage[t][τ,ℓ,m,κ]
  with κ the lag and κ = t-k.

> If not enough history is provided by the user, based on the maximum dimension and the lag order some default history will be 
  created by the algorithm.

> In each iteration, based on the sampled stagewise independent noise (sampling is done using the SDDP.jl functionality),
  the AR process definition is used to compute the stagewise dependent noise and to parameterize the uncertain data.
  >> In order for this to work, the required lagged values are stored in node.ext[:process_state], which is a dict with
     stages as keys and the corresponding values of the uncertain data as values.
  >> After the new realization is computed, the process state for the following node is updated.
  >> In addition, during the parameterization process all existing nonlinear cuts are evaluated for the considered scenario.

"""