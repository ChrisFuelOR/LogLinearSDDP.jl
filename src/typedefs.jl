# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Copyright (c) 2026 Christian Fuellner <christian.fuellner@kit.edu>
################################################################################

################################################################################
# DEFINING CUTS
################################################################################
"""
Struct for storing information related to a cut.

The argument "coefficients" is the cut slope vector β. It can be computed as the dual vector corresponding
    to copy constraints.

The argument "deterministic_intercept" is used to account for the contribution of deterministic constraints
    to the intercept. Handling them as coupling constraints is unnecessary, e.g. from a memory perspective, 
    but not taking them into account leads to a wrong intercept.

The argument "stochastic_intercept_tight" is the value of the full intercept at the state (including scenario) 
    of construction, i.e. where the cut is tight. This is used for checks and to compute other values.
    It may also be used for cut selection purposes.

The argument "intercept_factors" is not a scalar intercept as in standard SDDP, 
    but a matrix of intercept factors for each τ=t,... ,T and each component ℓ of the autoregressive process. 
    These factors are used to compute (adapt) the intercept of a cut to the scenario at hand.
    This is done by fixing the corresponding cut_intercept_variable.

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
    deterministic_intercept::Float64
    stochastic_intercept_tight::Float64
    intercept_factors::Array{Float64,2}
    trial_state::Dict{Symbol,Float64}
    constraint_ref::Union{Nothing,JuMP.ConstraintRef}
    cut_intercept_variable::Union{Nothing,JuMP.VariableRef}
    non_dominated_count::Int64
    iteration::Int64
end


################################################################################
# SIMULATION
################################################################################
"""
Different simulation regimes.

Simulation means that after training the model, we perform a simulation with
    number_of_replications. This can be either an in-sample simulation
    (SDDP.InSampleMonteCarlo) or an out-of-sample simulation
    (SDDP.OutOfSampleMonteCarlo). In the latter case, we have to provide a
    method to generate new scenario trees. Different sampling schemes from
    SDDP.jl such as HistoricalSampling or PSRSampling are not supported yet.
NoSimulation means that we do not perform a simulation after training the model,
    either because we do not want to or because we solve a determinist model.
Default is NoSimulation.

The simulation_seed determines which scenarios are sampled. That is, for InSampleMonteCarlo
it determines the scenario path chosen on the in-sample realizations. For OutOfSampleMonteCarlo
it determines which scenarios are drawn from the underlying distribution. 
"""

# Simulation regimes
abstract type AbstractSimulationRegime end

mutable struct Simulation <: AbstractSimulationRegime
    sampling_scheme :: SDDP.AbstractSamplingScheme
    number_of_replications :: Int64
    simulation_seed :: Int64

    function Simulation(;
        sampling_scheme = SDDP.InSampleMonteCarlo(),
        number_of_replications = 1000,
        simulation_seed = 11111,
    )
        return new(sampling_scheme, number_of_replications, simulation_seed)
    end
end

mutable struct NoSimulation <: AbstractSimulationRegime end


################################################################################
# DEFINING STRUCT FOR CONFIGURATION OF TEST PROBLEM PARAMETERS
################################################################################
"""
Stores some parameters of the test problem that is solved.
This is mainly used for logging purposes.

    number_of_stages:       number of stages of the multistage problem
    numer_of_realizations:  number of stagewise independent noise realization of ηₜ
    tree_seed:              seed for the scenario tree generation
"""

struct ProblemParams
    number_of_stages::Int64
    number_of_realizations::Int64
    tree_seed::Union{Nothing,Int}
    gurobi_coupling_index_start::Union{Nothing,Int}
    gurobi_cut_index_start::Union{Nothing,Int}
    gurobi_fix_start::Union{Nothing,Int}

    function ProblemParams(
        number_of_stages,
        number_of_realizations;
        tree_seed = nothing,
        gurobi_coupling_index_start = nothing,
        gurobi_cut_index_start = nothing,
        gurobi_fix_start = nothing,
    )
        return new(
            number_of_stages,
            number_of_realizations,
            tree_seed,
            gurobi_coupling_index_start,
            gurobi_cut_index_start,
            gurobi_fix_start,
        )
    end
end

################################################################################
# DEFINING STRUCT FOR CONFIGURATION OF ALGORITHM PARAMETERS
################################################################################
"""
Struct containing user-defined parameters for the algorithm.

Note that run_numerical_stability_report is not updated to the modified version
of SDDP yet.
"""

mutable struct AlgoParams
    stopping_rules::Vector{SDDP.AbstractStoppingRule}
    simulation_regime::LogLinearSDDP.AbstractSimulationRegime
    cut_selection::Bool
    ############################################################################
    print_level::Int64
    log_frequency::Int64
    log_file::String
    run_numerical_stability_report::Bool
    numerical_focus::Bool
    silent::Bool
    forward_pass_seed::Union{Nothing,Int}
    run_description::String
    model_approach::Symbol

    function AlgoParams(;
        stopping_rules = [SDDP.IterationLimit(100)],
        simulation_regime = LogLinearSDDP.NoSimulation(), #TODO
        cut_selection = false,
        print_level = 2,
        log_frequency = 1,
        log_file = "LogLinearSDDP.log",
        run_numerical_stability_report = false,
        numerical_focus = false,
        silent = true,
        forward_pass_seed = nothing,
        run_description = "",
        model_approach = :fitted_model,
    )
        return new(
            stopping_rules,
            simulation_regime,
            cut_selection,
            print_level,
            log_frequency,
            log_file,
            run_numerical_stability_report,
            numerical_focus,
            silent,
            forward_pass_seed,
            run_description,
            model_approach,
        )
    end
end

################################################################################
# DEFINING STRUCT FOR AUTOREGRESSIVE DATA
################################################################################
"""
Struct containing the parameters for the log-linear autoregressive process for a given stage.
Note that the process is defined componentwise for each component ℓ.

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
    intercept::Vector{Float64}
    coefficients::Array{Float64,3}
    psi::Vector{Float64}
    eta::Vector{Any}
    probabilities::Vector{Float64}

    function AutoregressiveProcessStage(
        intercept,
        coefficients,
        eta;
        psi = ones(length(intercept)),
        probabilities = fill(1 / length(eta), length(eta)),
    )
        return new(
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

1.) Note that we assume that the lag order is the same for all stages and components. Otherwise
the cut formulas become way more sophisticated (see paper). In practice, different components and stages
may require different lag orders, for instance in SPAR models. If a stage-component combination requires less
lags than globally defined, we can set the ar_coefficients corresponding to excessive lags to 1, so that 
they do not have any effect.

2.) Note that - in contrast to the paper - we also assume that the dimension of the process is the same 
for all stages. This allows us to accelerate nested loops with tools that do not allow for indices to be 
dependent on each other. This is a very natural assumption in practice. For instance, in our hydrothermal
scheduling example, we have the same number of reservoirs for each stage.

3.) Note that we assume the first stage data to be deterministic. Therefore, it should be included in
ar_history instead of ar_data.

dimension:      Int64 which defines the dimension of the random process;
                denoted by L in the paper
lag_order:      Int64 which defines the lag order of the random process (for each component and stage);
                denoted by p in the paper
parameters:     Dict containing the stage-specific data of the AR process. 
                The key is the stage and the value is the actual data struct;
                one-dimensional with component t
history:        Dict containing the historic values of the AR process (including stage 1).
                The key is the stage and the value is a vector of index ℓ.
simplified:     If true, there are no dependencies between different components (i.e. spatial dependencies).
                Therefore, some computations can be simplified.
"""

struct AutoregressiveProcess
    dimension::Int64
    lag_order::Int64
    parameters::Dict{Int64,AutoregressiveProcessStage}
    history::Dict{Int64,Vector{Float64}}
    simplified::Bool
end

