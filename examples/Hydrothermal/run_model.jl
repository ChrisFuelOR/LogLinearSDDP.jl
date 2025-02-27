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
import DataFrames
import DataFramesMeta
import CSV
import Random
import Dates

include("hydrothermal_model.jl")
include("set_up_ar_process.jl")
include("simulation.jl")
include("cross_simulation_loglinear.jl")


function run_model(forward_pass_seed::Int, model_approach::Symbol, model_approaches_alternative::Vector{Symbol}, model_directories_lin::Vector{String})

    # MAIN MODEL AND RUN PARAMETERS    
    ###########################################################################################################
    number_of_stages = 120
    number_of_realizations = 100
    simulation_replications = 2000
    ###########################################################################################################
    file_identifier = "Run_" * string(model_approach) * "_" * string(forward_pass_seed)
    file_path = "C:/Users/cg4102/Documents/julia_logs/Cut-sharing/" * file_identifier
    log_file = file_path * "LogLinearSDDP.log"
    run_description = ""
    ###########################################################################################################
    problem_params = LogLinearSDDP.ProblemParams(number_of_stages, number_of_realizations)
    simulation_regime = LogLinearSDDP.Simulation(sampling_scheme = SDDP.InSampleMonteCarlo(), number_of_replications = simulation_replications)
    algo_params = LogLinearSDDP.AlgoParams(stopping_rules = [SDDP.IterationLimit(10)], forward_pass_seed = forward_pass_seed, simulation_regime = simulation_regime, 
        log_file = log_file, silent = true, model_approach = model_approach, run_description = run_description)

    # CREATE AND RUN MODEL
    ###########################################################################################################
    f = open(file_path * "inflows.txt", "w")
    ar_process = set_up_ar_process_loglinear(number_of_stages, number_of_realizations, String(model_approach), String(model_approach))
    model = model_definition(ar_process, problem_params, algo_params, f)

    # Train model
    Random.seed!(algo_params.forward_pass_seed)
    LogLinearSDDP.train_loglinear(model, algo_params, problem_params, ar_process)
    close(f)

    # SIMULATION USING THE LINEARIZED PROCESS
    ###########################################################################################################
    model.ext[:simulation_attributes] = [:level, :inflow, :spillage, :gen, :exchange, :deficit_part, :hydro_gen]
    
    # In-sample simulation
    simulation_results = LogLinearSDDP.simulate_loglinear(model, algo_params, ar_process, String(model_approach), algo_params.simulation_regime)
    extended_simulation_analysis(simulation_results, file_path, String(model_approach), "_in_sample")

    #----------------------------------------------------------------------------------------------------------
    # Out-of-sample simulation
    Random.seed!(12345+algo_params.forward_pass_seed)
    sampling_scheme_loglinear = SDDP.OutOfSampleMonteCarlo(model, use_insample_transition = true) do stage
        return get_out_of_sample_realizations_loglinear(number_of_realizations, stage, String(model_approach))
    end
    simulation_loglinear = LogLinearSDDP.Simulation(sampling_scheme = sampling_scheme_loglinear, number_of_replications = simulation_replications)
    simulation_results = LogLinearSDDP.simulate_loglinear(model, algo_params, ar_process, String(model_approach), simulation_loglinear)
    extended_simulation_analysis(simulation_results, file_path, String(model_approach), String(model_approach))

    #----------------------------------------------------------------------------------------------------------
    # Out-of-sample simulation (alternative log-linear model)
    for model_approach_alt in model_approaches_alternative
        loglin_ar_process = set_up_ar_process_loglinear(number_of_stages, number_of_realizations, String(model_approach_alt), "bic_model")
        LogLinearSDDP.initialize_process_state(model, loglin_ar_process)

        Random.seed!(12345+algo_params.forward_pass_seed)  
        sampling_scheme_loglinear = SDDP.OutOfSampleMonteCarlo(model, use_insample_transition = true) do stage
            return get_out_of_sample_realizations_loglinear(number_of_realizations, stage, String(model_approach_alt))
        end
        simulation_loglinear = LogLinearSDDP.Simulation(sampling_scheme = sampling_scheme_loglinear, number_of_replications = simulation_replications)
        simulation_results = LogLinearSDDP.simulate_loglinear(model, algo_params, loglin_ar_process, String(model_approach_alt), simulation_loglinear)
        extended_simulation_analysis(simulation_results, file_path, String(model_approach), String(model_approach_alt))
    end

    # SIMULATION USING A LINEAR PROCESS
    # ###########################################################################################################
    # Get the corresponding process data
    for model_directory_lin in model_directories_lin
        lin_ar_process = set_up_ar_process_linear(number_of_stages, number_of_realizations, model_directory_lin, String(model_approach))

        # Create the stagewise independent sample data (realizations) for the simulation
        Random.seed!(12345+algo_params.forward_pass_seed)
        sampling_scheme_linear = SDDP.OutOfSampleMonteCarlo(model, use_insample_transition = true) do stage
            if model_directory_lin in ["msppy_model", "shapiro_model"]
                return get_out_of_sample_realizations_multivariate_linear(number_of_realizations, stage, model_directory_lin)
            else
                return get_out_of_sample_realizations_linear(number_of_realizations, stage, model_directory_lin)
            end
        end
        simulation_linear = LogLinearSDDP.Simulation(sampling_scheme = sampling_scheme_linear, number_of_replications = simulation_replications)

        # Using the sample data and the process data perform a simulation
        simulation_results = cross_simulate_linear(model, algo_params, lin_ar_process, model_directory_lin, simulation_linear)
        extended_simulation_analysis(simulation_results, file_path, String(model_approach), model_directory_lin)
    end

    return
end


""" 
LINEAR MODEL OPTIONS are
> shapiro_model, fitted_model, msppy_model

LOGLINEAR MODEL OPTIONS are
> custom_model, bic_model
"""
function run_model_starter()

    run_model(11111, :custom_model, [:bic_model], ["fitted_model", "shapiro_model"])
    # run_model(22222, :custom_model, [:bic_model], ["fitted_model", "shapiro_model"])
    # run_model(33333, :custom_model, [:bic_model], ["fitted_model", "shapiro_model"])
    # run_model(444444, :custom_model, [:bic_model], ["fitted_model", "shapiro_model"])
    # run_model(55555, :custom_model, [:bic_model], ["fitted_model", "shapiro_model"])

end

run_model_starter()