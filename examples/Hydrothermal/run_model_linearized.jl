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

include("hydrothermal_model_linearized.jl")
include("set_up_ar_process.jl")
include("simulation.jl")
include("cross_simulation_loglinear.jl")


function run_model(forward_pass_seed::Int, model_directory::String, model_directories_alternative::Vector{String}, model_directories_loglin::Vector{String})

    # MAIN MODEL AND RUN PARAMETERS    
    ###########################################################################################################
    number_of_stages = 120
    number_of_realizations = 100
    simulation_replications = 2000
    ###########################################################################################################
    file_identifier = "Run_" * model_directory * "_" * string(forward_pass_seed)
    file_path = "C:/Users/cg4102/Documents/julia_logs/Cut-sharing/" * file_identifier * "/"
    ispath(file_path) || mkdir(file_path)
    log_file = file_path * "LinearizedSDDP.log"
    ###########################################################################################################
    problem_params = LogLinearSDDP.ProblemParams(number_of_stages, number_of_realizations)
    simulation_regime = LogLinearSDDP.Simulation(sampling_scheme = SDDP.InSampleMonteCarlo(), number_of_replications = simulation_replications)
    algo_params = LogLinearSDDP.AlgoParams(stopping_rules = [SDDP.TimeLimit(15)], forward_pass_seed = forward_pass_seed, simulation_regime = simulation_regime, log_file = log_file, silent = false)
  
    # ADDITIONAL LOGGING TO SDDP.jl
    ###########################################################################################################
    log_f = open(log_file, "a")
    println(log_f, "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
    println(log_f, "PATH")
    println(log_f, "calling ")
    println(log_f, @__DIR__)
    println(log_f, Base.source_path())

    # Printing the time
    println(log_f, "DATETIME")
    println(log_f, Dates.now())

    # Printing the algo params
    println(log_f, "RUN DESCRIPTION")
    println(log_f, algo_params)
    close(log_f)

    # CREATE AND RUN MODEL
    ###########################################################################################################
    f = open(file_path * "inflows_lin.txt", "w")
    ar_process = set_up_ar_process_linear(number_of_stages, number_of_realizations, model_directory, "bic_model")
    model = model_definition(ar_process, problem_params, algo_params, f)  
    Random.seed!(algo_params.forward_pass_seed)

    # Train model
    SDDP.train(
        model,
        print_level = algo_params.print_level,
        log_file = algo_params.log_file,
        log_frequency = algo_params.log_frequency,
        stopping_rules = algo_params.stopping_rules,
        run_numerical_stability_report = algo_params.run_numerical_stability_report,
        log_every_seconds = 0.0
    )
    close(f)

    # SIMULATION USING THE LINEARIZED PROCESS
    ###########################################################################################################
    model.ext[:simulation_attributes] = [:level, :inflow, :spillage, :gen, :exchange, :deficit_part, :hydro_gen]
    
    # In-sample simulation
    simulation_results = LogLinearSDDP.simulate_linear(model, algo_params, problem_params, model_directory, algo_params.simulation_regime)
    extended_simulation_analysis(simulation_results, file_path, problem_params, model_directory, "in_sample")

    # ----------------------------------------------------------------------------------------------------------
    # Out-of-sample simulation
    Random.seed!(12345+algo_params.forward_pass_seed)
    sampling_scheme_linear = SDDP.OutOfSampleMonteCarlo(model, use_insample_transition = true) do stage
        if model_directory in ["msppy_model", "shapiro_model"]
            return get_out_of_sample_realizations_multivariate_linear(number_of_realizations, stage, model_directory)
        else
            return get_out_of_sample_realizations_linear(number_of_realizations, stage, model_directory)
        end
    end
    simulation_linear = LogLinearSDDP.Simulation(sampling_scheme = sampling_scheme_linear, number_of_replications = simulation_replications)
    simulation_results = LogLinearSDDP.simulate_linear(model, algo_params, problem_params, model_directory, simulation_linear)
    extended_simulation_analysis(simulation_results, file_path, problem_params, model_directory, model_directory)

    # ----------------------------------------------------------------------------------------------------------
    # Out-of-sample simulation (alternative linear models)
    for model_directory_alt in model_directories_alternative
        lin_ar_process = set_up_ar_process_linear(number_of_stages, number_of_realizations, model_directory_alt, "bic_model")
        for (node_index, node) in model.nodes
            if node_index > 1
               node.subproblem.ext[:ar_process_stage] = lin_ar_process.parameters[node_index]
            end
        end
    
        Random.seed!(12345+algo_params.forward_pass_seed)
        sampling_scheme_linear = SDDP.OutOfSampleMonteCarlo(model, use_insample_transition = true) do stage
            if model_directory_alt in ["msppy_model", "shapiro_model"]
                return get_out_of_sample_realizations_multivariate_linear(number_of_realizations, stage, model_directory_alt)
            else
                return get_out_of_sample_realizations_linear(number_of_realizations, stage, model_directory_alt)
            end
        end
        simulation_linear = LogLinearSDDP.Simulation(sampling_scheme = sampling_scheme_linear, number_of_replications = simulation_replications)
        simulation_results = LogLinearSDDP.simulate_linear(model, algo_params, problem_params, model_directory_alt, simulation_linear)
        extended_simulation_analysis(simulation_results, file_path, problem_params, model_directory, model_directory_alt)
    end

    # SIMULATION USING A LOGLINEAR PROCESS
    ###########################################################################################################
    # Get the corresponding process data
    for model_directory_loglin in model_directories_loglin
        loglin_ar_process = set_up_ar_process_loglinear(number_of_stages, number_of_realizations, model_directory_loglin, model_directory_loglin)
        LogLinearSDDP.initialize_process_state(model, loglin_ar_process)

        # Create the stagewise independent sample data (realizations) for the simulation
        Random.seed!(12345+algo_params.forward_pass_seed)

        sampling_scheme_loglinear = SDDP.OutOfSampleMonteCarlo(model, use_insample_transition = true) do stage
            return get_out_of_sample_realizations_loglinear(number_of_realizations, stage, model_directory_loglin)
        end
        simulation_loglinear = LogLinearSDDP.Simulation(sampling_scheme = sampling_scheme_loglinear, number_of_replications = simulation_replications)

        # Using the sample data and the process data perform a simulation
        simulation_results = cross_simulate_loglinear(model, algo_params, problem_params, loglin_ar_process, model_directory_loglin, simulation_loglinear)
        extended_simulation_analysis(simulation_results, file_path, problem_params, model_directory, model_directory_loglin)
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

    run_model(11111, "shapiro_model", ["fitted_model"], ["custom_model", "bic_model"])
    run_model(22222, "shapiro_model", ["fitted_model"], ["custom_model", "bic_model"])
    run_model(33333, "shapiro_model", ["fitted_model"], ["custom_model", "bic_model"])
    run_model(444444, "shapiro_model", ["fitted_model"], ["custom_model", "bic_model"])
    run_model(55555, "shapiro_model", ["fitted_model"], ["custom_model", "bic_model"])

    run_model(11111, "fitted_model", ["shapiro_model"], ["custom_model", "bic_model"])
    run_model(22222, "fitted_model", ["shapiro_model"], ["custom_model", "bic_model"])
    run_model(33333, "fitted_model", ["shapiro_model"], ["custom_model", "bic_model"])
    run_model(444444, "fitted_model", ["shapiro_model"], ["custom_model", "bic_model"])
    run_model(55555, "fitted_model", ["shapiro_model"], ["custom_model", "bic_model"])

end

run_model_starter()