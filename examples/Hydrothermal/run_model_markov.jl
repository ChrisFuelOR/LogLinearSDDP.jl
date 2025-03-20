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

#include("set_up_ar_process.jl")
include("simulation.jl")
#include("cross_simulation_loglinear.jl")
#include("historical_simulation_linear.jl")

function run_model(forward_pass_seed::Int, forward_pass_model::String, models_sim::Vector{String})
    
    # MAIN MODEL AND RUN PARAMETERS    
    ###########################################################################################################
    number_of_stages = 120
    number_of_realizations = 100 #TODO: required?
    number_of_markov_nodes = 100
    simulation_replications = 2000
    ###########################################################################################################
    file_identifier = "Run_MC-SDDP_" * forward_pass_model * "_" * string(forward_pass_seed)
    file_path = "C:/Users/cg4102/Documents/julia_logs/Cut-sharing/" * file_identifier * "/"
    ispath(file_path) || mkdir(file_path)
    log_file = file_path * "MC-SDDP.log"

    # CREATE MODEL AND PARAMETER STORAGE
    ###########################################################################################################
    model = get_hydrothermal_model_markov(number_of_markov_nodes)
    JuMP.set_optimizer(model, () -> Gurobi.Optimizer(GRB_ENV))

    if forward_pass_model == "lattice"
        sampling_scheme_fp = SDDP.InSampleMonteCarlo()
        Random.seed!(algo_params.forward_pass_seed)
    else
        all_sample_paths = get_inflows_for_forward_pass(model, model_approach, forward_pass_seed, number_of_replications, number_of_stages)
        sampling_scheme_fp = SDDP.Historical(all_sample_paths)
    end

    run_description = "MC-SDDP: " * string(number_of_markov_nodes) * " nodes, " * forward_pass_model
    problem_params = LogLinearSDDP.ProblemParams(number_of_stages, number_of_realizations)
    simulation_regime = LogLinearSDDP.NoSimulation()
    algo_params = LogLinearSDDP.AlgoParams(stopping_rules = [SDDP.TimeLimit(3600)], forward_pass_seed = forward_pass_seed, simulation_regime = simulation_regime, log_file = log_file, silent = false, run_description = run_description)

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


    ###########################################################################################################
    # RUNNING THE MODEL
    ###########################################################################################################
    # TODO: f = open(file_path * "inflows_lin.txt", "w")

    # Train model
    SDDP.train(
        model,
        sampling_scheme = sampling_scheme_fp,
        print_level = algo_params.print_level,
        log_file = algo_params.log_file,
        log_frequency = algo_params.log_frequency,
        stopping_rules = algo_params.stopping_rules,
        run_numerical_stability_report = algo_params.run_numerical_stability_report,
        log_every_seconds = 0.0
    )
    close(f)


    ###########################################################################################################
    # SIMULATION
    ###########################################################################################################
    model.ext[:simulation_attributes] = [:level, :inflow, :spillage, :gen, :exchange, :deficit_part, :hydro_gen]

    # In-sample simulation
    # ----------------------------------------------------------------------------------------------------------
    """ We always perform an in-sample simulation using the lattice here, even if the forward pass model is not the lattice.
    Otherwise it is not an "in-sample-simulation" but very similar to the out-of-sample simulation."""

    # TODO Prepare in-sample data
    # TODO Perform simulation
    extended_simulation_analysis(simulation_results, file_path, problem_params, model_directory, "in_sample")

    # Out-of-sample simulation
    # ----------------------------------------------------------------------------------------------------------
    for model_sim in models_sim
        # TODO Prepare out-of-sample data
        # TODO Perform simulation
        extended_simulation_analysis(simulation_results, file_path, problem_params, model_directory, model_sim)
    end

    # Simulation using historical data
    # ----------------------------------------------------------------------------------------------------------
    sampling_scheme_historical = SDDP.Historical(get_historical_sample_paths(problem_params.number_of_stages)) 
    simulation_historical = LogLinearSDDP.Simulation(sampling_scheme = sampling_scheme_historical)
    # TODO Perform simulation
    extended_simulation_analysis(simulation_results, file_path, problem_params, model_directory, "historical")   

    return
end


""" We do not use the BIC model here, as it is not Markovian."""
function run_model_starter()    
    run_model(11111, "lattice", ["custom_model", "fitted_model", "shapiro_model"])
end


run_model_starter()