# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Copyright (c) 2026 Christian Fuellner <christian.fuellner@kit.edu>
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

include("Simulation/simulation.jl")
include("Simulation/historical_simulation_Markov.jl")
include("markov.jl")

function run_model(forward_pass_seed::Int, forward_pass_model::String, models_sim::Vector{String})
    
    # MAIN MODEL AND RUN PARAMETERS    
    ###########################################################################################################
    number_of_stages = 120
    number_of_realizations = 100
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

    run_description = "MC-SDDP: " * string(number_of_markov_nodes) * " nodes, " * forward_pass_model
    problem_params = LogLinearSDDP.ProblemParams(number_of_stages, number_of_realizations)
    simulation_regime = LogLinearSDDP.NoSimulation()
    algo_params = LogLinearSDDP.AlgoParams(stopping_rules = [SDDP.TimeLimit(7200)], forward_pass_seed = forward_pass_seed, simulation_regime = simulation_regime, log_file = log_file, silent = false, run_description = run_description)

    if forward_pass_model == "lattice"
        sampling_scheme_fp = SDDP.InSampleMonteCarlo()
        Random.seed!(algo_params.forward_pass_seed)
    else
        all_sample_paths = get_inflows_for_forward_pass(model, forward_pass_model, forward_pass_seed, 1000, number_of_stages)
        sampling_scheme_fp = SDDP.Historical(all_sample_paths)
    end

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
    close(log_f)

    ###########################################################################################################
    # SIMULATION
    ###########################################################################################################
    model.ext[:simulation_attributes] = [:v_1, :v_2, :v_3, :v_4, :gd_1_1, :gd_1_2, :gd_1_3, :gd_1_4, :gd_2_1, :gd_2_2, :gd_2_3, :gd_2_4, :gd_3_1, :gd_3_2, :gd_3_3, :gd_3_4, :gd_4_1, :gd_4_2, :gd_4_3, :gd_4_4,
    :th_1, :th_2, :th_3, :th_4, :s_1, :s_2, :s_3, :s_4, :q_1, :q_2, :q_3, :q_4, :f_12, :f_13, :f_15, :f_21, :f_31, :f_35, :f_45, :f_51, :f_53, :f_54]  

    # In-sample simulation
    # ----------------------------------------------------------------------------------------------------------
    """ We always perform an in-sample simulation using the lattice here, even if the forward pass model is not the lattice.
    Otherwise it is not an "in-sample-simulation" but very similar to the out-of-sample simulation."""
    simulation = LogLinearSDDP.Simulation(sampling_scheme = SDDP.InSampleMonteCarlo(), number_of_replications = simulation_replications)
    simulation_results = LogLinearSDDP.simulate_linear(model, algo_params, problem_params, "markov", simulation)
    extended_simulation_analysis_markov(simulation_results, file_path, problem_params, "markov", "in_sample")

    # Out-of-sample simulation
    # ----------------------------------------------------------------------------------------------------------
    for model_sim in models_sim
        all_sample_paths_sim = get_inflows_for_simulation(model, model_sim, forward_pass_seed, simulation_replications, number_of_stages)
        sampling_scheme = SDDP.Historical(all_sample_paths_sim)
        simulation = LogLinearSDDP.Simulation(sampling_scheme = sampling_scheme, number_of_replications = simulation_replications)
        simulation_results = LogLinearSDDP.simulate_linear(model, algo_params, problem_params, model_sim, simulation)
        extended_simulation_analysis_markov(simulation_results, file_path, problem_params, "markov", model_sim)
    end

    # Simulation using historical data
    # ----------------------------------------------------------------------------------------------------------
    all_sample_paths_hist = get_inflows_historical(model, number_of_stages)
    sampling_scheme_historical = SDDP.Historical(all_sample_paths_hist) 
    simulation_historical = LogLinearSDDP.Simulation(sampling_scheme = sampling_scheme_historical)
    simulation_results = historical_simulate_for_linear(model, algo_params, problem_params, "markov", simulation_historical)
    extended_simulation_analysis_markov(simulation_results, file_path, problem_params, "markov", "historical")   
    return
end


""" We do not use the BIC model here, as it is not Markovian."""
function run_model_starter()    
    run_model(11111, "lattice", ["custom_model", "fitted_model", "shapiro_model"])
    run_model(22222, "lattice", ["custom_model", "fitted_model", "shapiro_model"])
    run_model(33333, "lattice", ["custom_model", "fitted_model", "shapiro_model"])
    run_model(444444, "lattice", ["custom_model", "fitted_model", "shapiro_model"])
    run_model(55555, "lattice", ["custom_model", "fitted_model", "shapiro_model"])

    run_model(11111, "custom_model", ["custom_model", "fitted_model", "shapiro_model"])
    run_model(22222, "custom_model", ["custom_model", "fitted_model", "shapiro_model"])
    run_model(33333, "custom_model", ["custom_model", "fitted_model", "shapiro_model"])
    run_model(444444, "custom_model", ["custom_model", "fitted_model", "shapiro_model"])
    run_model(55555, "custom_model", ["custom_model", "fitted_model", "shapiro_model"])

    run_model(11111, "fitted_model", ["custom_model", "fitted_model", "shapiro_model"])
    run_model(22222, "fitted_model", ["custom_model", "fitted_model", "shapiro_model"])
    run_model(33333, "fitted_model", ["custom_model", "fitted_model", "shapiro_model"])
    run_model(444444, "fitted_model", ["custom_model", "fitted_model", "shapiro_model"])
    run_model(55555, "fitted_model", ["custom_model", "fitted_model", "shapiro_model"])

    run_model(11111, "shapiro_model", ["custom_model", "fitted_model", "shapiro_model"])
    run_model(22222, "shapiro_model", ["custom_model", "fitted_model", "shapiro_model"])
    run_model(33333, "shapiro_model", ["custom_model", "fitted_model", "shapiro_model"])
    run_model(444444, "shapiro_model", ["custom_model", "fitted_model", "shapiro_model"])
    run_model(55555, "shapiro_model", ["custom_model", "fitted_model", "shapiro_model"])
end


run_model_starter()