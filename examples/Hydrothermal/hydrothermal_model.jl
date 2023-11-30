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

include("simulation.jl")


struct Generator
    name::String
    cost::Float64 # in $/MWh
    min_gen::Float64 # in MWmonth
    max_gen::Float64 # in MWmonth
    system::Int64 
end

struct Reservoir
    system::Int64
    name::String
    max_gen::Float64 # in MWmonth
    max_level::Float64 # in MWmonth
    init_level::Float64 # in MWmonth
end


""" 
This long-term hydrothermal model covers a simplified version of the Brazilian power system, which is represented by 5 subsystems with 95 thermal generators and 4 energy equivalent reservoirs.
Among others, this model was analyzed in 
 > Shapiro et al. (2013): Risk neutral and risk averse Stochastic Dual Dynamic Programming method
 > Löhndorf and Shapiro (2019): Modeling time-dependent randomness in stochastic dual dynamic programming.

The data for this problem was provided by Nils Löhndorf.
"""

function model_definition(ar_process::LogLinearSDDP.AutoregressiveProcess, problem_params::LogLinearSDDP.ProblemParams, algo_params::LogLinearSDDP.AlgoParams)

    demand = CSV.read("demand.csv", DataFrames.DataFrame, header=false, delim=";")
    DataFrames.rename!(demand, [:t, Symbol(1), Symbol(2), Symbol(3), Symbol(4), Symbol(5)])
    
    num_of_sys = 5
    #1: SUDESTE (SE), 2: SUL (S), 3: NORDESTE (NE), 4: NORTE (N), 5: artificial for interconnections

    reservoirs = [
        Reservoir(1, "SUDESTE", 45414.3, 200717.6, 59419.3),
        Reservoir(2, "SUL", 13081.5, 19617.2, 5874.9),
        Reservoir(3, "NORDESTE", 9900.9, 51806.1, 12859.2),
        Reservoir(4, "NORTE", 7629.9, 12744.9, 5271.5),
    ]

    generators = [
        Generator("ANGRA 1", 21.49, 520.0, 657.0, 1), 
        Generator("ANGRA 2", 18.96, 1080.0, 1350.0, 1), 
        Generator("CARIOBA", 937, 0.0, 36.0, 1), 
        Generator("CCBS_L", 194.79, 59.3, 250.0, 1), 
        Generator("CCBS_TC", 222.22, 27.1, 250.0, 1), 
        Generator("COCAL", 140.58, 0.0, 28.0, 1), 
        Generator("CUIABA G CC", 6.27, 0.0, 529.0, 1), 
        Generator("DAIA", 505.92, 0.0, 44.0, 1), 
        Generator("DO ATLAN_CSA", 0.01, 219.8, 255.0, 1), 
        Generator("DO ATLANTICO", 112.46, 200.0, 235.0, 1), 
        Generator("EBOLT_L", 159.97, 0.0, 386.0, 1), 
        Generator("EBOLT_TC", 250.87, 0.0, 386.0, 1), 
        Generator("GOIANIA II", 550.66, 0.0, 145.0, 1), 
        Generator("IBIRITERMO", 188.89, 0.0, 226.0, 1), 
        Generator("IGARAPE", 645.3, 0.0, 131.0, 1), 
        Generator("JUIZ DE FORA", 150, 0.0, 87.0, 1), 
        Generator("LINHARES", 145.68, 0.0, 204.0, 1), 
        Generator("MACAE_L", 274.54, 0.0, 923.0, 1), 
        Generator("MACAE_TC", 253.83, 0.0, 923.0, 1), 
        Generator("NORTEFLU-1", 37.8, 400.0, 400.0, 1), 
        Generator("NORTEFLU-2", 51.93, 0.0, 100.0, 1), 
        Generator("NORTEFLU-3", 90.69, 0.0, 200.0, 1), 
        Generator("NORTEFLU-4", 131.68, 0.0, 169.0, 1), 
        Generator("NOVA PIRAT", 317.98, 0.0, 386.0, 1), 
        Generator("PIE-RP", 152.8, 0.0, 28.0, 1), 
        Generator("PIRAT.12 G", 470.34, 0.0, 200.0, 1), 
        Generator("PIRAT.34 VAP", 317.98, 0.0, 272.0, 1), 
        Generator("R.SILVEIRA", 523.35, 0.0, 30.0, 1), 
        Generator("ST.CRUZ 12", 730.54, 0.0, 168.0, 1), 
        Generator("ST.CRUZ 34", 310.41, 0.0, 440.0, 1), 
        Generator("ST.CRUZ N.DI", 730.54, 0.0, 400.0, 1), 
        Generator("T LAGOAS_L", 101.33, 0.0, 258.0, 1), 
        Generator("T LAGOAS_T", 140.34, 0.0, 258.0, 1), 
        Generator("T LAGOAS_TC", 292.49, 0.0, 258.0, 1), 
        Generator("T.NORTE 1", 610.33, 0.0, 64.0, 1), 
        Generator("T.NORTE 2", 487.56, 0.0, 340.0, 1), 
        Generator("TERMORIO_L", 122.65, 71.7, 1058.0, 1), 
        Generator("TERMORIO_TC", 214.48, 28.8, 1058.0, 1), 
        Generator("UTE BRASILIA", 1047.38, 0.0, 10.0, 1), 
        Generator("UTE SOL", 0.01, 133.0, 197.0, 1), 
        Generator("VIANA", 329.57, 0.0, 175.0, 1), 
        Generator("W.ARJONA", 197.85, 0.0, 206.0, 1), 
        Generator("XAVANTES", 733.54, 0.0, 54.0, 1), 
        Generator("ALEGRETE", 564.57, 0.0, 66.0, 2), 
        Generator("ARAUC_FIC", 219, 0.0, 485.0, 2), 
        Generator("ARAUCARIA", 219, 0.0, 485.0, 2), 
        Generator("CANDIOTA 3", 50.47, 210.0, 350.0, 2), 
        Generator("CANOAS", 541.93, 0.0, 161.0, 2),
        Generator("CHARQUEADAS", 154.1, 27.0, 72.0, 2), 
        Generator("CISFRAMA", 180.51, 0.0, 4.0, 2), 
        Generator("FIGUEIRA", 218.77, 9.6, 20.0, 2), 
        Generator("J.LACERDA A1", 189.54, 25.0, 100.0, 2), 
        Generator("J.LACERDA A2", 143.04, 79.5, 132.0, 2), 
        Generator("J.LACERDA B", 142.86, 147.5, 262.0, 2), 
        Generator("J.LACERDA C", 116.9, 228.0, 363.0, 2), 
        Generator("NUTEPA", 780, 0.0, 24.0, 2), 
        Generator("P.MEDICI A", 115.9, 49.7, 126.0, 2), 
        Generator("P.MEDICI B", 115.9, 105.0, 320.0, 2), 
        Generator("S.JERONIMO", 248.31, 5.0, 20.0, 2), 
        Generator("URUGUAIANA", 141.18, 0.0, 640.0, 2), 
        Generator("ALTOS", 464.64, 0.0, 13.0, 3), 
        Generator("ARACATI", 464.64, 0.0, 11.0, 3), 
        Generator("BAHIA I", 455.13, 0.0, 32.0, 3), 
        Generator("BATURITE", 464.64, 0.0, 11.0, 3), 
        Generator("CAMACARI D/G", 834.35, 0.7, 347.0, 3), 
        Generator("CAMACARI MI", 509.86, 0.0, 152.0, 3), 
        Generator("CAMACARI PI", 509.86, 0.0, 150.0, 3), 
        Generator("CAMPO MAIOR", 464.64, 0.0, 13.0, 3), 
        Generator("CAUCAIA", 464.64, 0.0, 15.0, 3), 
        Generator("CEARA_L", 185.09, 0.0, 220.0, 3), 
        Generator("CEARA_TC", 492.29, 0.0, 220.0, 3), 
        Generator("CRATO", 464.64, 0.0, 13.0, 3), 
        Generator("ENGUIA PECEM", 464.64, 0.0, 15.0, 3), 
        Generator("FAFEN", 188.15, 0.0, 138.0, 3), 
        Generator("FORTALEZA", 82.34, 223.0, 347.0, 3), 
        Generator("GLOBAL I", 329.37, 0.0, 149.0, 3), 
        Generator("GLOBAL II", 329.37, 0.0, 149.0, 3), 
        Generator("IGUATU", 464.64, 0.0, 15.0, 3), 
        Generator("JAGUARARI", 464.64, 0.0, 102.0, 3), 
        Generator("JUAZEIRO N", 464.64, 0.0, 15.0, 3), 
        Generator("MARACANAU I", 317.19, 0.0, 168.0, 3), 
        Generator("MARAMBAIA", 464.64, 0.0, 13.0, 3), 
        Generator("NAZARIA", 464.64, 0.0, 13.0, 3), 
        Generator("PAU FERRO I", 678.03, 0.0, 103.0, 3), 
        Generator("PETROLINA", 559.39, 0.0, 136.0, 3), 
        Generator("POTIGUAR", 611.57, 0.0, 53.0, 3), 
        Generator("POTIGUAR III", 611.56, 0.0, 66.0, 3), 
        Generator("TERMOBAHIA", 204.43, 0.0, 186.0, 3), 
        Generator("TERMOCABO", 325.67, 0.0, 50.0, 3), 
        Generator("TERMOMANAUS", 678.03, 0.0, 156.0, 3), 
        Generator("TERMONE", 329.2, 0.0, 171.0, 3), 
        Generator("TERMOPE", 70.16, 348.8, 533.0, 3), 
        Generator("VALE DO ACU", 287.83, 0.0, 323.0, 3), 
        Generator("NOVA OLINDA", 329.56, 0.0, 166.0, 4), 
        Generator("TOCANTINOPO", 329.56, 0.0, 166.0, 4)
    ]    
 
    num_of_res = length(reservoirs)
    num_of_gen = length(generators)

    curtailment_ratio = [0.05, 0.05, 0.1, 0.8]
    deficit_cost = [1142.8, 2465.4, 5152.46, 5845.54]
    # annual_discount_rate = 1-0.12
    monthly_discount_rate = 0.9906

    #exchange capacity from sytem k to l in MWmonth
    exchange_cap = [0.0 7379.0 1000.0 0.0 4000.0; 5625.0 0.0 0.0 0.0 0.0; 600.0 0.0 0.0 0.0 2236.0; 0.0 0.0 0.0 0.0 99999999.0; 3154.0 0.0 3951.0 3053.0 0.0]

    #exchange penalties for exchange from system k to l in %?
    exchange_pen = [0.0 0.001 0.001 0.001 0.0005; 0.001 0.0 0.001 0.001 0.0005; 0.001 0.001 0.0 0.001 0.0005; 0.001 0.001 0.001 0.0 0.0005; 0.0005 0.0005 0.0005 0.0005 0.0]

    model = SDDP.LinearPolicyGraph(
        stages = problem_params.number_of_stages,
        optimizer = Gurobi.Optimizer,
        sense = :Min,
        lower_bound = 0.0,
    )  do subproblem, t

        # Subproblem variables    
        JuMP.@variable(subproblem, level[k in 1:num_of_res], SDDP.State, lower_bound = 0.0, upper_bound = reservoirs[k].max_level, initial_value = reservoirs[k].init_level)
        JuMP.@variable(subproblem, hydro_gen[k in 1:num_of_res], lower_bound = 0.0, upper_bound = reservoirs[k].max_gen)
        JuMP.@variable(subproblem, spillage[k in 1:num_of_res], lower_bound = 0.0)
        JuMP.@variable(subproblem, inflow[k in 1:num_of_res]) # random variable
        JuMP.@variable(subproblem, gen[j in 1:num_of_gen], lower_bound = generators[j].min_gen, upper_bound = generators[j].max_gen)
        JuMP.@variable(subproblem, exchange[k in 1:num_of_sys, l in 1:num_of_sys], lower_bound = 0.0, upper_bound = exchange_cap[k,l])
        JuMP.@variable(subproblem, deficit[k in 1:num_of_sys])
       
        # Load curtailment modeling
        JuMP.fix(deficit[5], 0.0) # no curtailment for system 5
        JuMP.@variable(subproblem, deficit_part[k in 1:num_of_sys, i=1:4], lower_bound = 0.0, upper_bound = curtailment_ratio[i] * demand[t,Symbol(k)])        
    
        """ The deficit (load curtailment) costs are piecewise linear convex, i.e., they increase with the amount of load curtailment.
        We model this by splitting the whole deficit variable "deficit" into four parts "deficit_part" with different cost coefficients.
        Together they sum up to "deficit". Clearly, the system tries to fully allocate the deficit to those variables with the lowest cost.
        Therefore, naturally deficit_part[k,i] only takes non-negative values if deficit_part_[k,i-1] has reached its upper bound.
        The upper bounds are specified as percentages of the total load."""

        # Hydro balance
        coupling_ref = JuMP.@constraint(subproblem, hydro_balance[k in 1:num_of_res], level[k].out == level[k].in + inflow[k] - hydro_gen[k] - spillage[k])

        # Load balance
        JuMP.@constraint(subproblem, load_balance[k in 1:num_of_sys], sum(hydro_gen[l] for l in 1:num_of_res if reservoirs[l].system == k) + sum(gen[j] for j in 1:num_of_gen if generators[j].system == k) + sum(deficit_part[k,i] for i in 1:4) + sum(exchange[l,k] - exchange[k,l] for l in 1:num_of_sys) == demand[t,Symbol(k)])

        # Objective function
        SDDP.@stageobjective(subproblem, (monthly_discount_rate)^(t-1) * sum(sum(generators[j].cost * gen[j] for j in 1:num_of_gen if generators[j].system == k) + sum(deficit_cost[i] * deficit_part[k,i] for i in 1:4) + sum(exchange_pen[k,l] * exchange[k,l] for l in 1:num_of_sys) for k in 1:num_of_sys))

        # Parameterize inflow and demand
        if t == 1
            realizations = [[0.0, 0.0, 0.0, 0.0]]
        else
            realizations = ar_process.parameters[t].eta
        end

        # Store coupling constraint reference to access dual multipliers later
        if t != 1
            coupling_refs = subproblem.ext[:coupling_constraints] = Vector{JuMP.ConstraintRef}(undef, ar_process.parameters[t].dimension)
            for i in eachindex(coupling_refs)
                coupling_refs[i] = coupling_ref[i]
            end
        end

        SDDP.parameterize(subproblem, realizations) do ω
            JuMP.fix(inflow[1], ω[1])
            JuMP.fix(inflow[2], ω[2])
            JuMP.fix(inflow[3], ω[3])
            JuMP.fix(inflow[4], ω[4])

            #println(ω[1], ",", ω[2], ",", ω[3], ",", ω[4])
        end
    end

    return model
end


""" Reads stored model data."""
function read_model(file_name)
    f = open(file_name)
    df = CSV.read(f, DataFrames.DataFrame, header=false, delim=";")
    DataFrames.rename!(df, ["Month", "Lag_order", "Intercept", "Coefficients", "Psi", "Sigma"])    
    close(f)
    return df
end

""" Read stored realization data."""
function read_realization_data(model_directory::String)
    f = open("AutoregressivePreparation/" * model_directory * "/scenarios_nonlinear.txt")
    df = CSV.read(f, DataFrames.DataFrame, header=false, delim=";")
    DataFrames.rename!(df, ["Stage", "Realization_number", "Probability", "Realization_SE", "Realization_S", "Realization_NE", "Realization_N"])    
    close(f)
    return df
end

""" Get the realization data for a specific stage and system."""

function get_realization_data(eta_df::DataFrames.DataFrame, t::Int64, number_of_realizations::Int64)
    realizations = Tuple{Float64, Float64, Float64, Float64}[]

    for i in 1:number_of_realizations
        row = DataFramesMeta.@rsubset(eta_df, :Stage == t, :Realization_number == i)
        @assert DataFrames.nrow(row) == 1
        push!(realizations, (row[1, "Realization_SE"], row[1, "Realization_S"], row[1, "Realization_NE"], row[1, "Realization_N"]))
    end

    return realizations
end

""" Read stored history data for the process."""
function read_history_data(model_directory::String)
    f = open("AutoregressivePreparation/" * model_directory * "/history_nonlinear.txt")
    df = CSV.read(f, DataFrames.DataFrame, header=false, delim=";")
    DataFrames.rename!(df, ["Stage", "History_SE", "History_S", "History_NE", "History_N"])    
    close(f)
    return df
end

function get_ar_process(number_of_stages::Int64, number_of_realizations::Int64, model_directory::String)

    # AUTOREGRESSIVE PROCESS
    ###########################################################################################################
    # Main configuration
    # ---------------------------------------------------------------------------------------------------------
    # Read AR model data for all four reservoir systems
    data_SE = read_model("AutoregressivePreparation/" * model_directory * "/model_SE.txt")
    data_S = read_model("AutoregressivePreparation/" * model_directory * "/model_S.txt")
    data_NE = read_model("AutoregressivePreparation/" * model_directory * "/model_NE.txt")
    data_N = read_model("AutoregressivePreparation/" * model_directory * "/model_N.txt")
    data = [data_SE, data_S, data_NE, data_N]
    dim = 4 # number of hydro reservoirs

    # Read realization data
    eta_df = read_realization_data(model_directory)

    # Get max lag order (which is used as constant lag order for the process)
    lag_order = 0
    for df in data
        for month in 1:12
            current_lag_order = df[month, "Lag_order"]
            if current_lag_order > lag_order
                lag_order = current_lag_order
            end
        end
    end

    # Process history
    # ---------------------------------------------------------------------------------------------------------
    # define also ξ from -11 to 1
    ar_history = Dict{Int64,Any}()
    
    # Read history data and store in AR history
    history_data = read_history_data(model_directory)
    for t in -11:1
        row = t+12
        ar_history[t] = [history_data[row,"History_SE"], history_data[row,"History_S"], history_data[row,"History_NE"], history_data[row,"History_N"]]
    end

    # Process definition
    # ---------------------------------------------------------------------------------------------------------   
    ar_parameters = Dict{Int64, LogLinearSDDP.AutoregressiveProcessStage}()
    for t in 2:number_of_stages
        # Get month to stage
        month = mod(t, 12) > 0 ? mod(t,12) : 12

        intercept = Float64[]
        coefficients = zeros(dim, dim, lag_order)
        psi = Float64[]

        for ℓ in 1:4
            # Get model data
            df = data[ℓ]  

            # Get intercept
            push!(intercept, df[month, "Intercept"])

            # Get coefficients
            current_coefficients = df[month, "Coefficients"]
            current_coefficients = strip(current_coefficients, ']')
            current_coefficients = strip(current_coefficients, '[')
            current_coefficients = split(current_coefficients, ",")
            for k in eachindex(current_coefficients)
                coefficient = current_coefficients[k]
                coefficients[ℓ, ℓ, k] = parse(Float64, coefficient)
            end

            # Get psi
            push!(psi, df[month, "Psi"])
        end

        # Get eta data        
        eta = get_realization_data(eta_df, t, number_of_realizations)

        ar_parameters[t] = LogLinearSDDP.AutoregressiveProcessStage(dim, intercept, coefficients, eta, psi = psi)
    end
    
    # All stages
    ar_process = LogLinearSDDP.AutoregressiveProcess(lag_order, ar_parameters, ar_history)
    Infiltrator.@infiltrate

    return ar_process
end


function model_and_train()

    # MAIN MODEL AND RUN PARAMETERS    
    ###########################################################################################################
    number_of_stages = 120 #120
    number_of_realizations = 100 #100
    model_approach = :bic_model

    applied_solver = LogLinearSDDP.AppliedSolver()
    problem_params = LogLinearSDDP.ProblemParams(number_of_stages, number_of_realizations)
    simulation_regime = LogLinearSDDP.Simulation(sampling_scheme = SDDP.InSampleMonteCarlo(), number_of_replications = 10)

    algo_params = LogLinearSDDP.AlgoParams(stopping_rules = [SDDP.IterationLimit(30)], forward_pass_seed = 11111, simulation_regime = simulation_regime, model_approach = model_approach)
  
    # CREATE AND RUN MODEL
    ###########################################################################################################
    ar_process = get_ar_process(number_of_stages, number_of_realizations, String(model_approach))
    model = model_definition(ar_process, problem_params, algo_params)
    
    # Train model
    Random.seed!(algo_params.forward_pass_seed)
    LogLinearSDDP.train_loglinear(model, algo_params, problem_params, applied_solver, ar_process)

    # SIMULATION
    ###########################################################################################################
    # (1) In-sample simulation
    LogLinearSDDP.simulate_loglinear(model, algo_params, problem_params, algo_params.simulation_regime)

    # (2) Out-of-sample simulation using the nonlinear process
    sampling_scheme_loglinear = SDDP.OutOfSampleMonteCarlo(model, use_insample_transition = true) do stage
        return get_out_of_sample_realizations_loglinear(number_of_realizations, stage, String(model_approach))
    end
    simulation_loglinear = LogLinearSDDP.Simulation(sampling_scheme = sampling_scheme_loglinear, number_of_replications = 10)
    LogLinearSDDP.simulate_loglinear(model, algo_params, problem_params, simulation_loglinear)

    # (3) Out-of-sample simulation using the linear process
    # model_approach_lin = :TODO
    # sampling_scheme_linear = SDDP.OutOfSampleMonteCarlo(model, use_insample_transition = true) do stage
    #     return get_out_of_sample_realizations_linear(number_of_realizations, stage, String(model_approach_lin))
    # end
    # simulation_linear = LogLinearSDDP.Simulation(sampling_scheme = sampling_scheme_linear, number_of_replications = 10)
    # LogLinearSDDP.simulate_loglinear(model, algo_params, problem_params, simulation_linear)

    return
end

model_and_train()