import CSV
import DataFrames
import Distributions
import Random
import Infiltrator
include("LinearizedAutoregressivePreparation.jl")
include("../read_and_write_files.jl")
import .LinearizedAutoregressivePreparation

""" Method that generates "scenarios" in the sense that it generates a given number
of realizations and stages for the stagewise independent term (error term) in the PAR model.
The scenario data is stored in a txt file. 
Note that we do not required data for stage 1. """
function scenario_generation(number_of_realizations::Int64, number_of_stages::Int64, model_directory::String)

    output_file_name = model_directory * "/" * "scenarios_linear.txt"
    file_names = ["model_lin_SE.txt", "model_lin_S.txt", "model_lin_NE.txt", "model_lin_N.txt"]
    f = open(output_file_name, "w")

    # Read and store required standard deviations
    stds_all_systems = Vector{Float64}[]

    for system_number in 1:4
        # Get the standard deviation for the error terms
        file_name = model_directory * "/" * file_names[system_number]
        stds = read_model_std(file_name)
        push!(stds_all_systems, stds)
    end

    for t in 2:number_of_stages
        for i in 1:number_of_realizations    
            print(f, t, ";", i, ";", 1/number_of_realizations)

            for system in 1:4
                # Determine month
                month = mod(t, 12) > 0 ? mod(t,12) : 12

                # Generate a realization
                d = Distributions.Normal(0.0, stds_all_systems[system][month])
                realization = Distributions.rand(d, 1)

                # Write to file
                print(f, ";", realization[1])
            end
            println(f)
        end
    end
    close(f)
    return
end

""" Method that generates the history of the stochastic process that is/may be required for SDDP due to the lags of the process.
We use the last 2 months of the true historical data and then use the AR process definition to compute a fixed value
for stage 1 of the SDDP problem."""
function history_generation(model_directory::String)
    # Get last year of historical data as vector for all four reservoir systems
    historic_data_SE = LinearizedAutoregressivePreparation.data_frame_to_vector(last(LinearizedAutoregressivePreparation.read_raw_data("historical_data/hist1.csv"), 1))
    historic_data_S = LinearizedAutoregressivePreparation.data_frame_to_vector(last(LinearizedAutoregressivePreparation.read_raw_data("historical_data/hist2.csv"), 1))
    historic_data_NE = LinearizedAutoregressivePreparation.data_frame_to_vector(last(LinearizedAutoregressivePreparation.read_raw_data("historical_data/hist3.csv"), 1))
    historic_data_N = LinearizedAutoregressivePreparation.data_frame_to_vector(last(LinearizedAutoregressivePreparation.read_raw_data("historical_data/hist4.csv"), 1))
    historic_data = [historic_data_SE, historic_data_S, historic_data_NE, historic_data_N]

    # Read AR model data for all four reservoir systems
    model_SE = read_model_linear(model_directory * "/" * "model_lin_SE.txt")
    model_S = read_model_linear(model_directory * "/" * "model_lin_S.txt")
    model_NE = read_model_linear(model_directory * "/" * "model_lin_NE.txt")
    model_N = read_model_linear(model_directory * "/" * "model_lin_N.txt")
    models = [model_SE, model_S, model_NE, model_N]

    # Prepare output file
    output_file_name = model_directory * "/" * "history_linear.txt"
    f = open(output_file_name, "w")
    
    # Prepare storing AR history (in addition to output to file)
    ar_history = Dict{Int64,Any}()

    # Get stage and month
    for t in 0:1
        # Iterate over stages to set
        ar_history_stage = Vector{Float64}(undef, 4)
        
        month = t == 0 ? 12 : 1
        print(f, t)

        # Iterate over systems
        for ℓ in 1:4         
            if t == 1
                # Get model data for current month and system
                sigma = models[ℓ][month, "Sigma"]
                coefficients = parse_coefficients(models[ℓ][month, "Corr_coefficients"])

                # Generate a realization for eta
                d = Distributions.Normal(0.0, sigma)
                realization = Distributions.rand(d, 1)[1]

                # Compute the required value for eta
                ar_history_stage[ℓ] = coefficients[1] * exp(realization) * ar_history[t-1][ℓ] + exp(realization) * coefficients[2]               
            else
                # Get historical value 
                ar_history_stage[ℓ] = historic_data[ℓ][month]       
            end
            # Write to file
            print(f, ";", ar_history_stage[ℓ])
        end
        println(f)
        ar_history[t] = ar_history_stage
    end
    close(f)

    return ar_history
end

function scenario_and_history_generation(number_of_realizations::Int64, number_of_stages::Int64, scenario_generation_seed::Int64, model_directory::String)

    Random.seed!(scenario_generation_seed)
    scenario_generation(number_of_realizations, number_of_stages, model_directory)
    ar_history = history_generation(model_directory)

    return ar_history
end

scenario_and_history_generation(100, 120, 11111, "fitted_model")
