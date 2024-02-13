import CSV
import DataFrames
import Distributions
import Random
include("AutoregressivePreparation.jl")
include("../read_and_write_files.jl")
import .AutoregressivePreparation

""" Method that generates "scenarios" in the sense that it generates a given number
of realizations and stages for the stagewise independent term (error term) in the PAR model.
The scenario data is stored in a txt file. 
Note that we do not required data for stage 1. """
function scenario_generation(number_of_realizations::Int64, number_of_stages::Int64, model_directory::String)

    output_file_name = model_directory * "/" * "scenarios_nonlinear.txt"
    file_names = ["model_SE.txt", "model_S.txt", "model_NE.txt", "model_N.txt"]
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
We use the last 12 months of the true historical data and then use the AR process definition to compute a fixed value
for stage 1 of the SDDP problem."""
function history_generation(model_directory::String)
    # Get last year of historical data as vector for all four reservoir systems
    historic_data_SE = AutoregressivePreparation.data_frame_to_vector(last(AutoregressivePreparation.read_raw_data("historical_data/hist1.csv"), 1))
    historic_data_S = AutoregressivePreparation.data_frame_to_vector(last(AutoregressivePreparation.read_raw_data("historical_data/hist2.csv"), 1))
    historic_data_NE = AutoregressivePreparation.data_frame_to_vector(last(AutoregressivePreparation.read_raw_data("historical_data/hist3.csv"), 1))
    historic_data_N = AutoregressivePreparation.data_frame_to_vector(last(AutoregressivePreparation.read_raw_data("historical_data/hist4.csv"), 1))
    historic_data = [historic_data_SE, historic_data_S, historic_data_NE, historic_data_N]

    # Read AR model data for all four reservoir systems
    model_SE = read_model_loglinear(model_directory * "/" * "model_SE.txt")
    model_S = read_model_loglinear(model_directory * "/" * "model_S.txt")
    model_NE = read_model_loglinear(model_directory * "/" * "model_NE.txt")
    model_N = read_model_loglinear(model_directory * "/" * "model_N.txt")
    models = [model_SE, model_S, model_NE, model_N]

    all_months = [12,11,10,9,8,7,6,5,4,3,2,1]

    # Prepare output file
    output_file_name = model_directory * "/" * "history_nonlinear.txt"
    f = open(output_file_name, "w")
    
    # Prepare storing AR history (in addition to output to file)
    ar_history = Dict{Int64,Any}()

    # Iterate over stages to set
    for counter in 12:-1:0
        ar_history_stage = Vector{Float64}(undef, 4)

        # Get stage and month
        t = 1 - counter
        month_identifier = mod(counter, 12) > 0 ? Int(mod(counter, 12)) : 12
        month = all_months[month_identifier]

        print(f, t)

        # Iterate over systems
        for ℓ in 1:4         

            if t == 1
                # Get model data for current month and system
                lag_order = models[ℓ][month, "Lag_order"]
                intercept = models[ℓ][month, "Intercept"]
                psi = models[ℓ][month, "Psi"]
                sigma = models[ℓ][month, "Sigma"]
                current_coefficients = parse_coefficients(models[ℓ][month, "Coefficients"])

                coefficients = zeros(lag_order, 4, 4)
                for k in eachindex(current_coefficients)
                    coefficients[k, ℓ, ℓ] = current_coefficients[k]
                end

                # Generate a realization for eta
                d = Distributions.Normal(0.0, sigma)
                realization = Distributions.rand(d, 1)[1]

                # Compute the required value for eta
                ar_history_stage[ℓ] = exp(intercept) * exp(psi * realization) * prod(ar_history[t-k][ℓ] ^ coefficients[k, ℓ, ℓ] for k in 1:lag_order)
                
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


scenario_and_history_generation(100, 120, 11111, "bic_model")
