# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Copyright (c) 2026 Christian Fuellner <christian.fuellner@kit.edu>
################################################################################

import Distributions
import DataFrames
import Random

include("../read_and_write_files.jl")

""" Get the data for the given model. """
function get_given_model(
    system_number::Int64,
    μ::Vector{Float64},
    σ::Vector{Float64},
    model_directory::String,
)

    all_monthly_models = MonthlyModelStorage[]

    # Read gamma data
    gamma = read_gamma_data(system_number, model_directory)

    for month in 1:12
        # Prepare data (if not all lags are in same year, delete offset number of years from Y)
        lag_order = 1
        year_offset = 0
        if month - lag_order <= 0
            year_offset = 1
        end

        # Store model information for validation, forecasts and usage in SDDP
        push!(all_monthly_models, MonthlyModelStorage(μ[month], σ[month], gamma[month], lag_order, year_offset))        
    end

    return all_monthly_models
end

""" Get the covariance matrix of the residuals for the given model. """
function get_given_sigma(
    model_directory::String,
)

    # Vector to store sigma matrices
    sigma_matrices = Array{Float64,2}[]

    for month in 1:12
        # Read sigma data
        sigma_matrix = read_sigma_data(month, model_directory)

        # Store sigma matrix
        push!(sigma_matrices, sigma_matrix)        
    end

    return sigma_matrices
end


""" Writes the output of the model to the REPL and to a txt file.
Corrects the coefficients so that they fit SDDP before."""
function model_output_given_model(
    all_monthly_models::Vector{MonthlyModelStorage},
    sigma_matrices::Vector{Array{Float64,2}},
    output_file_name::String,
)

    f = open(output_file_name, "w")

    for month in 1:12
        # Get model parameters (note that we ignore the very small intercepts here, to stay in accordance with Shapiro et al.)
        lag_order = all_monthly_models[month].lag_order
        coefficient = all_monthly_models[month].fitted_model
        sigma_matrix = sigma_matrices[month]
        monthly_mean = all_monthly_models[month].detrending_mean

        # Get lag month and corresponding max_gen
        lag_month = month - 1 > 0 ? month - 1 : Int(12 - mod((1 - month), 12))
        lag_mean = all_monthly_models[lag_month].detrending_mean 

        # Compute corrected coefficients for SDDP
        coefficient_corrected = Vector{Float64}(undef, 2)
        coefficient_corrected[1] = coefficient * exp(monthly_mean - lag_mean)
        coefficient_corrected[2] = (1-coefficient) * exp(monthly_mean)

        # Output to console
        println("Month: ", month)
        println("Lag order: ", lag_order)
        println("AR(1) coefficients: ", coefficient)
        println("Residual standard errors: ", sigma_matrix)
        println("-----------------------------------------")  
        println("Corrected coefficients: ", coefficient_corrected)
        println()  

        # Output to file
        println(f, month, ";", lag_order, ";", coefficient, ";", coefficient_corrected, ";", Inf)

    end

    println("#############################################")     
    close(f)   

end


""" Call point forecast method. A one-step ahead forecast is applied."""
function static_point_forecast_given_model(
    all_monthly_models::Vector{MonthlyModelStorage},
    data_orig::Vector{Float64},
    with_plots::Bool,
)

    # Store existing data in DataFrame
    df = DataFrames.DataFrame()
    df[:, :orig] = data_orig
    df[:, :month] = repeat([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], outer=Int(floor(length(data_orig)/12)))
    df[:, :row_number] = 1:size(df,1)
    df[:, :year] = df[!, :row_number]
    df = DataFramesMeta.@eachrow df begin :year = ceil(:row_number/12) end

    # Create forecast columns in DataFrame
    DataFrames.insertcols!(df, :predict => 0.0)
    DataFrames.insertcols!(df, :fc_orig => 0.0)
    DataFrames.insertcols!(df, :fc_naive => 0.0)

    for row in 1:DataFrames.nrow(df)
        # Get current month
        month = df[row,:month]
        
        # Get correct month corresponding to lag = 1
        lag_month = month - 1 > 0 ? month - 1 : Int(12 - mod((1 - month), 12))
        μ_lag = all_monthly_models[lag_month].detrending_mean

        # Get model data for current month
        coef = all_monthly_models[month].fitted_model
        lag_order = all_monthly_models[month].lag_order # always 1
        μ = all_monthly_models[month].detrending_mean

        if row > 12 && row <= (maximum(df[:, :year])-lag_order)*12
            # Compute predicted value using the AR model
            prediction = exp(μ) + coef * exp(μ - μ_lag) * (df[row-lag_order, :orig] - exp(μ_lag))

            df[row, :fc_orig] = prediction

            # Naive forecast
            df[row, :fc_naive] = df[row-lag_order, :orig]
        end
    end

    # Delete rows without prediction from dataframe
    DataFrames.delete!(df, [i for i in (maximum(df[:, :year])-1)*12 + 1 : DataFrames.nrow(df)])        
    DataFrames.delete!(df, [i for i in 1:12])        

    # Plotting
    if with_plots
        plot_forecasts(df)
    end

    # Computation of statistics (mean, std, MAE, RMSE)
    compute_forecast_statistics(df[:,:orig], df[:,:fc_naive], "Naive")
    compute_forecast_statistics(df[:,:orig], df[:,:fc_orig], "Original")
    println()

    return
end


""" Generates number_of_scenarios different scenarios/sample paths given the fitted model
by sampling from Normal distributed error terms."""
function generate_full_scenarios_given_model(
    system_number::Int,
    df_original::DataFrames.DataFrame,
    all_monthly_models::Vector{MonthlyModelStorage},
    starting_value::Vector{Float64},
    sigma_matrices::Vector{Array{Float64,2}},
    number_of_scenarios::Int64,
    with_plots::Bool,
    )
    
    # Store existing data in DataFrame
    df = DataFrames.DataFrame()
    df[:, :month] = repeat([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], outer=79)
    df[:, :row_number] = 1:size(df,1)
    df[:, :year] = df[!, :row_number]
    df = DataFramesMeta.@eachrow df begin :year = ceil(:row_number/12) end

    # Create forecast columns in DataFrame
    DataFrames.insertcols!(df, :fc_orig => 0.0)

    # Create a DataFrame for the monthly means and std
    all_means = DataFrames.DataFrame()
    all_means[:, :scenario_number] = [i for i in 1:number_of_scenarios]
    for month in 1:12
        DataFrames.insertcols!(all_means, Symbol(month) => 0.0)
    end
    all_stds = copy(all_means)

    # Create vector for yearly means
    yearly_means = Float64[]

    for scenario in 1:number_of_scenarios
        for row in 1:DataFrames.nrow(df)
            # Get current month
            month = df[row,:month]
            
            # Get correct month corresponding to lag = 1
            lag_month = month - 1 > 0 ? month - 1 : Int(12 - mod((1 - month), 12))
            μ_lag = all_monthly_models[lag_month].detrending_mean

            # Get model data for current month
            # We ignore the (negligibly small) intercept here
            coef = all_monthly_models[month].fitted_model
            lag_order = all_monthly_models[month].lag_order # always 1
            μ = all_monthly_models[month].detrending_mean

            # Compute predicted value using the AR model
            if row <= length(starting_value)
                prediction = starting_value[row] #* Random.rand()
            else
                prediction = exp(μ) + coef * exp(μ - μ_lag) * (df[row-lag_order, :fc_orig] - exp(μ_lag))
            end

            # Sample error from multivariate normal distribution and compute forecast value
            d = Distributions.MvNormal(zeros(4), sigma_matrices[month])
            error = rand(d, 1)
            df[row, :fc_orig] = prediction * exp(error[system_number])
        end

        current_df = df[121:948,:]

        # Get monthly statistics
        for month in 1:12
            monthly_df = current_df[(current_df.month .== month),:]
            mean = Statistics.mean(monthly_df.fc_orig)
            std = Statistics.std(monthly_df.fc_orig)
            all_means[scenario, Symbol(month)] = mean
            all_stds[scenario, Symbol(month)] = std
        end

        # Get yearly statistics
        for year in 11:79
            yearly_df = current_df[(current_df.year .== year),:]
            mean = Statistics.mean(yearly_df.fc_orig)
            push!(yearly_means, mean)
        end

    end
    all_means = select!(all_means, Not([:scenario_number]))
    all_stds = select!(all_stds, Not([:scenario_number]))

    # Plots if required
    if with_plots
        plot_scenario_statistics(system_number, all_means, all_stds, df_original)
        plot_yearly_qq(yearly_means, df_original)
    end

    return
end


""" Function to validate the AR models provided by the msppy package and the paper from Shapiro et al."""
function validate_ar_model()

    # PARAMETER CONFIGURATION
    training_test_split = true
    with_plots = false
    
    # FILE PATH COMPONENTS
    directory_name = "historical_data"
    file_names = ["hist1.csv", "hist2.csv", "hist3.csv", "hist4.csv"]
    system_names = ["SE", "S", "NE", "N"]
    output_directory = "shapiro_model/"

    # ITERATE OVER POWER SYSTEMS AND PREPARE AUTOREGRESSIVE MODEL
    for system_number in 4:4

        # We reset the seed for each system to make sure that the same multivariate Normal distribution is used for each
        Random.seed!(12345)

        system_name = system_names[system_number]
        file_name = directory_name * "/" * file_names[system_number] 
        output_file_name = output_directory * "/model_lin_" * string(system_name) * ".txt"   
 
        println()
        println("##############################################################################")
        println("##############################################################################")
        println("ANALYSIS FOR DATA FROM ", file_name)
        println("##############################################################################")
        
        # READING AND PREPARING THE DATA
        #######################################################################################
        # Get raw data data as dataframe (columns = months)
        df = read_raw_data(file_name)
        # Get logarithmitzed data as dataframe
        log_df = log.(df)
        # Get deseasonalized logarithmized data
        μ, σ, detrended_log_df = detrend_data(log_df, false, with_plots)

        # Split data into training data and test data if intended
        train_data_orig, test_data_orig = split_data(df, training_test_split, 0.82)     

        # READ THE GIVEN MODEL INFORMATION
        #######################################################################################
        all_monthly_models = get_given_model(system_number, μ, σ, output_directory)
        sigma_matrices = get_given_sigma(output_directory)

        # MODEL VALIDATION: FORECASTS
        #######################################################################################
        println()
        # In-sample forecasts (on full data, training data, test data)
        static_point_forecast_given_model(all_monthly_models, data_frame_to_vector(train_data_orig), with_plots)
        static_point_forecast_given_model(all_monthly_models, data_frame_to_vector(test_data_orig), with_plots)
       
        # MODEL VALIDATION: SIMULATION
        #######################################################################################
        generate_full_scenarios_given_model(system_number, df, all_monthly_models, [df[1,1]], sigma_matrices, 1000, true)

        # COEFFICIENT REFORMULATION AND MODEL OUTPUT
        #######################################################################################
        println("Model output for ", file_name)
        model_output_given_model(all_monthly_models, sigma_matrices, output_file_name)

    end
end

