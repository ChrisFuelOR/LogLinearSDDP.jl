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
function get_cross_models(
    system_number::Int64,
    μ::Vector{Float64},
    σ::Vector{Float64},
)

    all_monthly_models = MonthlyModelStorage[]
    std_errors = Vector{Float64}()

    systems = ["SE", "S", "NE", "N"]
    system = systems[system_number]

    # Get model data for given system
    file_name = "cross_log_model/model_" * system * ".txt"
    loglinear_model = read_model_log_linear(file_name)

    for month in 1:12
        lag_order = loglinear_model[month, "Lag_order"]

        # Prepare data (if not all lags are in same year, delete offset number of years from Y)
        year_offset = 0
        if month - lag_order <= 0
            year_offset = 1
        end
     
        # Get monthly model data
        coefficients = parse_coefficients(loglinear_model[month, "Coefficients"])
        sigma = loglinear_model[month, "Sigma"]
        psi = loglinear_model[month, "Psi"]
        intercept = loglinear_model[month, "Intercept"] 

        # Store model information for validation, forecasts and usage in SDDP
        push!(all_monthly_models, MonthlyModelStorage(μ[month], σ[month], [coefficients[1], intercept, psi], lag_order, zeros(1,1), year_offset))
        push!(std_errors, sigma)    
    end

    return all_monthly_models, std_errors
end


""" Call point forecast method. A one-step ahead forecast is applied."""
function static_point_forecast_cross_model(
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
    DataFrames.insertcols!(df, :fc_orig => 0.0)
    DataFrames.insertcols!(df, :fc_naive => 0.0)

    for row in 1:DataFrames.nrow(df)
        # Get current month
        month = df[row,:month]
        
        # Get model data for current month
        coef = all_monthly_models[month].fitted_model[1]
        intercept = all_monthly_models[month].fitted_model[2]
        lag_order = all_monthly_models[month].lag_order # always 1

        if row > 12 && row <= (maximum(df[:, :year])-lag_order)*12
            # Compute predicted value using the AR model
            prediction = exp(intercept) * df[row-lag_order, :orig]^coef
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
        plot_forecasts2(df)
    end

    # Computation of statistics (mean, std, MAE, RMSE)
    compute_forecast_statistics(df[:,:orig], df[:,:fc_naive], "Naive")
    compute_forecast_statistics(df[:,:orig], df[:,:fc_orig], "Original")
    println()

    return
end


""" Generates number_of_scenarios different scenarios/sample paths given the fitted model
by sampling from Normal distributed error terms."""
function generate_full_scenarios_cross_model(
    system_number::Int,
    df_original::DataFrames.DataFrame,
    all_monthly_models::Vector{MonthlyModelStorage},
    starting_value::Vector{Float64},
    std_errors::Vector{Float64},
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
            
            # Get model data for current month
            # We ignore the (negligibly small) intercept here
            coef = all_monthly_models[month].fitted_model[1]
            intercept = all_monthly_models[month].fitted_model[2]
            psi = all_monthly_models[month].fitted_model[3]
            lag_order = all_monthly_models[month].lag_order # always 1

            # Compute predicted value using the AR model
            if row <= length(starting_value)
                prediction = starting_value[row] #* Random.rand()
            else
                prediction = exp(intercept) * df[row-lag_order, :fc_orig]^coef
            end

            # Sample error from normal distribution and compute forecast value
            d = Distributions.Normal(0, std_errors[month])
            error = rand(d, 1)
            df[row, :fc_orig] = prediction * exp(error[1] * psi)
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
function validate_ar_cross_model()

    # PARAMETER CONFIGURATION
    training_test_split = true
    with_plots = true
    
    # FILE PATH COMPONENTS
    directory_name = "historical_data"
    file_names = ["hist1.csv", "hist2.csv", "hist3.csv", "hist4.csv"]
    system_names = ["SE", "S", "NE", "N"]

    # ITERATE OVER POWER SYSTEMS AND PREPARE AUTOREGRESSIVE MODEL
    for system_number in 1:4

        # We reset the seed for each system to make sure that the same multivariate Normal distribution is used for each
        Random.seed!(12345)

        system_name = system_names[system_number]
        file_name = directory_name * "/" * file_names[system_number] 
 
        println()
        println("##############################################################################")
        println("##############################################################################")
        println("ANALYSIS FOR DATA FROM ", file_name)
        println("##############################################################################")
        
        # READING AND PREPARING THE DATA
        #######################################################################################
        # Get raw data data as dataframe (columns = months)
        df = read_raw_data(file_name)
        # Get logarithmized data as dataframe
        log_df = log.(df)
        # Get deseasonalized logarithmized data
        μ, σ, detrended_log_df = detrend_data(log_df, false, with_plots)

        # Split data into training data and test data if intended
        train_data_orig, test_data_orig = split_data(df, training_test_split, 0.82)     

        # READ THE GIVEN MODEL INFORMATION
        #######################################################################################
        cross_models, std_errors = get_cross_models(system_number, μ, σ)

        # MODEL VALIDATION: FORECASTS
        #######################################################################################
        println()
        # In-sample forecasts (on full data, training data, test data)
        static_point_forecast_cross_model(cross_models, data_frame_to_vector(train_data_orig), with_plots)
        static_point_forecast_cross_model(cross_models, data_frame_to_vector(test_data_orig), with_plots)
       
        # MODEL VALIDATION: SIMULATION
        #######################################################################################
        generate_full_scenarios_cross_model(system_number, df, cross_models, [df[1,1]], std_errors, 200, with_plots)

    end
end

