import Statistics
import Distances
import Distributions
import DataFrames
using DataFramesMeta
import Infiltrator

""" Compute mean, std, mae and rmse for a vector of historical data (true data) and an associated forecast."""
function compute_forecast_statistics(
    orig_vector::Vector{Float64},
    forecast_vector::Vector{Float64},
    description::String,
    )

    mean = Statistics.mean(orig_vector)
    mean_fc = Statistics.mean(forecast_vector)
    std = Statistics.std(orig_vector) 
    std_fc = Statistics.std(forecast_vector) 
    mae = Distances.meanad(orig_vector, forecast_vector)
    rmse = Distances.rmsd(orig_vector, forecast_vector)
    println(description, ": ", round(mean_fc,digits=2), " (", round(mean,digits=2), ")", ", std: ", round(std_fc,digits=2), " (", round(std,digits=2), ")", ", MAE: ", round(mae,digits=2), ", RMSE: ", round(rmse,digits=2))

    return
end

""" Compute point forecasts using a fitted model. 
Call methods for evaluation and plotting."""
function point_forecast(
    all_monthly_models::Vector{MonthlyModelStorage},
    data_log_detrended::Vector{Float64},
    data_log::Vector{Float64},
    data_orig::Vector{Float64},
    detrending_with_sigma::Bool,
    regressor_symbol::Symbol,
    with_plots::Bool,
)

    # Store existing data in DataFrame
    df = DataFrames.DataFrame()
    df[:, :log_detrended] = data_log_detrended
    df[:, :log] = data_log
    df[:, :orig] = data_orig
    df[:, :month] = repeat([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], outer=Int(floor(length(data_log_detrended)/12)))
    df[:, :row_number] = 1:size(df,1)
    df[:, :year] = df[!, :row_number]
    df = DataFramesMeta.@eachrow df begin :year = ceil(:row_number/12) end

    # Create forecast columns in DataFrame
    DataFrames.insertcols!(df, :predict => 0.0)
    DataFrames.insertcols!(df, :fc_log_detrended => 0.0)
    DataFrames.insertcols!(df, :fc_log => 0.0)
    DataFrames.insertcols!(df, :fc_orig => 0.0)
    DataFrames.insertcols!(df, :fc_orig_corr => 0.0)
    DataFrames.insertcols!(df, :fc_naive => 0.0)

    for row in 1:DataFrames.nrow(df)
        # Get current month
        month = df[row,:month]
        
        # Get model data for current month
        coef = GLM.coef(all_monthly_models[month].fitted_model)
        lag_order = all_monthly_models[month].lag_order
        μ = all_monthly_models[month].detrending_mean

        if detrending_with_sigma
            sigma = all_monthly_models[month].detrending_sigma
        else
            sigma = 1
        end

        if row > 12 && row <= (maximum(df[:, :year])-1)*12
            # Predicted value from model itself
            df[row, :predict] = GLM.predict(all_monthly_models[month].fitted_model)[df[row, :year]-all_monthly_models[month].year_offset]
            
            # Compute predicted value using the AR model
            prediction = coef[1]
            for i in 1:lag_order
                if row <= 24
                    prediction = prediction + coef[1+i] * df[row-i, :log_detrended]
                else
                    prediction = prediction + coef[1+i] * df[row-i, regressor_symbol]
                end
            end
            df[row, :fc_log_detrended] = prediction
            #residual = df[row, :log_detrended] - df[row, :fc_log_detrended]

            # Remove deseasonalization
            df[row, :fc_log] = df[row, :fc_log_detrended] * sigma + μ

            # Get standard error of regression
            n = length(GLM.residuals(all_monthly_models[month].fitted_model))
            std_error = sqrt(sum(GLM.residuals(all_monthly_models[month].fitted_model) .^2) / (n - 2)) 
            correction_factor = exp(std_error^2 / 2)

            # Remove log 
            df[row, :fc_orig] = exp(df[row, :fc_log])
            df[row, :fc_orig_corr] = correction_factor * df[row, :fc_orig]

            # Naive forecast
            df[row, :fc_naive] = df[row-1, :log_detrended]
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
    compute_forecast_statistics(df[:,:log_detrended], df[:,:fc_log_detrended], "Log-detrended")
    compute_forecast_statistics(df[:,:log_detrended], df[:,:fc_naive], "Naive")
    compute_forecast_statistics(df[:,:log], df[:,:fc_log], "Log")
    compute_forecast_statistics(df[:,:orig], df[:,:fc_orig], "Original")
    compute_forecast_statistics(df[:,:orig], df[:,:fc_orig_corr], "Original corrected")
    println()

    return
end

""" Call point forecast method. A one-step ahead forecast is applied."""
function static_point_forecast(
    all_monthly_models::Vector{MonthlyModelStorage},
    data_log_detrended::Vector{Float64},
    data_log::Vector{Float64},
    data_orig::Vector{Float64},
    detrending_with_sigma::Bool,
    with_plots::Bool,
)

    return point_forecast(all_monthly_models, data_log_detrended, data_log, data_orig, detrending_with_sigma, :log_detrended, with_plots)
end

""" Call point forecast method. A dynamic forecast is applied, i.e.
the last forecast values are used to predict the next one."""
function dynamic_point_forecast(
    all_monthly_models::Vector{MonthlyModelStorage},
    data_log_detrended::Vector{Float64},
    data_log::Vector{Float64},
    data_orig::Vector{Float64},
    detrending_with_sigma::Bool,
    with_plots::Bool,
)

    return point_forecast(all_monthly_models, data_log_detrended, data_log, data_orig, detrending_with_sigma, :fc_log_detrended, with_plots::Bool,)
end

""" Generates number_of_scenarios different scenarios/sample paths given the fitted model
by sampling from Normal distributed error terms."""
function generate_full_scenarios(
    system_number::Int64,
    df_original::DataFrames.DataFrame,
    all_monthly_models::Vector{MonthlyModelStorage},
    starting_value::Vector{Float64},
    number_of_scenarios::Int64,
    detrending_with_sigma::Bool,
    with_plots::Bool,
    )
    
    # Store existing data in DataFrame
    df = DataFrames.DataFrame()
    df[:, :month] = repeat([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], outer=79)
    df[:, :row_number] = 1:size(df,1)
    df[:, :year] = df[!, :row_number]
    df = DataFramesMeta.@eachrow df begin :year = ceil(:row_number/12) end

    # Create forecast columns in DataFrame
    DataFrames.insertcols!(df, :fc_log_detrended => 0.0)
    DataFrames.insertcols!(df, :fc_log => 0.0)
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
            coef = GLM.coef(all_monthly_models[month].fitted_model)
            lag_order = all_monthly_models[month].lag_order
            μ = all_monthly_models[month].detrending_mean

            if detrending_with_sigma
                sigma = all_monthly_models[month].detrending_sigma
            else
                sigma = 1
            end

            # Compute predicted value using the AR model
            if row <= length(starting_value)
                prediction = starting_value[row] 
            else
                prediction = coef[1]
                for i in 1:lag_order
                    prediction = prediction + coef[1+i] * df[row-i, :fc_log_detrended]
                end
            end

            # Get standard error of regression
            n = length(GLM.residuals(all_monthly_models[month].fitted_model))
            std_error = sqrt(sum(GLM.residuals(all_monthly_models[month].fitted_model) .^2) / (n - 2)) 

            # Sample error from normal distribution
            d = Distributions.Normal(0, std_error)
            error = rand(d, 1)
            df[row, :fc_log_detrended] = prediction + error[1]

            # Remove deseasonalization
            df[row, :fc_log] = df[row, :fc_log_detrended] * sigma + μ

            # Remove log 
            #correction_factor = exp(std_error^2 / 2)
            #df[row, :fc_orig] = correction_factor * exp(df[row, :fc_log])
            df[row, :fc_orig] = exp(df[row, :fc_log])
        end

        # Get monthly statistics
        for month in 1:12
            monthly_df = df[(df.month .== month),:]
            mean = Statistics.mean(monthly_df.fc_orig)
            std = Statistics.std(monthly_df.fc_orig)
            all_means[scenario, Symbol(month)] = mean
            all_stds[scenario, Symbol(month)] = std
        end

        # Get yearly statistics
        for year in 1:79
            yearly_df = df[(df.year .== year),:]
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