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

""" Compute static point forecasts using a fitted model. 
Call methods for evaluation and plotting."""

""" Call point forecast method. A one-step ahead forecast is applied."""
function static_point_forecast(
    all_monthly_models::Vector{MonthlyModelStorage},
    data_orig::Vector{Float64},
    std_errors::Vector{Float64},
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
        # We ignore the (negligibly small) intercept here
        coef = GLM.coef(all_monthly_models[month].fitted_model)[2]
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
function generate_full_scenarios(
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
            
            # Get correct month corresponding to lag = 1
            lag_month = month - 1 > 0 ? month - 1 : Int(12 - mod((1 - month), 12))
            μ_lag = all_monthly_models[lag_month].detrending_mean

            # Get model data for current month
            # We ignore the (negligibly small) intercept here
            coef = GLM.coef(all_monthly_models[month].fitted_model)[2]
            lag_order = all_monthly_models[month].lag_order # always 1
            μ = all_monthly_models[month].detrending_mean

            # Compute predicted value using the AR model
            if row <= length(starting_value)
                prediction = starting_value[row] 
            else
                prediction = exp(μ) + coef * exp(μ - μ_lag) * (df[row-lag_order, :fc_orig] - exp(μ_lag))
            end

            # Sample error from normal distribution and compute forecast value
            d = Distributions.Normal(0, std_errors[month])
            error = rand(d, 1)
            df[row, :fc_orig] = prediction * exp(error[1])
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
        plot_scenario_statistics(all_means, all_stds, df_original)
        plot_yearly_qq(yearly_means, df_original)
    end

    return
end