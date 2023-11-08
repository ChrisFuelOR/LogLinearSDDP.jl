import CSV
import DataFrames
using DataFramesMeta
import Infiltrator
import Statistics
import Distributions
import StatsBase
using Plots; gr()#; pgfplotsx()
import StatsPlots
import GLM
import HypothesisTests
import Distances


struct MonthlyModelStorage
    mean::Float64
    sigma::Float64
    lag_order::Int
    coefficients::Vector{Float64}
    prediction::Vector{Float64}
    residuals::Vector{Float64}
    status_flag::Symbol

    function MonthlyModelStorage(
        input_data,
        fitted_model,
        lag_order,
        flag
    )
        return new(
            Statistics.mean(input_data),
            Statistics.std(input_data),
            lag_order,
            GLM.coef(fitted_model),
            GLM.predict(fitted_model),
            GLM.residuals(fitted_model),
            flag,
        )
    end
end


function read_raw_data(file_name::String)
    raw_data = CSV.read(file_name, DataFrames.DataFrame, header=false, delim=";")
    DataFrames.rename!(raw_data, ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])    

    return raw_data
end

function data_frame_to_vector(data_frame::DataFrames.DataFrame)
    full_vector = Float64[]
    for row in eachrow(data_frame)
        append!(full_vector, Vector(row))
    end

    return full_vector
end

function detrend_data(df::DataFrames.DataFrame, with_sigma::Bool)

    # Actual detrending (either subtraction of mean or subtraction of mean + division by standard deviation)
    #######################################################################################
    μ = Statistics.mean.(eachcol(df))
    if with_sigma
        σ = Statistics.std.(eachcol(df))
        residuals = copy(df)
        for col_name in names(df)
            residuals[!, col_name] = (df[!, col_name] .- μ[parse(Int, col_name)]) ./ σ[parse(Int, col_name)]
        end
    else
        residuals = copy(df)
        for col_name in names(df)
            residuals[!, col_name] = (df[!, col_name] .- μ[parse(Int, col_name)])
        end
    end

    # Compare distribution of monthly data before and after deseasonalization (box-plots)
    #######################################################################################
    bx_plot_1 = StatsPlots.@df df StatsPlots.boxplot(cols(), legend=false, xticks=1:12)
    bx_plot_2 = StatsPlots.@df residuals StatsPlots.boxplot(cols(), legend=false, xticks=1:12)
    Plots.display(bx_plot_1)
    Plots.display(bx_plot_2)

    # Plot of the original time series which illustrates the seasonality
    ts_plot = Plots.plot(data_frame_to_vector(df), legend=false, color=:red, lw=2)
    Plots.display(ts_plot)
    return residuals
end

function split_data(df::DataFrames.DataFrame, split_bool::Bool, split_percentage::Float64)

    if split_bool
        split_bound = Int(floor(split_percentage * DataFrames.nrow(df)))
        training_df = df[1:split_bound, :]
        test_df = df[split_bound+1:DataFrames.nrow(df), :]
        return training_df, test_df
    else
        return df, df
    end
end


function model_fitting(df::DataFrames.DataFrame, lag_order::Int)

    if lag_order == 1
        return GLM.lm(GLM.@formula(Y~A), df)    
    elseif lag_order == 2
        return GLM.lm(GLM.@formula(Y~A+B), df)    
    elseif lag_order == 3
        return GLM.lm(GLM.@formula(Y~A+B+C), df)    
    elseif lag_order == 4
        return GLM.lm(GLM.@formula(Y~A+B+C+D), df)    
    elseif lag_order == 5
        return GLM.lm(GLM.@formula(Y~A+B+C+D+E), df)    
    end                    

end


function model_fitting(df::DataFrames.DataFrame, month::Int, lag_order::Int)

    # data modification flag
    flag = :full

    # Get data for regressand
    Y = copy(df[!, month])
    
    # Prepare data (if not all lags are in same year, delete first year for Y)
    if month - lag_order <= 0
        Y = deleteat!(Y, 1)
        flag = :not_full
    end
    
    # Construct DataFrame for fitting
    df_for_fitting = DataFrames.DataFrame(Y = Y)

    # Iterate over lags
    for lag in 1:lag_order
        # Get correct month corresponding to lag
        lag_month = month - lag > 0 ? month - lag : 12 - (lag - month)    

        # Get regressor time series
        X = copy(df[!, lag_month]) 

        # Prepare data (if not all lags are in same year, delete first or last year for Y)
        if month - lag_order <= 0 && lag_month < month
            X = deleteat!(X,1)
        elseif month - lag_order <= 0 && lag_month >= month
            X = deleteat!(X,length(X))
        end
       
        column_names = ["A", "B", "C", "D", "E"]
        df_for_fitting[!, column_names[lag]] = X
    end

    # Fitting the model using OLS
    ols_result = model_fitting(df_for_fitting, lag_order)

    return ols_result, df_for_fitting, flag
end


function model_identification(df::DataFrames.DataFrame, month::Int)

    lowest_bic = Inf
    best_lag_order = 0

    println()
    println("##############################################################################")

    for lag_order in 1:5
        print("MODEL IDENTIFICATION - PAR MODEL FOR MONTH ", month, " AND LAG ORDER ", lag_order)

        fitted_model, _ = model_fitting(df, month, lag_order)

        if GLM.bic(fitted_model) < lowest_bic
            lowest_bic = GLM.bic(fitted_model)
            best_lag_order = lag_order
        end
        println("   - BIC: ", round(GLM.bic(fitted_model), digits=2))
    end

    return best_lag_order
end


function goodness_of_fit(fitted_model::Any)
    # adjusted R²
    println("adjusted R²: ", GLM.adjr2(fitted_model))

    return
end


function significance_tests(fitted_model::Any, lag_order::Int)
    # t-test statistics and p-value for 5% significance level
    alpha = 0.05
    significant = true
    for i in 1:lag_order
        dof = length(GLM.fitted(fitted_model)) - lag_order - 1
        t_critical = Distributions.quantile(Distributions.TDist(dof), 1-alpha)

        if !(GLM.coeftable(fitted_model).cols[3][i+1] > t_critical && GLM.coeftable(fitted_model).cols[4][i+1] < alpha)
            significant = false
        end
    end
    println("p-value and t-test imply significance: ", significant)            

    # F-test (required for higher lag orders)
    println("F-test: ", GLM.ftest(fitted_model.model))

    return
end


function autocorrelation_tests(residuals::Vector{Float64}, data_for_fitting::DataFrames.DataFrame)
    # PACF of the residuals
    pacf = StatsBase.pacf(residuals, [i for i=1:30])
    plot_pacf = Plots.plot(Plots.bar(pacf), ylims=(-1,1), legend=false)
    Plots.hline!([-0.2, 0.2], color=:black, lw=2, ls=:dash)
    Plots.display(plot_pacf)


    # Ljung-Box test for residuals
    # lb_statistics = HypothesisTests.LjungBoxTest(residuals, 10, lag_order)
    # println("Ljung-Box test for residuals: ", lb_statistics)
    # println("Ljung-Box test for residuals: ", "p-value = ", HypothesisTests.pvalue(lb_statistics), ", ", HypothesisTests.pvalue(lb_statistics) < 0.05 ? "reject h_0" : "fail to reject h_0")
    # NOTE: Removed, as Ljung-box test is not appropriate for residuals of AR models, see https://stats.stackexchange.com/questions/148004/testing-for-autocorrelation-ljung-box-versus-breusch-godfrey
    
    # Breusch-Godfrey test for residuals
    # Get regressor_matrix
    regressor_matrix = Matrix(DataFrames.select!(data_for_fitting, DataFrames.Not([:Y])))
    bg_statistics = HypothesisTests.BreuschGodfreyTest(regressor_matrix, residuals, 30, true)
    # println("Breusch-Godfrey test for residuals: ", bg_statistics)
    println("Breusch-Godfrey test: ", "p-value = ", round(HypothesisTests.pvalue(bg_statistics),digits=2), ", ", HypothesisTests.pvalue(bg_statistics) < 0.05 ? "reject h_0 (no autocorrelation)" : "fail to reject h_0 (no autocorrelation)")
 
    return
end


function heteroscedasticity_tests(residuals::Vector{Float64})

    # No formal diagnostic test, but plots for analysis
    plot_scat = Plots.scatter(residuals, legend=false)
    Plots.display(plot_scat)

    return
end


function normality_tests(residuals::Vector{Float64})

    # Plot 1: Normalized histogram of residuals
    plot_hist = Plots.histogram(residuals, label="Empirical", color=:gray)
    Plots.display(plot_hist)

    # Plot 2: Quantile-quantile plot of residuals 
    plot_qq = Plots.plot(StatsPlots.qqplot(Distributions.Normal(0,StatsBase.std(residuals)), residuals, qqline = :R))
    # plot_qq = Plots.plot(StatsPlots.qqnorm(residuals, qqline = :R))
    Plots.display(plot_qq)

    # Jarque-Bera test
    jb_statistics = HypothesisTests.JarqueBeraTest(residuals, adjusted=false)
    println("Jarque-Bera test: ", "p-value = ", round(HypothesisTests.pvalue(jb_statistics),digits=2), ", ", HypothesisTests.pvalue(jb_statistics) < 0.05 ? "reject h_0 (normality)" : "fail to reject h_0 (normality)")

    alm_statistics = HypothesisTests.JarqueBeraTest(residuals, adjusted=true)
    println("Adjusted Lagrangian Multiplier test: ", "p-value = ", round(HypothesisTests.pvalue(jb_statistics),digits=2), ", ", HypothesisTests.pvalue(jb_statistics) < 0.05 ? "reject h_0 (normality)" : "fail to reject h_0 (normality)")

    # Anderson-Darling test
    ad_statistics = HypothesisTests.OneSampleADTest(residuals, Distributions.Normal(0, StatsBase.std(residuals)))
    println("Anderson-Darling test: ", "p-value = ", round(HypothesisTests.pvalue(ad_statistics),digits=2), ", ", HypothesisTests.pvalue(ad_statistics) < 0.05 ? "reject h_0 (normality)" : "fail to reject h_0 (normality)")

    # Kolmogorov-Smirnov test
    ks_statistics = HypothesisTests.ExactOneSampleKSTest(residuals, Distributions.Normal(0, StatsBase.std(residuals)))
    println("Kolmogorov-Smirnov test: ", "p-value = ", round(HypothesisTests.pvalue(ks_statistics),digits=2), ", ", HypothesisTests.pvalue(ks_statistics) < 0.05 ? "reject h_0 (normality)" : "fail to reject h_0 (normality)")

    return
end


function model_validation_tests(fitted_model::Any, lag_order::Int, residuals::Vector{Float64}, data_for_fitting::DataFrames.DataFrame)

    # (1) Goodness of fit
    println("> GOODNESS OF FIT")
    goodness_of_fit(fitted_model)

    # (2) Model significance
    println("> SIGNIFICANCE")
    significance_tests(fitted_model, lag_order)

    # (3) Autocorrelation of residuals
    println("> AUTOCORRELATION OF RESIDUALS")
    autocorrelation_tests(residuals, data_for_fitting)

    # (4) Heteroscedasticity of residuals
    heteroscedasticity_tests(residuals)

    # (5) Normal distribution of residuals
    println("> NORMAL DISTRIBUTION OF RESIDUALS")
    normality_tests(residuals)

end


function analyze_full_time_series(series::Vector{Float64}, detrended_series::Vector{Float64})
    
    println()
    println("AR(1) MODEL FOR FULL HORIZON")
    println("##############################################################################")

    println("STATIONARITY ANALYSIS")

    # Analyze the partial autocorrelation function
    #######################################################################################
    pacf = StatsBase.pacf(series, [i for i=1:75])
    pacf_detrended = StatsBase.pacf(detrended_series, [i for i=1:75])
   
    p1 = Plots.plot(Plots.bar(pacf))
    Plots.hline!([-0.2, 0.2], color=:black, lw=2, ls=:dash)
    p2 = Plots.plot(Plots.bar(pacf_detrended))
    Plots.hline!([-0.2, 0.2], color=:black, lw=2, ls=:dash)
    pfull = Plots.plot(p1, p2, ylims=(-1,1), legend=false)
    Plots.display(pfull)

    """These plots may give a hint on the required lag, even though we use a monthly model.
    The quick decline in PACF for the detrended data indicates stationarity."""

    # Perform a Dickey-Fuller test
    #######################################################################################
    adf = HypothesisTests.ADFTest(detrended_series, :none, 1)
    println("Augmented Dickey-Fuller test for detrended time series")
    println(adf)

    # Fit an AR process and analyze residuals
    #######################################################################################
    println("MODEL FITTING")
    # Fit an AR(1) process using the GLM package
    # Copy vector so that it is not permanently changed 
    Y = copy(detrended_series) # original data       
    X = copy(detrended_series) # data shifted by lag 1
    input_data = DataFrames.DataFrame(Y = deleteat!(Y, 1), X = deleteat!(X, 948))
    fitted_model = GLM.lm(GLM.@formula(Y~X),input_data)
    residuals = GLM.residuals(fitted_model)

    # Model validation tests
    #######################################################################################
    println("MODEL VALIDATION")
    model_validation_tests(fitted_model, 1, residuals, input_data)
        
    return
end

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

function plot_forecasts(
    df::DataFrames.DataFrame,
    )      

    plot_fc_1 = Plots.plot(df[:,:log_detrended],legend=false, color=:blue)
    Plots.plot!(df[:,:fc_log_detrended],color=:red, lw=2)
    # Plots.plot!(df[:,:fc_naive],color=:black)
    Plots.display(plot_fc_1)

    plot_fc_2 = Plots.plot(df[:,:log],legend=false,color=:blue)
    Plots.plot!(df[:,:fc_log],color=:red, lw=2)
    Plots.display(plot_fc_2)

    plot_fc_3 = Plots.plot(df[:,:orig],legend=false, color=:blue)
    Plots.plot!(df[:,:fc_orig],color=:red, lw=2)
    Plots.plot!(df[:,:fc_orig_corr],color=:green, lw=2)
    Plots.display(plot_fc_3)

    return
end


function point_forecast(
    all_monthly_models::Vector{MonthlyModelStorage},
    data_log_detrended::Vector{Float64},
    data_log::Vector{Float64},
    data_orig::Vector{Float64},
    detrending_with_sigma::Bool,
    regressor_symbol::Symbol,
)

    all_residuals = Float64[]

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
        coef = all_monthly_models[month].coefficients
        lag_order = all_monthly_models[month].lag_order
        μ = all_monthly_models[month].mean

        if detrending_with_sigma
            sigma = all_monthly_models[month].sigma
        else
            sigma = 1
        end

        if row > 12 && row <= (maximum(df[:, :year])-1)*12
            # Predicted value from model itself
            if all_monthly_models[month].status_flag == :not_full
                df[row, :predict] = all_monthly_models[month].prediction[df[row, :year]-1]
            else
                df[row, :predict] = all_monthly_models[month].prediction[df[row, :year]]
            end

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
            residual = df[row, :log_detrended] - df[row, :fc_log_detrended]
            push!(all_residuals, residual)

            # Remove deseasonalization
            df[row, :fc_log] = df[row, :fc_log_detrended] * sigma + μ

            # Get standard error of regression
            std_error = sqrt(sum(all_monthly_models[month].residuals .^2) / (length(all_monthly_models[month].residuals) - 2)) 
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
    plot_forecasts(df)

    # Computation of statistics (mean, std, MAE, RMSE)
    compute_forecast_statistics(df[:,:log_detrended], df[:,:fc_log_detrended], "Log-detrended")
    compute_forecast_statistics(df[:,:log_detrended], df[:,:fc_naive], "Naive")
    compute_forecast_statistics(df[:,:log], df[:,:fc_log], "Log")
    compute_forecast_statistics(df[:,:orig], df[:,:fc_orig], "Original")
    compute_forecast_statistics(df[:,:orig], df[:,:fc_orig_corr], "Original corrected")
    println()

    return all_residuals
end


function static_point_forecast(
    all_monthly_models::Vector{MonthlyModelStorage},
    data_log_detrended::Vector{Float64},
    data_log::Vector{Float64},
    data_orig::Vector{Float64},
    detrending_with_sigma::Bool,
)

    return point_forecast(all_monthly_models, data_log_detrended, data_log, data_orig, detrending_with_sigma, :log_detrended)
end

function dynamic_point_forecast(
    all_monthly_models::Vector{MonthlyModelStorage},
    data_log_detrended::Vector{Float64},
    data_log::Vector{Float64},
    data_orig::Vector{Float64},
    detrending_with_sigma::Bool,
)

    return point_forecast(all_monthly_models, data_log_detrended, data_log, data_orig, detrending_with_sigma, :fc_log_detrended)
end

function generate_full_scenarios(
    all_monthly_models::Vector{MonthlyModelStorage},
    starting_value::Vector{Float64},
    number_of_scenarios::Int,
    detrending_with_sigma::Bool,
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
            coef = all_monthly_models[month].coefficients
            lag_order = all_monthly_models[month].lag_order
            μ = all_monthly_models[month].mean

            if detrending_with_sigma
                sigma = all_monthly_models[month].sigma
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
            std_error = sqrt(sum(all_monthly_models[month].residuals .^2) / (length(all_monthly_models[month].residuals) - 2)) 

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

    return all_means, all_stds, yearly_means
end

function plot_scenario_statistics(
    all_means::DataFrames.DataFrame,
    all_stds::DataFrames.DataFrame,
    df::DataFrames.DataFrame
    )

    bx_plot_mean = StatsPlots.@df all_means StatsPlots.boxplot(cols(), legend=false, xticks=1:12)
    for col_name in names(df)
        Plots.scatter!([parse(Int, col_name)], [Statistics.mean(df[:,col_name])], color = "black", label = "", markersize = 5, markershape = :x)
    end
    Plots.display(bx_plot_mean)

    bx_plot_std = StatsPlots.@df all_stds StatsPlots.boxplot(cols(), legend=false, xticks=1:12)
    for col_name in names(df)
        Plots.scatter!([parse(Int, col_name)], [Statistics.std(df[:,col_name])], color = "black", label = "", markersize = 5, markershape = :x)
    end
    Plots.display(bx_plot_std)

    return
end

function plot_yearly_qq(
    yearly_means::Vector{Float64},
    historic_df::DataFrames.DataFrame,
    )

    # Get yearly_means from historic df
    historic_means = DataFrames.reduce(+, DataFrames.eachcol(historic_df)) ./ DataFrames.ncol(historic_df)

    # Make a quantile-quantile plot
    plot_qq = Plots.plot(StatsPlots.qqplot(historic_means, yearly_means, qqline = :identity))
    Plots.display(plot_qq)

    return
end


function prepare_model()

    training_test_split = false
    detrending_with_sigma = true
    use_bic_order = true

    for file_name in ["hist1.csv"] #, "hist1.csv", "hist2.csv", "hist3.csv", "hist4.csv"

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

        # Visualize data
        plot_hist = Plots.histogram(data_frame_to_vector(df), label="Empirical", color=:gray, legend=false, nbins=30)
        Plots.display(plot_hist)
        plot_hist = Plots.histogram(data_frame_to_vector(log_df), label="Empirical", color=:gray, legend=false, nbins=30)
        Plots.display(plot_hist)

        # Get deseasonalized logarithmized data
        detrended_log_df = detrend_data(log_df, detrending_with_sigma)

        # Split data into training data and test data if intended
        training_data, test_data = split_data(detrended_log_df, training_test_split, 0.82)     

        # ANALYZING AND MODELING THE FULL DATA (NON-PERIODIC MODEL)
        #######################################################################################
        analyze_full_time_series(data_frame_to_vector(log_df), data_frame_to_vector(detrended_log_df))
        
        # ANALYZING AND MODELING THE MONTHLY DATA (PAR MODEL)
        #######################################################################################
        all_monthly_models = MonthlyModelStorage[]  

        for month in 1:12
            # Get overview on data
            DataFrames.describe(training_data)

            # Model identification: Use BIC to identify best lag_order
            #######################################################################################
            best_lag_order = model_identification(training_data, month)
            #println(month, ",", best_lag_order)

            # Model fitting: Use identified lag_order (or default value 1) and fit a model
            #######################################################################################
            if use_bic_order
                lag_order = best_lag_order
            else
                lag_order = 1
            end

            println()
            println("PAR model for month ", month, " and lag order ", lag_order)
            println("--------------------------------------------------------------------------")

            println("MODEL FITTING")
            fitted_model, data_for_fitting, flag = model_fitting(training_data, month, lag_order)
            residuals = GLM.residuals(fitted_model)
            # studentized: residuals = GLM.residuals(fitted_model) / StatsBase.std(GLM.residuals(fitted_model))

            # print fitting results
            println(GLM.coeftable(fitted_model))

            # Model validation: Diagnostic tests
            #######################################################################################
            println("MODEL VALIDATION")
            model_validation_tests(fitted_model, lag_order, residuals, data_for_fitting)
            
            # Store model information for forecasts and usage in SDDP
            push!(all_monthly_models, MonthlyModelStorage(log_df[!, month], fitted_model, lag_order, flag))
        end

        # Model validation: Forecasts
        #######################################################################################
        println()
        train_data_log, test_data_log = split_data(log_df, training_test_split, 0.82)     
        train_data_orig, test_data_orig = split_data(df, training_test_split, 0.82)     

        # In-sample forecasts (on full data)
        # static_point_forecast(all_monthly_models, data_frame_to_vector(detrended_log_df), data_frame_to_vector(log_df), data_frame_to_vector(df), detrending_with_sigma)
        
        # In-sample forecasts (on training data)
        all_residuals_1 = static_point_forecast(all_monthly_models, data_frame_to_vector(training_data), data_frame_to_vector(train_data_log), data_frame_to_vector(train_data_orig), detrending_with_sigma)
        #dynamic_point_forecast(all_monthly_models, data_frame_to_vector(training_data), data_frame_to_vector(train_data_log), data_frame_to_vector(train_data_orig), detrending_with_sigma)
        
        # Out-of-sample forecasts (on test data)
        all_residuals_2 = static_point_forecast(all_monthly_models, data_frame_to_vector(test_data), data_frame_to_vector(test_data_log), data_frame_to_vector(test_data_orig), detrending_with_sigma)
        #dynamic_point_forecast(all_monthly_models, data_frame_to_vector(test_data), data_frame_to_vector(test_data_log), data_frame_to_vector(test_data_orig), detrending_with_sigma)
       
        # Model validation: Simulation / scenario generation
        #######################################################################################
        all_means, all_stds, yearly_means = generate_full_scenarios(all_monthly_models, [detrended_log_df[1,1], detrended_log_df[1,2]], 200, detrending_with_sigma)
        plot_scenario_statistics(all_means, all_stds, df)
        plot_yearly_qq(yearly_means, df)

        # Model output
        #######################################################################################
        println("Model output for ", file_name)
        for month in 1:12
            println("Month: ", month)
            println("Lag order: ", all_monthly_models[month].lag_order)
            println("Coefficients: ", all_monthly_models[month].coefficients)
            std_error = sqrt(sum(all_monthly_models[month].residuals .^2) / (length(all_monthly_models[month].residuals) - 2)) 
            println("Residual standard error: ", std_error)
            println()
        end
        println("#############################################")
        Infiltrator.@infiltrate

    end

end

prepare_model()