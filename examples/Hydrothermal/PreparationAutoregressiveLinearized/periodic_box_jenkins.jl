""" Analysis and tests of residuals for autocorrelation. """
function periodic_autocorrelation_tests(
    month::Int64,
    all_residuals::DataFrames.DataFrame,
    monthly_model::MonthlyModelStorage,
    with_plots::Bool
    )

    monthly_residuals = all_residuals[:, month]   
    years = length(monthly_residuals)
    
    # Plot the standardized monthly residuals
    ###########################################################################################################
    if with_plots
        time_series_plot(monthly_residuals/Statistics.std(monthly_residuals))
    end

    # Analyze sample periodic residual ACF (RACF)
    """ Note that we do not consider the periodic residual PACF here."""
    ###########################################################################################################
    # The maximum number of lags to be considered is number of observations / 4, according to Box & Jenkins.
    max_lag = Int(floor(length(monthly_residuals) / 4))
    periodic_racf = Float64[]

    for lag in 1:max_lag
        # Get correct month corresponding to lag
        lag_month = month - lag > 0 ? month - lag : Int(12 - mod((lag - month), 12))
        lag_year_offset = month - lag > 0 ? 0 : Int(ceil((lag - month + 1)/12))

        # Get required standard deviations of the residuals
        std_month = Statistics.std(all_residuals[:, month])
        std_lag_month = Statistics.std(all_residuals[:, lag_month])

        # Compute sample periodic RACF
        periodic_rac = 1/((years-lag_year_offset) * std_month * std_lag_month) * sum(all_residuals[year,month] * all_residuals[year-lag_year_offset, lag_month] for year in 1+lag_year_offset:years)
        push!(periodic_racf, periodic_rac)
    end

    if with_plots
        pacf_plot(periodic_racf[1:max_lag], [1/sqrt(years) for i in 1:max_lag])
    end

    # NOTE: No Breusch-Godfrey test, since it is not clear which regressor matrix to use

    # Simple Portmanteau test from Hipel & McLeod (small sample properties have to be taken carefully)
    ###########################################################################################################
    test_statistic = years * sum(periodic_racf[k]^2 for k in 1:max_lag)
    dof = max_lag - monthly_model.lag_order
    p_value = HypothesisTests.pvalue(Distributions.Chisq(dof), test_statistic; tail=:right)
    println("Portmanteau test for lag: ", max_lag, ", p-value = ", round(p_value,digits=3), ", ", p_value < 0.05 ? "reject h_0 (no autocorrelation)" : "fail to reject h_0 (no autocorrelation)")

    test_statistic = years * sum(periodic_racf[k]^2 for k in 1:12)
    dof = 12 - monthly_model.lag_order
    p_value = HypothesisTests.pvalue(Distributions.Chisq(12 - monthly_model.lag_order), test_statistic; tail=:right)
    println("Portmanteau test for lag: ", 12, ", p-value = ", round(p_value,digits=3), ", ", p_value < 0.05 ? "reject h_0 (no autocorrelation)" : "fail to reject h_0 (no autocorrelation)")

    return test_statistic, dof
end

""" Execute validation tests, especially for residuals."""
function periodic_model_validation_tests(
    all_monthly_models::Vector{MonthlyModelStorage},
    all_residuals::DataFrames.DataFrame,
    with_plots::Bool
    )

    overall_test_statistic = 0.0
    overall_dof = 0

    for month in 1:12
        println("MONTH: ", month)

        # Get monthly model
        monthly_model = all_monthly_models[month]

        # (1) Goodness of fit
        println("> GOODNESS OF FIT")
        goodness_of_fit(monthly_model.fitted_model)

        # (2) Model significance
        println("> SIGNIFICANCE")
        significance_tests(monthly_model.fitted_model, monthly_model.lag_order)

        # (3) Autocorrelation of residuals
        println("> AUTOCORRELATION OF RESIDUALS")
        test_statistic, dof = periodic_autocorrelation_tests(month, all_residuals, monthly_model, with_plots)
        overall_test_statistic += test_statistic
        overall_dof += dof

        # (4) Heteroscedasticity of residuals
        heteroscedasticity_tests(all_residuals[:, Symbol(month)], with_plots)

        # (5) Normal distribution of residuals
        println("> NORMAL DISTRIBUTION OF RESIDUALS")
        normality_tests(all_residuals[:, Symbol(month)], with_plots)

        println()
    end

    # Portmanteau test over all periods
    println("OVERALL: ")
    p_value = HypothesisTests.pvalue(Distributions.Chisq(overall_dof), overall_test_statistic; tail=:right)
    println("Portmanteau test for lag: ", 12, ", p-value = ", round(p_value,digits=3), ", ", p_value < 0.05 ? "reject h_0 (no autocorrelation)" : "fail to reject h_0 (no autocorrelation)")

    return
end

function autocorrelation_tests(residuals::Vector{Float64}, with_plots::Bool)
    # Plot the standardized residuals
    ###########################################################################################################
    if with_plots
        time_series_plot(residuals/Statistics.std(residuals))
    end

    # Analyze ACF/PACF of the residuals
    ###########################################################################################################
    # The maximum number of lags to be considered is number of observations / 4, according to Box & Jenkins.
    max_lag = Int(floor(length(residuals) / 4))
    acf = StatsBase.autocor(residuals, [i for i=1:max_lag])
    pacf = StatsBase.pacf(residuals, [i for i=1:max_lag])

    if with_plots
        pacf_plot(acf[1:max_lag], [1/sqrt(length(residuals)) for i in 1:max_lag])
        pacf_plot(pacf[1:max_lag], [1/sqrt(length(residuals)) for i in 1:max_lag])
    end

    # NOTE: No Breusch-Godfrey test, since it is not clear which regressor matrix to use
    return
end

""" Execute validation tests, especially for residuals."""
function model_validation_tests(
    all_residuals::Vector{Float64},
    with_plots::Bool
    )

    # (1) Autocorrelation of residuals
    println("> AUTOCORRELATION OF RESIDUALS")
    autocorrelation_tests(all_residuals, with_plots)

    # (2) Heteroscedasticity of residuals
    heteroscedasticity_tests(all_residuals, with_plots)

    # (3) Normal distribution of residuals
    println("> NORMAL DISTRIBUTION OF RESIDUALS")
    normality_tests(all_residuals, with_plots)

    return
end

""" Fit a monthly model as part of a PAR model, given a month and a lag order."""
function model_fitting(df::DataFrames.DataFrame, month::Int64, lag_order::Int64)

    # Get data for regressand
    Y = copy(df[!, month])
    
    # Prepare data (if not all lags are in same year, delete offset number of years from Y)
    year_offset = 0
    if month - lag_order <= 0
        year_offset = Int(ceil((lag_order - month + 1)/12))
        Y = deleteat!(Y, [i for i in 1:year_offset])
    end
    
    # Construct DataFrame for fitting
    df_for_fitting = DataFrames.DataFrame()
    DataFrames.insertcols!(df_for_fitting, :Y => Y)

    # Iterate over lags
    for lag in 1:lag_order
        # Get correct month corresponding to lag
        lag_month = month - lag > 0 ? month - lag : Int(12 - mod((lag - month), 12))
        lag_year_offset = month - lag > 0 ? 0 : Int(ceil((lag - month + 1)/12))
        @assert lag_year_offset <= year_offset

        # Get regressor time series
        X = copy(df[!, lag_month]) 

        # Prepare data (if not all lags are in same year, delete first or last year for Y)
        if year_offset > 0
            last_elements = [length(X)-i for i in lag_year_offset:-1:1]
            first_elements = [i for i in 1:year_offset-lag_year_offset]
            X = deleteat!(X, [first_elements; last_elements])
        end

        #println(month, ", ", lag, ", ", lag_month, ", ", lag_year_offset, ", ", year_offset)

        # Create column for lag data
        DataFrames.insertcols!(df_for_fitting, Symbol(lag) => X)
    end

    # Fitting the model using OLS (include all columns in df_for_fitting as regressors, but the one called :Y)
    ols_result = GLM.lm(GLM.term(:Y)~sum(GLM.term.(names(df_for_fitting[:,Not(:Y)]))), df_for_fitting)
    return ols_result, df_for_fitting, year_offset
end

""" Model identification - Compute and plot the periodic ACF for each month and lag."""
function model_identification_periodic_acf(acv_df::DataFrames.DataFrame, max_lag::Int64, years::Int64, with_plots::Bool)
    
    for month in 1:12
        periodic_acf = Float64[]
        standard_errors = Float64[]
        
        for lag in 1:max_lag
            # Identify the correct month for the given lag
            lag_month = month - lag > 0 ? month - lag : Int(12 - mod((lag - month), 12))

            # Compute the periodic ACF
            periodic_autocor = acv_df[month, Symbol(lag)] / sqrt(acv_df[month, Symbol(0)] * acv_df[lag_month, Symbol(0)])
            push!(periodic_acf, periodic_autocor)

            # Compute the standard error (for model identification; Bartlett's formula)
            standard_error = 1/sqrt(years) * sqrt(1 + 2 * sum(periodic_acf[i]^2 for i in 1:lag))
            push!(standard_errors, standard_error)
        end

        # Plot the periodic ACF if required
        if with_plots
            pacf_plot(periodic_acf, standard_errors)
        end
    end
end

""" Compute residuals for the linearized model with multiplicative errors."""
function get_residuals_linearized_model(
    all_monthly_models::Vector{MonthlyModelStorage},
    df::DataFrames.DataFrame,
    )

    # Create dataframe for residuals
    all_residuals = DataFrames.DataFrame()
    all_residuals[:, :row_number] = 1:size(df,1)

    for month in 1:12
        DataFrames.insertcols!(all_residuals, Symbol(month) => 0.0)

        # Get correct values for μ and AR coefficient
        μ = all_monthly_models[month].detrending_mean
        coef = GLM.coef(all_monthly_models[month].fitted_model)[2]

        # Get correct month corresponding to lag = 1
        lag_month = month - 1 > 0 ? month - 1 : Int(12 - mod((1 - month), 12))
        lag_year_offset = month - 1 > 0 ? 0 : Int(ceil((1 - month + 1)/12))
        μ_lag = all_monthly_models[lag_month].detrending_mean

        for i in 1+lag_year_offset:DataFrames.nrow(df)
            # Compute residuals
            denom = exp(μ) + coef * exp(μ - μ_lag) * (df[i-lag_year_offset, Symbol(lag_month)] - exp(μ_lag))
            nom = df[i, Symbol(month)]
            all_residuals[i, Symbol(month)] = log(nom/denom)
        end
    end    

    all_residuals = DataFrames.select!(all_residuals, DataFrames.Not([:row_number]))

    return all_residuals
end



""" Perform the box-jenkins method using periodic models.
Note that according to the paper by Shapiro et al., we can fit a model using OLS for data_adapt
to determine the coefficients that are required in the linearized model for the original data.
"""
function box_jenkins_periodic(
    data_adapt::DataFrames.DataFrame,
    data_orig::DataFrames.DataFrame,
    acv_df::DataFrames.DataFrame,
    μ::Vector{Float64},
    σ::Vector{Float64},
    with_plots::Bool
    )

    # Setting parameters
    years = DataFrames.nrow(data_adapt)
    max_lag = Int(floor(years/4)) #12
    
    all_monthly_models = MonthlyModelStorage[]  
       
    # Get overview on data
    DataFrames.describe(data_adapt)
    
    # Model identification
    #######################################################################################     
    # Compute and plot periodic ACF
    model_identification_periodic_acf(acv_df, max_lag, years, with_plots)
   
    # Model fitting: Use lag order 1 and fit a model
    #######################################################################################
    for month in 1:12
        lag_order = 1

        println()
        println("Linearized PAR model for month ", month, " and lag order ", lag_order)
        println("--------------------------------------------------------------------------")

        println("MODEL FITTING / COMPUTE COEFFICIENTS BY AUXILIARY REGRESSION")
        fitted_model, data_for_fitting, year_offset = model_fitting(data_adapt, month, lag_order)

        # Store model information for validation, forecasts and usage in SDDP
        push!(all_monthly_models, MonthlyModelStorage(μ[month], σ[month], fitted_model, lag_order, year_offset))

        # print fitting results
        println(GLM.coeftable(fitted_model))     
    end

    # Model validation: Diagnostic tests
    #######################################################################################
    # Compute residuals
    all_residuals = get_residuals_linearized_model(all_monthly_models, data_orig)
    std_errors = Statistics.std.(eachcol(all_residuals))

    println("MODEL VALIDATION - PERIODIC")
    periodic_model_validation_tests(all_monthly_models, all_residuals, with_plots)
    
    println()
    println("MODEL VALIDATION - ACROSS ALL PERIODS")
    model_validation_tests(data_frame_to_vector(all_residuals), with_plots)

    return all_monthly_models, std_errors
end