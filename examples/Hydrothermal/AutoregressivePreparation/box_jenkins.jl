""" Prints the adjusted R^2 value as a measure of goodness of fit for a given fitted model."""
function goodness_of_fit(fitted_model::Any)
    println("adjusted R²: ", GLM.adjr2(fitted_model))
    return
end

""" Prints the signifance results for a given fitted model.
For lag order 1, the t-test is used. For lag order > 1, only the F-test is used."""
function significance_tests(fitted_model::Any, lag_order::Int)
    if lag_order == 1
        # t-test statistics and p-value for 5% significance level
        alpha = 0.05
        significant = true
        dof = length(GLM.fitted(fitted_model)) - lag_order - 1
        t_critical = Distributions.quantile(Distributions.TDist(dof), 1-alpha)

        if !(GLM.coeftable(fitted_model).cols[3][2] > t_critical && GLM.coeftable(fitted_model).cols[4][2] < alpha)
            significant = false
        end
        println("p-value and t-test imply significance: ", significant)            
    else
        # F-test
        println("F-test: ", GLM.ftest(fitted_model.model))
    end

    return
end

""" Makes a scatter plot of the residuals of a fitted model to assess heteroscedasticity."""
function heteroscedasticity_tests(residuals::Vector{Float64}, with_plots::Bool)
    if with_plots
       scatter_plot(residuals)
    end
    return
end

""" Analysis and tests of residuals for autocorrelation. """
function full_autocorrelation_tests(residuals::Vector{Float64}, data_for_fitting::DataFrames.DataFrame, with_plots::Bool)
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

    # Breusch-Godfrey test for residuals
    ###########################################################################################################
    # Breusch-Godfrey test for max_lag_order
    regressor_matrix = Matrix(DataFrames.select!(data_for_fitting, DataFrames.Not([:Y])))
 
    # Analyze p-value of Breusch-Godfrey test for different lags
    bg_p_values = Float64[]
    for lag in 1:24
        bg_statistics = HypothesisTests.BreuschGodfreyTest(regressor_matrix, residuals, lag, false)
        println("Breusch-Godfrey test for lag: ", lag, ", p-value = ", round(HypothesisTests.pvalue(bg_statistics),digits=3), ", ", HypothesisTests.pvalue(bg_statistics) < 0.05 ? "reject h_0 (no autocorrelation)" : "fail to reject h_0 (no autocorrelation)")
        push!(bg_p_values, HypothesisTests.pvalue(bg_statistics))
    end

    if with_plots
        time_series_plot(bg_p_values)
    end

    return
end

""" Execute different tests to test for Normal distribution of residuals."""
function normality_tests(residuals::Vector{Float64}, with_plots::Bool)

    if with_plots
        # Plot 1: Normalized histogram of residuals
        plot_hist = Plots.histogram(residuals, color=:gray)
        Plots.display(plot_hist)

        # Plot 2: Quantile-quantile plot of residuals 
        plot_qq = Plots.plot(StatsPlots.qqplot(Distributions.Normal(0,StatsBase.std(residuals)), residuals, qqline = :R))
        # plot_qq = Plots.plot(StatsPlots.qqnorm(residuals, qqline = :R))
        Plots.display(plot_qq)
    end

    # Jarque-Bera test
    jb_statistics = HypothesisTests.JarqueBeraTest(residuals, adjusted=false)
    println("Jarque-Bera test: ", "p-value = ", round(HypothesisTests.pvalue(jb_statistics),digits=2), ", ", HypothesisTests.pvalue(jb_statistics) < 0.05 ? "reject h_0 (normality)" : "fail to reject h_0 (normality)")

    alm_statistics = HypothesisTests.JarqueBeraTest(residuals, adjusted=true)
    println("Adjusted Lagrangian Multiplier test: ", "p-value = ", round(HypothesisTests.pvalue(alm_statistics),digits=2), ", ", HypothesisTests.pvalue(alm_statistics) < 0.05 ? "reject h_0 (normality)" : "fail to reject h_0 (normality)")

    # Anderson-Darling test
    ad_statistics = HypothesisTests.OneSampleADTest(residuals, Distributions.Normal(0, StatsBase.std(residuals)))
    println("Anderson-Darling test: ", "p-value = ", round(HypothesisTests.pvalue(ad_statistics),digits=2), ", ", HypothesisTests.pvalue(ad_statistics) < 0.05 ? "reject h_0 (normality)" : "fail to reject h_0 (normality)")

    # Kolmogorov-Smirnov test
    ks_statistics = HypothesisTests.ExactOneSampleKSTest(residuals, Distributions.Normal(0, StatsBase.std(residuals)))
    println("Kolmogorov-Smirnov test: ", "p-value = ", round(HypothesisTests.pvalue(ks_statistics),digits=2), ", ", HypothesisTests.pvalue(ks_statistics) < 0.05 ? "reject h_0 (normality)" : "fail to reject h_0 (normality)")

    return
end

""" Execute validation tests, especially for residuals."""
function full_model_validation_tests(fitted_model::Any, lag_order::Int, residuals::Vector{Float64}, data_for_fitting::DataFrames.DataFrame, with_plots::Bool)

    # (1) Goodness of fit
    println("> GOODNESS OF FIT")
    goodness_of_fit(fitted_model)

    # (2) Model significance
    println("> SIGNIFICANCE")
    significance_tests(fitted_model, lag_order)

    # (3) Autocorrelation of residuals
    println("> AUTOCORRELATION OF RESIDUALS")
    full_autocorrelation_tests(residuals, data_for_fitting, with_plots)

    # (4) Heteroscedasticity of residuals
    heteroscedasticity_tests(residuals, with_plots)

    # (5) Normal distribution of residuals
    println("> NORMAL DISTRIBUTION OF RESIDUALS")
    normality_tests(residuals, with_plots)

    return
end

""" Perform the box-jenkins method for the full time series."""
function box_jenkins_full_time_series(series::Vector{Float64}, detrended_series::Vector{Float64}, with_plots::Bool)
    
    println()
    println("AR(1) MODEL FOR FULL HORIZON")
    println("##############################################################################")

    println("STATIONARITY ANALYSIS")

    # Analyze the autocorrelation function
    #######################################################################################
    # The maximum number of lags to be considered is number of observations / 4, according to Box & Jenkins.
    max_lag = Int(floor(length(series) / 4))
    acf = StatsBase.autocor(series, [i for i=1:max_lag])
    acf_detrended = StatsBase.autocor(detrended_series, [i for i=1:max_lag])
    
    if with_plots
        pacf_comparison_plot(acf, acf_detrended, length(series))
    end

    # Analyze the partial autocorrelation function
    #######################################################################################
    # The maximum number of lags to be considered is number of observations / 4, according to Box & Jenkins.
    max_lag = Int(floor(length(series) / 4))
    pacf = StatsBase.pacf(series, [i for i=1:max_lag])
    pacf_detrended = StatsBase.pacf(detrended_series, [i for i=1:max_lag])
    
    if with_plots
        pacf_comparison_plot(pacf, pacf_detrended, length(series))
    end

    # Perform an augmented Dickey-Fuller test
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
    input_data = DataFrames.DataFrame(Y = deleteat!(Y, 1), X = deleteat!(X, length(X)))
    fitted_model = GLM.lm(GLM.@formula(Y~X),input_data)
    residuals = GLM.residuals(fitted_model)

    # Model validation tests
    #######################################################################################
    println("MODEL VALIDATION")
    full_model_validation_tests(fitted_model, 1, residuals, input_data, with_plots)

    return
end