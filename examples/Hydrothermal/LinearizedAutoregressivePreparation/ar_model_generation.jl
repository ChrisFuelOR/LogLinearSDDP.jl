import Distributions
import DataFrames
import Random

""" Writes the output of the model to the REPL and to a txt file.
Corrects the coefficients so that they fit SDDP before."""
function model_output(
    all_monthly_models::Vector{MonthlyModelStorage},
    std_errors::Vector{Float64},
    output_file_name::String,
)

    f = open(output_file_name, "w")

    for month in 1:12
        # Get model parameters (note that we ignore the very small intercepts here, to stay in accordance with Shapiro et al.)
        lag_order = all_monthly_models[month].lag_order
        coefficient = GLM.coef(all_monthly_models[month].fitted_model)[2]
        std_error = std_errors[month]
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
        println("Residual standard error: ", std_error)
        println("-----------------------------------------")  
        println("Corrected coefficients: ", coefficient_corrected)
        println()  

        # Output to file
        println(f, month, ";", lag_order, ";", coefficient, ";", coefficient_corrected, ";", std_error)

    end

    println("#############################################")     
    close(f)   

end

""" Writes the output of a "cross model" to a txt file.
Cross model means that the same coefficients as for the log-linear process are used for a linearized process.
"""
function model_cross_output(
    all_monthly_models::Vector{MonthlyModelStorage},
    std_errors::Vector{Float64},
    output_file_name::String,
)
    f = open(output_file_name, "w")
    psi = Vector{Float64}()

    for month in 1:12
        # Get model parameters (note that we ignore the very small intercepts here, to stay in accordance with Shapiro et al.)
        lag_order = all_monthly_models[month].lag_order
        coefficient = GLM.coef(all_monthly_models[month].fitted_model)[2]
        std_error = std_errors[month]
        monthly_mean = all_monthly_models[month].detrending_mean

        # Get lag month and corresponding max_gen
        lag_month = month - 1 > 0 ? month - 1 : Int(12 - mod((1 - month), 12))
        lag_mean = all_monthly_models[lag_month].detrending_mean 

        # Compute coefficient for SDDP
        coefficient_loglin = Vector{Float64}(undef, 1)
        coefficient_loglin[1] = coefficient

        # Compute intercept for SDDP
        intercept = monthly_mean - coefficient * lag_mean

        # Output to file
        # Note that we store Psi (error_factor) in place of coefficient
        println(f, month, ";", lag_order, ";", intercept, ";", coefficient_loglin, ";", 1.0, ";", std_error)
    end

    println("#############################################")     
    close(f)   

    return
end


""" Analysis, fitting and validation of the linearized PAR model (with respect to the original data).
This approach goes back to a paper by Shapiro et al. (2013). The idea is to approximate the true
nonlinear AR(1) process with a first-order Taylor approximation to obtain a linear AR(1) model,
but with multiplicative error terms. The multiplicative error exp(eta) is log-normally distributed with
parameters (1,sigma) or (0, log(sigma)), depending on the definition.

There are some important things to note here, especially in comparison to our preparation of the 
nonlinear model.

(1) As the errors are multiplicative, it is not immediately clear that a traditional OLS estimator
will yield the best fit in the sense of minimizing the multiplicative residuals.
Still, we use OLS to fit the linearized models in accordance with the approach by Shapiro et al.
who follow this procedure.

(2) Shapiro et al. identify a lag order of 1 for the autoregressive model based on an analysis of 
the original time series. This does neither imply that a lag order of 1 is the most appropriately
for each individual month in a PAR model, nor that it is the best choice for the actually fitted
linearized model. As the linearized model (based on Taylor approximation) is only introduced
for AR(1) processes in their paper, we follow their approach and stick to a fixed lag order of 1,
though. Hence, in contrast to the nonlinear (log-linear) model, we do not perform an extensive
model identification step.

(3) We do not apply a detrending using the standard deviation here (as it is not required
for the way we fit the model here).

(4) In contrast to Shapiro et al., we use the standard errors of the monthly residuals to
define the Normal distribution of the error terms. Shapiro et al. use a multivariate Normal
distribution based on the sample covariance matrix.
"""
function prepare_ar_model()

    Random.seed!(12345)

    # PARAMETER CONFIGURATION
    training_test_split = true
    with_plots = false
    
    # FILE PATH COMPONENTS
    directory_name = "historical_data"
    file_names = ["hist1.csv", "hist2.csv", "hist3.csv", "hist4.csv"]
    system_names = ["SE", "S", "NE", "N"]
    output_directory = "fitted_model"

    # ITERATE OVER POWER SYSTEMS AND PREPARE AUTOREGRESSIVE MODEL
    for system_number in 1:4
        system_name = system_names[system_number]
        file_name = directory_name * "/" * file_names[system_number]
        output_file_name = output_directory * "/model_lin_" * String(system_name) * ".txt"   

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

        # Visualize data and logarithmized data
        if with_plots
            histogram_plot(df)
            histogram_plot(log_df)
        end

        # Get deseasonalized logarithmized data
        μ, σ, detrended_log_df = detrend_data(log_df, false, with_plots)

        # Get adapted data for linearization fitting
        adapted_df = adapt_data(df, μ)

        # Split data into training data and test data if intended
        train_data, test_data = split_data(detrended_log_df, training_test_split, 0.82)
        train_data_log, test_data_log = split_data(log_df, training_test_split, 0.82)
        train_data_orig, test_data_orig = split_data(df, training_test_split, 0.82)     
        train_data_adapt, test_data_adapt = split_data(adapted_df, training_test_split, 0.82)

        # ANALYZING AND MODELING THE FULL DATA (ASSUMING A NON-PERIODIC MODEL)
        #######################################################################################
        box_jenkins_full_time_series(data_frame_to_vector(train_data_log), data_frame_to_vector(train_data), false)
        
        # ANALYZING AND MODELING THE MONTHLY DATA (PERIODIC MODEL)
        #######################################################################################
        acv_df = compute_sample_autocovariance(train_data_adapt)
        # NOTE: 
        # (1) Importantly, here the residuals should be computed as multiplicative deviations!
        # (2) To stay in accordance with Shapiro et al., we ignore the intercepts (which are negligibly small). 
        # It would be better to fit models without intercepts then, but GLM forecasting in Julia relies on models with intercepts.
        all_monthly_models, std_errors = box_jenkins_periodic(train_data_adapt, train_data_orig, acv_df, μ, σ, false)

        # MODEL VALIDATION: FORECASTS
        #######################################################################################
        println()
        # In-sample forecasts (on full data, training data, test data)
        static_point_forecast(all_monthly_models, data_frame_to_vector(train_data_orig), with_plots)
        static_point_forecast(all_monthly_models, data_frame_to_vector(test_data_orig), with_plots)
       
        # MODEL VALIDATION: SIMULATION
        #######################################################################################
        generate_full_scenarios(system_number, df, all_monthly_models, [df[1,1]], std_errors, 1000, true)

        # COEFFICIENT REFORMULATION AND MODEL OUTPUT
        #######################################################################################
        println("Model output for ", file_name)
        model_output(all_monthly_models, std_errors, output_file_name)

        # OUTPUT FOR CROSS MODEL
        #######################################################################################
        # model_cross_output(all_monthly_models, std_errors, "../AutoregressivePreparation/cross_log_model/model_" * system_name * ".txt")

    end

    # # Write realizations of linear model to loglinear cross model
    # realization_df = read_realization_data(output_directory * "/scenarios_linear.txt")
    # file_name = "../AutoregressivePreparation/cross_log_model/" * "scenarios_nonlinear.txt"
    # f = open(file_name, "w")

    # for row in DataFrames.eachrow(realization_df)
    #     month = mod(row["Stage"], 12) > 0 ? mod(row["Stage"],12) : 12
    #     println(f, row["Stage"], ";", row["Realization_number"], ";", row["Probability"], ";", row["Realization_SE"], ";", row["Realization_S"],";", row["Realization_NE"],";", row["Realization_N"])
    # end
    # println(f)
end

