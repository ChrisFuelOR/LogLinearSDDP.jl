""" Run analysis, fitting and validation of PAR model."""
function prepare_ar_model()

    # Set parameter configuration
    training_test_split = true
    detrending_with_sigma = true
    identification_method = :bic
    with_plots = false

    file_names = ["hist1.csv", "hist2.csv", "hist3.csv", "hist4.csv"]
    system_names = ["SE", "S", "NE", "N"]

    for system_number in 1:4
        system_name = system_names[system_number]
        file_name = file_names[system_number] 
        output_file_name = "model_" * String(system_name) * ".txt"   

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
        μ, σ, detrended_log_df = detrend_data(log_df, detrending_with_sigma, with_plots)

        # Split data into training data and test data if intended
        train_data, test_data = split_data(detrended_log_df, training_test_split, 0.82)     
        train_data_log, test_data_log = split_data(log_df, training_test_split, 0.82)     
        train_data_orig, test_data_orig = split_data(df, training_test_split, 0.82)     

        # ANALYZING AND MODELING THE FULL DATA (ASSUMING A NON-PERIODIC MODEL)
        #######################################################################################
        box_jenkins_full_time_series(data_frame_to_vector(train_data_log), data_frame_to_vector(train_data), with_plots)
        
        # ANALYZING AND MODELING THE MONTHLY DATA (PERIODIC MODEL)
        #######################################################################################
        acv_df = compute_sample_autocovariance(train_data)
        all_monthly_models = box_jenkins_periodic(train_data, acv_df, μ, σ, identification_method, with_plots)

        # MODEL VALIDATION: FORECASTS
        #######################################################################################
        println()
        # In-sample forecasts (on full data, training data, test data)
        # static_point_forecast(all_monthly_models, data_frame_to_vector(detrended_log_df), data_frame_to_vector(log_df), data_frame_to_vector(df), detrending_with_sigma)   
        static_point_forecast(all_monthly_models, data_frame_to_vector(train_data), data_frame_to_vector(train_data_log), data_frame_to_vector(train_data_orig), detrending_with_sigma, with_plots)
        static_point_forecast(all_monthly_models, data_frame_to_vector(test_data), data_frame_to_vector(test_data_log), data_frame_to_vector(test_data_orig), detrending_with_sigma, with_plots)
       
        # MODEL VALIDATION: SIMULATION
        #######################################################################################
        generate_full_scenarios(df, all_monthly_models, [detrended_log_df[1,1], detrended_log_df[1,2], detrended_log_df[1,3], detrended_log_df[1,4], detrended_log_df[1,5], detrended_log_df[1,6], detrended_log_df[1,7], detrended_log_df[1,8], detrended_log_df[1,9], detrended_log_df[1,10]], 200, detrending_with_sigma, with_plots)

        # COEFFICIENT REFORMULATION AND MODEL OUTPUT
        #######################################################################################
        """ Note that we use (1) the exponential form of the log-linear process in our version of SDDP and (2) the non-detrended form,
        meaning that the model is formulated with respect to the original dependent variable.
        To apply our fitted model in SDDP, we therefore have to adapt the coefficients appropriately."""

        println("Model output for ", file_name)
        f = open(output_file_name, "w")

        for month in 1:12
            # Get model parameters
            lag_order = all_monthly_models[month].lag_order
            coefficients = GLM.coef(all_monthly_models[month].fitted_model)
            residuals = GLM.residuals(all_monthly_models[month].fitted_model)
            std_error = sqrt(sum(residuals .^2) / (length(residuals) - 2)) 
            monthly_mean = all_monthly_models[month].detrending_mean
            monthly_std = all_monthly_models[month].detrending_sigma
    
            # Compute corrected coefficients for SDDP
            coefficient_corrected = Vector{Float64}(undef, length(coefficients)-1)
            for k in 1:lag_order
                lag_month = month - k > 0 ? month - k : Int(12 - mod((k - month), 12))
                coefficient_corrected[k] = coefficients[k+1] * monthly_std / all_monthly_models[lag_month].detrending_sigma
            end

            intercept_corrected =  monthly_mean + coefficients[1] * monthly_std
            for k in 1:lag_order
                lag_month = month - k > 0 ? month - k : Int(12 - mod((k - month), 12))
                intercept_corrected -= coefficient_corrected[k] * all_monthly_models[lag_month].detrending_mean
            end
            
            error_factor = all_monthly_models[month].detrending_sigma
 
            # Output to console
            println("Month: ", month)
            println("Lag order: ", lag_order)
            println("Coefficients: ", coefficients)
            println("Residual standard error: ", std_error)
            println("-----------------------------------------")  
            println("Corrected intercept: ", intercept_corrected)
            println("Corrected coefficients: ", coefficient_corrected)
            println("Error term factor: ", error_factor)
            println()  

            # Output to file
            println(f, month, ";", lag_order, ";", intercept_corrected, ";", coefficient_corrected, ";", error_factor, ";", std_error)
        end

        println("#############################################")     
        close(f)   
    end

end

