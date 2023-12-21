import LogLinearSDDP

struct LinearAutoregressiveProcessStage
    dimension::Int64
    coefficients::Array{Float64,2}
    eta::Vector{Any}
    probabilities::Vector{Float64}

    function LinearAutoregressiveProcessStage(
        dimension,
        coefficients,
        eta;
        probabilities = fill(1 / length(eta), length(eta)),
    )
        return new(
            dimension,
            coefficients,
            eta,
            probabilities,
        )
    end
end

struct LinearAutoregressiveProcess
    lag_order::Int64
    parameters::Dict{Int64,LinearAutoregressiveProcessStage}
    history::Vector{Float64}
end


""" Reads stored model data."""
function read_model(file_name)
    f = open(file_name)
    df = CSV.read(f, DataFrames.DataFrame, header=false, delim=";")
    DataFrames.rename!(df, ["Month", "Lag_order", "Intercept", "Coefficients", "Psi", "Sigma"])    
    close(f)
    return df
end

""" Reads stored model data."""
function read_model_linear(file_name)
    f = open(file_name)
    df = CSV.read(f, DataFrames.DataFrame, header=false, delim=";")
    DataFrames.rename!(df, ["Month", "Lag_order", "Coefficient", "Corr_coefficients", "Sigma"])    
    close(f)
    return df
end

""" Read stored realization data."""
function read_realization_data(overall_directory::String, model_directory::String, file_name::String)
    f = open(overall_directory * "/" * model_directory * "/" * file_name)
    
    df = CSV.read(f, DataFrames.DataFrame, header=false, delim=";")
    DataFrames.rename!(df, ["Stage", "Realization_number", "Probability", "Realization_SE", "Realization_S", "Realization_NE", "Realization_N"])    
    close(f)
    return df
end

""" Get the realization data for a specific stage and system."""

function get_realization_data(eta_df::DataFrames.DataFrame, t::Int64, number_of_realizations::Int64)
    realizations = Tuple{Float64, Float64, Float64, Float64}[]

    for i in 1:number_of_realizations
        row = DataFramesMeta.@rsubset(eta_df, :Stage == t, :Realization_number == i)
        @assert DataFrames.nrow(row) == 1
        push!(realizations, (row[1, "Realization_SE"], row[1, "Realization_S"], row[1, "Realization_NE"], row[1, "Realization_N"]))
    end

    return realizations
end

""" Read stored history data for the process."""
function read_history_data(file_name::String)
    f = open(file_name)
    df = CSV.read(f, DataFrames.DataFrame, header=false, delim=";")
    DataFrames.rename!(df, ["Stage", "History_SE", "History_S", "History_NE", "History_N"])    
    close(f)
    return df
end


function set_up_ar_process_loglinear(number_of_stages::Int64, number_of_realizations::Int64, model_directory::String, history_file_name::String)

    # Main configuration
    # ---------------------------------------------------------------------------------------------------------
    # File information
    overall_directory = "AutoregressivePreparation"

    # Read AR model data for all four reservoir systems
    data_SE = read_model(overall_directory * "/" * model_directory * "/model_SE.txt")
    data_S = read_model(overall_directory * "/" * model_directory * "/model_S.txt")
    data_NE = read_model(overall_directory * "/" * model_directory * "/model_NE.txt")
    data_N = read_model(overall_directory * "/" * model_directory * "/model_N.txt")
    data = [data_SE, data_S, data_NE, data_N]
    dim = 4 # number of hydro reservoirs

    # Read realization data
    eta_df = read_realization_data(overall_directory, model_directory, "scenarios_nonlinear.txt")

    # Get max lag order (which is used as constant lag order for the process)
    lag_order = 0
    for df in data
        for month in 1:12
            current_lag_order = df[month, "Lag_order"]
            if current_lag_order > lag_order
                lag_order = current_lag_order
            end
        end
    end

    # Process history
    # ---------------------------------------------------------------------------------------------------------
    # define also ξ from -11 to 1
    ar_history = Dict{Int64,Any}()
    
    # Read history data and store in AR history
    history_data = read_history_data(history_file_name)
    for t in -11:1
        row = t+12
        ar_history[t] = [history_data[row,"History_SE"], history_data[row,"History_S"], history_data[row,"History_NE"], history_data[row,"History_N"]]
    end

    # Process definition
    # ---------------------------------------------------------------------------------------------------------   
    ar_parameters = Dict{Int64, LogLinearSDDP.AutoregressiveProcessStage}()
    for t in 2:number_of_stages
        # Get month to stage
        month = mod(t, 12) > 0 ? mod(t,12) : 12

        intercept = Float64[]
        coefficients = zeros(lag_order, dim, dim)
        psi = Float64[]

        for ℓ in 1:4
            # Get model data
            df = data[ℓ]  

            # Get intercept
            push!(intercept, df[month, "Intercept"])

            # Get coefficients
            current_coefficients = df[month, "Coefficients"]
            current_coefficients = strip(current_coefficients, ']')
            current_coefficients = strip(current_coefficients, '[')
            current_coefficients = split(current_coefficients, ",")
            for k in eachindex(current_coefficients)
                coefficient = current_coefficients[k]
                coefficients[k, ℓ, ℓ] = parse(Float64, coefficient)
            end

            # Get psi
            push!(psi, df[month, "Psi"])

            # Get eta data        
            eta = get_realization_data(eta_df, t, number_of_realizations)

            ar_parameters[t] = LogLinearSDDP.AutoregressiveProcessStage(dim, intercept, coefficients, eta, psi = psi)

        end
    end
    
    # All stages
    ar_process = LogLinearSDDP.AutoregressiveProcess(lag_order, ar_parameters, ar_history)

    return ar_process
end


function set_up_ar_process_linear(number_of_stages::Int64, number_of_realizations::Int64, model_directory::String, history_file_name::String)

    # Main configuration
    # ---------------------------------------------------------------------------------------------------------
    # File information
    overall_directory = "LinearizedAutoregressivePreparation"

    # Read AR model data for all four reservoir systems
    data_SE = read_model_linear(overall_directory * "/" * model_directory * "/model_lin_SE.txt")
    data_S = read_model_linear(overall_directory * "/" * model_directory * "/model_lin_S.txt")
    data_NE = read_model_linear(overall_directory * "/" * model_directory * "/model_lin_NE.txt")
    data_N = read_model_linear(overall_directory * "/" * model_directory * "/model_lin_N.txt")
    data = [data_SE, data_S, data_NE, data_N]
    dim = 4
    lag_order = 1

    # Get realization data        
    eta_df = read_realization_data(overall_directory, model_directory, "scenarios_linear.txt")

    # Process history
    # ---------------------------------------------------------------------------------------------------------
    # define also ξ₁
    # Read history data and store in AR history
    history_data = read_history_data(history_file_name)
    ar_history = [last(history_data)["History_SE"], last(history_data)["History_S"], last(history_data)["History_NE"], last(history_data)["History_N"]]

    # Process definition
    # ---------------------------------------------------------------------------------------------------------   
    ar_parameters = Dict{Int64, LinearAutoregressiveProcessStage}()

    for t in 2:number_of_stages
        # Get month to stage
        month = mod(t, 12) > 0 ? mod(t,12) : 12
        coefficients = zeros(dim, 2)

        for ℓ in 1:4
            # Get model data
            df = data[ℓ]  

            # Get coefficients
            current_coefficients = df[month, "Corr_coefficients"]
            current_coefficients = strip(current_coefficients, ']')
            current_coefficients = strip(current_coefficients, '[')
            current_coefficients = split(current_coefficients, ",")

            coefficients[ℓ, 1] = parse(Float64, current_coefficients[1])
            coefficients[ℓ, 2] = parse(Float64, current_coefficients[2])
        end

        # Get eta data        
        eta = get_realization_data(eta_df, t, number_of_realizations)

        ar_parameters[t] = LinearAutoregressiveProcessStage(dim, coefficients, eta)
    end
   
     # All stages
     ar_process = LinearAutoregressiveProcess(lag_order, ar_parameters, ar_history)

     return ar_process
end