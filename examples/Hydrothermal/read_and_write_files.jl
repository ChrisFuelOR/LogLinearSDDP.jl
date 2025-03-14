import DataFrames
import CSV
import DataFramesMeta


""" Reads stored model data for log-linear model."""
function read_model_log_linear(file_name)
    f = open(file_name)
    df = CSV.read(f, DataFrames.DataFrame, header=false, delim=";")
    DataFrames.rename!(df, ["Month", "Lag_order", "Intercept", "Coefficients", "Psi", "Sigma"])    
    close(f)
    return df
end

""" Reads stored model data for linear model."""
function read_model_linear(file_name)
    f = open(file_name)
    df = CSV.read(file_name, DataFrames.DataFrame, header=false, delim=";")
    DataFrames.rename!(df, ["Month", "Lag_order", "Coefficient", "Corr_coefficients", "Sigma"])    
    close(f)
    return df
end

""" Read stored realization data."""
function read_realization_data(file_name::String)
    f = open(file_name)
    
    df = CSV.read(f, DataFrames.DataFrame, header=false, delim=";")
    DataFrames.rename!(df, ["Stage", "Realization_number", "Probability", "Realization_SE", "Realization_S", "Realization_NE", "Realization_N"])    
    close(f)
    return df
end

""" Read stored realization data."""
function read_realization_data(overall_directory::String, model_directory::String, file_name::String)
    file_name = overall_directory * "/" * model_directory * "/" * file_name
    return read_realization_data(file_name)
end

""" Read data for covariance matrices of the residuals of the AR model """
function read_sigma_data(
    month::Int64,
    model_directory::String,
)

    sigma_data = CSV.read(model_directory * "/sigma_" * string(month-1) * ".csv", DataFrames.DataFrame, header=true, delim=",")
    DataFrames.rename!(sigma_data, ["row", "1", "2", "3", "4"])    
    sigma = Matrix(DataFrames.select!(sigma_data, DataFrames.Not([:row])))

    return sigma
end

function read_model_std(file_name)
    df = read_model(file_name)
    stds = df[:, "Sigma"]
    return stds
end


""" Read data for coefficients of the AR model """
function read_gamma_data(
    system_number::Int64,
    model_directory::String,
)

    gamma_data = CSV.read(model_directory * "/gamma.csv", DataFrames.DataFrame, header=true, delim=",")
    DataFrames.rename!(gamma_data, ["row", "1", "2", "3", "4"])    
    gamma = gamma_data[:, string(system_number)]

    return gamma
end

""" Read stored history data for the process."""
function read_history_data(file_name::String)
    f = open(file_name)
    df = CSV.read(f, DataFrames.DataFrame, header=false, delim=";")
    DataFrames.rename!(df, ["Stage", "History_SE", "History_S", "History_NE", "History_N"])    
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


function parse_coefficients(coefficients::Any)

    output_coefficients = Vector{Float64}()
    coefficients = strip(coefficients, ']')
    coefficients = strip(coefficients, '[')
    coefficients = split(coefficients, ",")

    for k in eachindex(coefficients)
        push!(output_coefficients, parse(Float64, coefficients[k]))
    end

    return output_coefficients
end

""" Reads stored historical sample paths."""
function read_historical_simulation_data(file_name)
    f = open(file_name)
    df = CSV.read(f, DataFrames.DataFrame, header=false, delim=";")
    df = DataFrames.select!(df, DataFrames.Not([:Column71]))
    DataFrames.rename!(df, [Symbol(i) for i in 1:70])    
    close(f)
    return df
end