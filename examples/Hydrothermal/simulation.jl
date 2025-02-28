import Distributions
import DataFrames
import SDDP
import Statistics
import StatsBase

function read_model_std(file_name)
    if startswith(file_name, "Linearized")
        df = read_model_linear(file_name)
    else
        df = read_model_log_linear(file_name)
    end
    stds = df[:, "Sigma"]
    return stds
end

function get_realization(std::Float64)

    d = Distributions.Normal(0.0, std)
    realization = Distributions.rand(d, 1)[1]
    
    return realization
end

function get_realization(std_matrix::Array{Float64,2})

    d = Distributions.MvNormal(zeros(4), std_matrix)
    realizations = Distributions.rand(d, 1)

    return realizations
end


""" Read data for covariance matrices of the residuals of the AR model for multivariate case """
function read_sigma_data(
    month::Int64,
    model_directory::String,
)

    sigma_data = CSV.read(model_directory * "/sigma_" * string(month-1) * ".csv", DataFrames.DataFrame, header=true, delim=",")
    DataFrames.rename!(sigma_data, ["row", "1", "2", "3", "4"])    
    sigma = Matrix(DataFrames.select!(sigma_data, DataFrames.Not([:row])))

    return sigma
end


""" Method that generates "scenarios" in the sense that it generates a given number
of realizations and stages for the stagewise independent term (error term) in the PAR model.
Note that we do not required data for stage 1. """
function get_out_of_sample_realizations(number_of_realizations::Int64, t::Int64, file_names::Vector{String})

    if t == 1
        realizations = [SDDP.Noise([0.0, 0.0, 0.0, 0.0], 1.0)]
    else
        # Read and store required standard deviations
        stds_all_systems = Vector{Float64}[]
        for system_number in 1:4
            file_name = file_names[system_number]
            stds = read_model_std(file_name)
            push!(stds_all_systems, stds)
        end

        # Determine month to current stage
        month = mod(t, 12) > 0 ? mod(t,12) : 12

        # Object to store realizations
        realizations = SDDP.Noise{Vector{Float64}}[]
        prob = 1/number_of_realizations

        for i in 1:number_of_realizations    
            # Get and store the realizations for all 4 systems
            realization_vector = [get_realization(stds_all_systems[1][month]), get_realization(stds_all_systems[2][month]), get_realization(stds_all_systems[3][month]), get_realization(stds_all_systems[4][month])]
            push!(realizations, SDDP.Noise(realization_vector, prob))
        end
    end
    
    return realizations
end


function get_out_of_sample_realizations_loglinear(number_of_realizations::Int64, t::Int64, model_directory::String)
    model_directory = "AutoregressivePreparation/" * model_directory

    return get_out_of_sample_realizations(
        number_of_realizations, 
        t, 
        [model_directory * "/model_SE.txt", model_directory * "/model_S.txt", model_directory * "/model_NE.txt", model_directory * "/model_N.txt"]
    )
end


function get_out_of_sample_realizations_linear(number_of_realizations::Int64, t::Int64, model_directory::String)
    model_directory = "LinearizedAutoregressivePreparation/" * model_directory

    return get_out_of_sample_realizations(
        number_of_realizations, 
        t, 
        [model_directory * "/model_lin_SE.txt", model_directory * "/model_lin_S.txt", model_directory * "/model_lin_NE.txt", model_directory * "/model_lin_N.txt"]
    )
end


""" Method that generates "scenarios" in the sense that it generates a given number
of realizations and stages for the stagewise independent term (error term) in the PAR model.
Note that we do not required data for stage 1. """
function get_out_of_sample_realizations_multivariate(number_of_realizations::Int64, t::Int64, model_directory::String)

    if t == 1
        realizations = [SDDP.Noise([0.0, 0.0, 0.0, 0.0], 1.0)]
    else
        # Determine month to current stage
        month = mod(t, 12) > 0 ? mod(t,12) : 12

        # Read and store required standard deviations
        sigma_matrix = read_sigma_data(month, model_directory)

        # Object to store realizations
        realizations = SDDP.Noise{Vector{Float64}}[]
        prob = 1/number_of_realizations

        for i in 1:number_of_realizations    
            # Get and store the realizations for all 4 systems
            realization_vector = vec(get_realization(sigma_matrix))
            push!(realizations, SDDP.Noise(realization_vector, prob))
        end
    end
    
    return realizations
end

function get_out_of_sample_realizations_multivariate_linear(number_of_realizations::Int64, t::Int64, model_directory::String)
    model_directory = "LinearizedAutoregressivePreparation/" * model_directory

    return get_out_of_sample_realizations_multivariate(
        number_of_realizations, 
        t, 
        model_directory,
    )
end

function extended_simulation_analysis(simulation_results::Any, file_path::String, policy_approach::String, simulation_approach::String)

    # GET EMPIRICAL CUMULATIVE DISTRIBUTION OF COSTS
    ############################################################################
    # Declare file name
    file_name = file_path * policy_approach * "_" * simulation_approach * "_cum_distrib.txt"
    f = open(file_name, "w")

    objectives = map(simulation_results) do simulation
        return sum(stage[:stage_objective] for stage in simulation)
    end

    distrib = StatsBase.ecdf(objectives)
    stepsize = round((maximum(objectives)-minimum(objectives))/100, digits=0)
    for x in minimum(objectives)-stepsize:stepsize:maximum(objectives)+stepsize
        y = distrib(x)
        println(f, "(", round(x, digits = 0), ",", round(y, digits = 2), ")")
    end
    close(f)

    # OBTAINING HYDRO RESERVOIR LEVELS
    ############################################################################
    reservoir_names = ["SE", "S", "NE", "N"]

    for k in 1:4
        # Declare file name
        file_name = file_path * policy_approach * "_" * simulation_approach * "_volumes_" * reservoir_names[k] * ".txt"
        f = open(file_name, "w")

        # Get volume data for given reservoir
        column_names = [Symbol(i) for i in 1:120]
        volume_df = DataFrames.DataFrame([name => Float64[] for name in column_names])
        for i in eachindex(simulation_results)
            outgoing_volume = map(simulation_results[i]) do node
                return node[:level][k].out
            end    
            push!(volume_df, outgoing_volume)
        end

        # mean
        for stage in 1:120
            println(f, "(", stage, ",", round(Statistics.mean(volume_df[!, Symbol(stage)]), digits = 2), ")")
        end
        println(f, "###################")

        # 0.05 quantile
        for stage in 1:120
            println(f, "(", stage, ",", round(Statistics.quantile(volume_df[!, Symbol(stage)], 0.05), digits = 2), ")")
        end
        println(f, "###################")

        # 0.95 quantile
        for stage in 1:120
            println(f, "(", stage, ",", round(Statistics.quantile(volume_df[!, Symbol(stage)], 0.95), digits = 2), ")")
        end
        println(f, "###################")

        close(f)
    end

    # OBTAINING INFLOW VALUES
    ############################################################################
    reservoir_names = ["SE", "S", "NE", "N"]

    for k in 1:4
        # Declare file name
        file_name = file_path * policy_approach * "_" * simulation_approach * "_inflows_" * reservoir_names[k] * ".txt"
        f = open(file_name, "w")

        # Get inflow data for given reservoir
        column_names = [Symbol(i) for i in 1:120]
        inflow_df = DataFrames.DataFrame([name => Float64[] for name in column_names])
        for i in eachindex(simulation_results)
            inflow = map(simulation_results[i]) do node
                if isa(node[:inflow][k], SDDP.State)
                    return node[:inflow][k].out
                else
                    return node[:inflow][k]
                end
            end    
            push!(inflow_df, inflow)
        end
        Infiltrator.@infiltrate

        # mean
        for stage in 1:120
            println(f, "(", stage, ",", round(Statistics.mean(inflow_df[!, Symbol(stage)]), digits = 2), ")")
        end
        println(f, "###################")

        # 0.05 quantile
        for stage in 1:120
            println(f, "(", stage, ",", round(Statistics.quantile(inflow_df[!, Symbol(stage)], 0.05), digits = 2), ")")
        end
        println(f, "###################")

        # 0.95 quantile
        for stage in 1:120
            println(f, "(", stage, ",", round(Statistics.quantile(inflow_df[!, Symbol(stage)], 0.95), digits = 2), ")")
        end
        println(f, "###################")

        # minimum (to rule out negative inflows)
        for stage in 1:120
            println(f, "(", stage, ",", round(minimum(inflow_df[!, Symbol(stage)]), digits = 2), ")")
        end
        println(f, "###################")

        close(f)
    end

    # OBTAINING COST-RELEVANT VARIABLES
    ############################################################################
    # Declare file name
    file_name = file_path * policy_approach * "_" * simulation_approach * "_data.txt"
    f = open(file_name, "w")

    # Get volume data for given reservoir
    column_names = [Symbol(i) for i in 1:120]
    value_df = DataFrames.DataFrame([name => Vector{Vector{Float64}}() for name in column_names])
    for i in eachindex(simulation_results)
        outgoing_values = map(simulation_results[i]) do node
            gen = sum(node[:gen][j] for j in 1:95)
            hydro_gen = sum(node[:hydro_gen][k] for k in 1:4)
            deficit = sum(node[:deficit_part][k,i] for i in 1:4 for k in 1:5)
            exchange = sum(node[:exchange][k,l] for l in 1:5 for k in 1:5)
            spillage = sum(node[:spillage][k] for k in 1:4)
            return [gen, hydro_gen, deficit, exchange, spillage]
        end    
        push!(value_df, outgoing_values)
    end

    # compute and log mean
    for stage in 1:120
        gen_sum = 0.0
        hydro_gen_sum = 0.0
        deficit_sum = 0.0
        exchange_sum = 0.0
        spillage_sum = 0.0

        for i in 1:DataFrames.nrow(value_df)
            gen_sum += value_df[!, Symbol(stage)][i][1] 
            hydro_gen_sum += value_df[!, Symbol(stage)][i][2] 
            deficit_sum += value_df[!, Symbol(stage)][i][3] 
            exchange_sum += value_df[!, Symbol(stage)][i][4] 
            spillage_sum += value_df[!, Symbol(stage)][i][5] 
        end

        gen_avg = gen_sum/DataFrames.nrow(value_df)
        hydro_gen_avg = hydro_gen_sum/DataFrames.nrow(value_df)
        deficit_avg = deficit_sum/DataFrames.nrow(value_df)
        exchange_avg = exchange_sum/DataFrames.nrow(value_df)
        spillage_avg = spillage_sum/DataFrames.nrow(value_df)

        println(f, "(", stage, ",", round(gen_avg, digits = 2), ",", round(hydro_gen_avg, digits = 2), ",", round(deficit_avg, digits = 2), ",", round(exchange_avg, digits = 2), ",", round(spillage_avg, digits = 2), ")")
    end
    println(f, "###################")

    close(f)

    return
end
