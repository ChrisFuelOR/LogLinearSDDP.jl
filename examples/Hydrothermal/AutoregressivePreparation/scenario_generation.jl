import CSV
import DataFrames
import Distributions

""" Reads stored model data."""
function read_model(file_name)
    f = open(file_name)
    df = CSV.read(file_name, DataFrames.DataFrame, header=false, delim=";")
    return df
end

function read_model_std(file_name)
    df = read_model(file_name)
    stds = df[:, :Column6]
    return stds
end


""" Method that generates "scenarios" in the sense that it generates a given number
of realizations and stages for the stagewise independent term (error term) in the PAR model.
The scenario data is stored in a txt file. 
Note that we do not required data for stage 1. """
function scenario_generation(number_of_realizations::Int, number_of_stages::Int)

    file_names = ["model_SE.txt", "model_S.txt", "model_NE.txt", "model_N.txt"]
    output_file_name = "scenarios_nonlinear.txt"
    f = open(output_file_name, "w")

    # Read and store required standard deviations
    stds_all_systems = Vector{Float64}[]

    for system_number in 1:4
        # Get the standard deviation for the error terms
        file_name = file_names[system_number]
        stds = read_model_std(file_name)
        push!(stds_all_systems, stds)
    end

    for t in 2:number_of_stages
        for i in 1:number_of_realizations    
            print(f, t, ";", 1/number_of_realizations)

            for system in 1:4
                # Determine month
                month = mod(t, 12) > 0 ? mod(t,12) : 12

                # Generate a realization
                d = Distributions.Normal(0.0, stds_all_systems[system][month])
                realization = Distributions.rand(d, 1)

                # Write to file
                print(f, ";", realization)
            end
            println(f)
        end
    end
    close(f)
end

scenario_generation(100, 120)