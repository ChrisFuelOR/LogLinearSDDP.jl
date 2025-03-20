using DataFrames
using CSV
using Infiltrator

function clean_up_inflow_data()
    # Parameters
    model_approach = "fitted_model"
    seed = 11111

    filename_input = "C:/Users/cg4102/Documents/julia_logs/Cut-sharing/Runs server 2025/Cut-sharing Time/Run_" * model_approach * "_" * string(seed) * "/inflows_lin.txt"
    filename_output = "C:/Users/cg4102/Documents/julia_logs/Cut-sharing/Runs server 2025/Cut-sharing Time/Run_" * model_approach * "_" * string(seed) * "/inflows_lin_cleaned.txt"
    f = open(filename_output, "w")

    df = CSV.read(filename_input, header=false, delim=";", DataFrame)
    #new_df = DataFrame()

    for row in eachrow(df)
        if rownumber(row) == 1
            #push!(new_df, row)
            # Add historical values for stage 1 first
            println(f, 1, ";", 77311.23025371083, ";", 3453.312280663998, ";", 12758.297678803967, ";", 6250.776303265441, ";")
            println(f, row[:Column1], ";", row[:Column2], ";", row[:Column3], ";", row[:Column4], ";", row[:Column5], ";")
        elseif row[:Column1] == 1
        
        elseif row[:Column1] > df[rownumber(row)-1, :Column1]
            #push!(new_df, row)
            println(f, row[:Column1], ";", row[:Column2], ";", row[:Column3], ";", row[:Column4], ";", row[:Column5], ";")
        elseif rownumber(row) < nrow(df) && row[:Column1] == 2 && df[rownumber(row)+1, :Column1] == 3
            #push!(new_df, row)
            # Add historical values for stage 1 first
            println(f, 1, ";", 77311.23025371083, ";", 3453.312280663998, ";", 12758.297678803967, ";", 6250.776303265441, ";")
            println(f, row[:Column1], ";", row[:Column2], ";", row[:Column3], ";", row[:Column4], ";", row[:Column5], ";")
        end
    end

    close(f)

    return
end

clean_up_inflow_data()