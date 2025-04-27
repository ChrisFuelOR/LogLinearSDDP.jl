using DataFrames
using Infiltrator
using CSV
using Statistics

struct RunConfigData
    model_name_run::String
    model_name_sim::String
    cut_selection::Bool
    statistic::String
end

function read_data(file_name::String)

    df = CSV.read(file_name, header=false, delim=",", ignorerepeated=true, DataFrame)
    return df
end

function evaluate_results_boxplot()

    # CONFIG
    ############################################################################################################################################
    
    # Define seeds that should be considered
    seeds_to_consider = [11111, 22222, 33333, 444444, 55555]

    # Define runs that should be considered
    runs_to_consider = (
        RunConfigData("bic", "custom", true, "mean"),
        RunConfigData("custom", "custom", true, "mean"),
        RunConfigData("fitted", "custom", false, "mean"),
        RunConfigData("shapiro", "custom", false, "mean"),
        RunConfigData("MC-SDDP_lattice", "custom", false, "mean"),
        RunConfigData("MC-SDDP_custom", "custom", false, "mean"),
        RunConfigData("MC-SDDP_fitted", "custom", false, "mean"),
        RunConfigData("MC-SDDP_shapiro", "custom", false, "mean"),
    )

    # Full number of stages or half of stages
    only_half_of_stages = true

    # Construct output file name
    output_dir = "C:/Users/cg4102/Documents/Forschung/Benders Decomposition & Extensions/SDDP/Cut-sharing Paper - 2023/VSCode - Cut-sharing/OPRE-style-file-2024/Bilder_Numerik/Boxplot_NoCS/"
    output_file_name_1 = output_dir * "data_long_" * "oos_lin.csv"
    output_file_name_2 = output_dir * "data_short_" * "oos_lin.csv"
    f_1 = open(output_file_name_1, "w")
    f_2 = open(output_file_name_2, "w")

    # Scaling factor for diagram
    scaling_factor = 1e8

    ############################################################################################################################################
    
    # Iterate over runs and read data from CSV files
    for run_index in eachindex(runs_to_consider)

        run = runs_to_consider[run_index]

        if run.cut_selection
            input_dir_1 = "C:/Users/cg4102/Documents/julia_logs/Cut-sharing/Runs server 2025/Cut-sharing Final Run (2h)/"
        else
            input_dir_1 = "C:/Users/cg4102/Documents/julia_logs/Cut-sharing/Runs server 2025/Cut-sharing Final Run (2h) - NoCS/"
        end

        # Create DataFrame
        data_df_run = DataFrame()

        for seed_index in eachindex(seeds_to_consider)

            seed = seeds_to_consider[seed_index]

            # Construct input file name
            if run.model_name_run == "MC-SDDP_lattice"
                appendix = "_"
            else
                appendix = "_model_"
            end
            input_dir_2 = "Run_" * run.model_name_run * appendix * string(seed) * "/"

            if run.model_name_run in ["MC-SDDP_lattice", "MC-SDDP_custom", "MC-SDDP_fitted", "MC-SDDP_shapiro"]
                run_identifier = "markov"
            else
                run_identifier = run.model_name_run * "_model"
            end

            if run.model_name_sim in ["in_sample", "historical"]
                sim_identifier = run.model_name_sim
            else
                sim_identifier = run.model_name_sim * "_model"
            end
            
            input_file_name = input_dir_1 * input_dir_2 * run_identifier * "_" * sim_identifier * "_total_cost.txt"

            # Read input file
            data_df_seed = read_data(input_file_name)

            # Only take column based on only_half_of_stages
            if only_half_of_stages
                data_df_seed = select!(data_df_seed, [:Column1])
                rename!(data_df_seed, :Column1 => Symbol(seed))
            else
                data_df_seed = select!(data_df_seed, [:Column2])
                rename!(data_df_seed, :Column2 => Symbol(seed))
            end

            # Add to existing DataFrame         
            if isempty(data_df_run)
                data_df_run = copy(data_df_seed)
            else
                data_df_run = hcat(data_df_run, data_df_seed)
            end

        end

        # Average over seeds - and write to file
        ##############################################
        avg_rows = Statistics.mean.(eachrow(data_df_run))
        for i in eachindex(avg_rows)
            print(f_1, round(avg_rows[i]/scaling_factor, digits=2))
            if i < length(avg_rows)
                print(f_1, ", ")
            end
        end
        println(f_1)

        # Average over replications - and write to file
        ##############################################
        avg_cols = Statistics.mean.(eachcol(data_df_run))
        for i in eachindex(avg_cols)
            print(f_2, round(avg_cols[i]/scaling_factor, digits=2))
            if i < length(avg_cols)
                print(f_2, ", ")
            end
        end
        println(f_2)
    end

    # Close output files 1 & 2
    close(f_1)
    close(f_2)  

end

evaluate_results_boxplot()