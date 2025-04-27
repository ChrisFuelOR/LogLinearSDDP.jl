using DataFrames
using Infiltrator
using CSV
using Statistics

struct RunConfigData
    model_name_run::String
    model_name_sim::String
    statistic::String
    color::String
end

function read_data(file_name::String, first_line_number::Int, number_of_rows::Int)

    df = CSV.read(file_name, header=false, skipto=first_line_number, limit=number_of_rows, delim=",", ignorerepeated=true, DataFrame)
    return df
end

function read_txt_file_data(file_name::String, statistic::String, seed::Int, property::String)

    # Get lines to consider
    if statistic == "mean"
        first_line_number = 1
    elseif statistic == "q1"
        first_line_number = 122
    elseif statistic == "q2"
        first_line_number = 243
    end

    # Read main data from table into dataframe
    df = read_data(file_name, first_line_number, 120)
    
    if property == "Gen"
        selected_column = :Column2
    elseif property == "HydGen"
        selected_column = :Column3
    elseif property == "Deficit"
        selected_column = :Column4
    elseif property == "DeficitCost"
        selected_column = :Column5
    elseif property == "Exchange"
        selected_column = :Column6
    elseif property == "Spillage"
        selected_column = :Column7
        df.Column7 .= replace.(df.Column7, r"\)" => "")
        df.Column7 = parse.(Float64, df.Column7)
    end  
        
    df = select!(df, [selected_column])
    rename!(df, selected_column => Symbol(property * string(seed)))

    return df
end


function evaluate_results_data()

    # CONFIG
    ############################################################################################################################################

    # Define seeds that should be considered
    seeds_to_consider = [11111, 22222, 33333, 444444, 55555]
    #seeds_to_consider = [55555]

    # Define runs that should be considered
    runs_to_consider = (
        #RunConfigData("bic", "in_sample", "mean", "red"),
        #RunConfigData("custom", "in_sample", "mean", "blue"),
        #RunConfigData("fitted", "in_sample", "mean", "green!70!black"),
        #RunConfigData("shapiro", "in_sample", "mean", "cyan"),

        # RunConfigData("bic", "custom", "mean", "red"),
        # RunConfigData("custom", "custom", "mean", "blue"),
        # RunConfigData("fitted", "custom", "mean", "green!70!black"),
        # RunConfigData("shapiro", "custom", "mean", "cyan"),
        # RunConfigData("MC-SDDP_lattice", "custom","mean", "magenta"),
        # RunConfigData("MC-SDDP_custom", "custom", "mean", "brown!80!black"),
        # RunConfigData("MC-SDDP_fitted", "custom", "mean", "black"),
        # RunConfigData("MC-SDDP_shapiro", "custom", "mean", "orange"),

        #RunConfigData("bic", "historical", "mean", "red"),
        RunConfigData("custom", "historical", "mean", "blue"),
        #RunConfigData("fitted", "historical", "mean", "green!70!black"),
        RunConfigData("shapiro", "historical", "mean", "cyan"),
        RunConfigData("MC-SDDP_lattice", "historical", "mean", "magenta"),
        #RunConfigData("MC-SDDP_custom", "historical", "mean", "brown!80!black"),
        #RunConfigData("MC-SDDP_fitted", "historical", "mean", "black"),
        #RunConfigData("MC-SDDP_shapiro", "historical", "mean", "orange"),
    )

    # Property to analyze
    property = "HydGen"

    # Cut selection?
    cut_selection_flag = false

    # Considered number of stages
    number_of_stages = 60

    # Output file path for LaTeX plots
    file_path_latex = "C:/Users/cg4102/Documents/julia_plots/Cut_sharing 2025/" * property * "_blub.tex"
    create_latex_plots_starter(file_path_latex, number_of_stages, property)

    # Create DataFrame
    data_df = DataFrame()

    ############################################################################################################################################
    
    # Iterate over runs and read data from CSV files
    for run_index in eachindex(runs_to_consider)

        run = runs_to_consider[run_index]

        if cut_selection_flag
            input_dir_1 = "C:/Users/cg4102/Documents/julia_logs/Cut-sharing/Runs server 2025/Cut-sharing Final Run (2h)/"
        else
            input_dir_1 = "C:/Users/cg4102/Documents/julia_logs/Cut-sharing/Runs server 2025/Cut-sharing Final Run (2h) - NoCS/"
        end

        df_run = DataFrame()

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

            input_file_name = input_dir_1 * input_dir_2 * run_identifier * "_" * sim_identifier * "_data.txt"

            # Read input file
            df_seed = read_txt_file_data(input_file_name, run.statistic, seed, property)

            # Add to existing DataFrame         
            if isempty(df_run)
                df_run = copy(df_seed)
            else
                df_run = hcat(df_run, df_seed)
            end

        end

        # Average over seeds
        ##############################################
        # Make column identifiable
        column_name = Symbol(run.model_name_run * "_" * run.model_name_sim * "_" * run.statistic)

        # Compute average
        df_run[!, column_name] = Statistics.mean.(eachrow(df_run))
        df_run = select!(df_run, [column_name])
        
        # Add to plot
        create_plot(file_path_latex, run, df_run, number_of_stages)

        # Add to volumes_df
        if isempty(data_df)
            data_df = copy(df_run)
        else
            data_df = hcat(data_df, df_run)
        end
    end

    create_latex_plots_finisher(file_path_latex)
    Infiltrator.@infiltrate
end




function create_latex_header(io::Any, axis_limits::Vector{Float64}, label::String)

	println(io, "\\documentclass[tikz]{standalone}")
	println(io, "\\usetikzlibrary{intersections}")
	println(io, "\\usepackage{pgfplots}")
	println(io, "\\usepackage{etoolbox}")
	println(io, "\\usetikzlibrary{matrix}")
	println(io)

	println(io, "\\begin{document}")
	println(io)

	println(io, "\\begin{tikzpicture}[scale=1,line cap=round,every mark/.append style={mark size=1pt}]")
	println(io, "%Styles")
	println(io, "\\pgfplotsset{")
	println(io, "axis line style={gray!30!black},")
	println(io, "every axis label/.append style ={gray!30!black},")
	println(io, "every tick label/.append style={gray!30!black}")
	println(io, "}")
	println(io)

	println(io, "\\begin{axis}")
	println(io, "[")
	println(io, "axis on top = true,")
	println(io, "axis lines = left,")
	println(io, "xmin=", axis_limits[1], ", xmax=", axis_limits[2], ", ymin=", axis_limits[3], ", ymax=", axis_limits[4], ",")
	println(io, "ticklabel style={fill=white},")

    println(io, "xlabel={Stage}, ylabel={", label, "},")

	println(io, "clip=false] ")
	println(io)
	println(io)

end

function create_latex_footer(io::Any)

	println(io, "\\end{axis}")
	println(io)

	println(io, "\\end{tikzpicture}")
	println(io, "\\end{document}")

end

function create_latex_plots_starter(file_path_latex::String, number_of_stages::Int, property::String)

    io = open(file_path_latex, "w")

    axis_limits = [0.0, Float64(number_of_stages), 0.0, 0.0]
    if property == "Gen"
        axis_limits[4] = 20000.0
    elseif property == "HydGen"
        axis_limits[3] = 40000.0
        axis_limits[4] = 70000.0
    elseif property == "Deficit"
        axis_limits[4] = 2000.0
    elseif property == "DeficitCost"
        axis_limits[4] = 10000000.0
    elseif property == "Exchange"
        axis_limits[3] = 5000.0
        axis_limits[4] = 15000.0
    elseif property == "Spillage"
        axis_limits[4] = 15000.0
    end

	create_latex_header(io, axis_limits, property)
    close(io)

    return
end

function create_latex_plots_finisher(file_path_latex::String)

    io = open(file_path_latex, "a")
	create_latex_footer(io)
    close(io)

    return
end

function create_plot(file_path_latex::String, run_config::RunConfigData, df::DataFrame, number_of_stages::Int)

	io = open(file_path_latex, "a")

	# PLOT HEADER
	############################################################################
	# println(io, "%", plot_config.legend)
	println(io, "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

	print(io, "\\addplot[color=")
	print(io, run_config.color)
	print(io, "] coordinates {")
	println(io)

	# Plot data points for each row
	for row in eachrow(df)
        if rownumber(row) <= number_of_stages
            print(io, "(   ")
            print(io, rownumber(row))
            print(io, "   ,   ")
            print(io, row[1])
            print(io, ")   ")
            println(io)
        end
	end

	# PLOT FOOTER
	############################################################################
	println(io, "};")
	println(io)
	println(io)
	println(io)

	close(io)

end

evaluate_results_data()
