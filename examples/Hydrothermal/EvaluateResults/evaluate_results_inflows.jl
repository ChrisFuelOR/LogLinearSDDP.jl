using DataFrames
using Infiltrator
using CSV
using Statistics

struct RunConfigInflows
    model_name_run::String
    model_name_sim::String
    reservoir::String
    statistic::String
    color::String
end

function read_data(file_name::String, first_line_number::Int, number_of_rows::Int)

    df = CSV.read(file_name, header=false, skipto=first_line_number, limit=number_of_rows, delim=",", ignorerepeated=true, DataFrame)
    return df
end

function read_txt_file_inflows(file_name::String, statistic::String, seed::Int)

    # Get lines to consider
    if statistic == "mean"
        first_line_number = 1
    elseif statistic == "q1"
        first_line_number = 122
    elseif statistic == "q2"
        first_line_number = 243
    elseif statistic == "min"
        first_line_number = 364
    end

    # Read main data from table into dataframe
    df = read_data(file_name, first_line_number, 120)
    df = select!(df, Not([:Column1]))
    rename!(df, :Column2 => :InflowValue)
    df.InflowValue .= replace.(df.InflowValue, r"\)" => "")
    df.InflowValue = parse.(Float64, df.InflowValue)
    rename!(df, :InflowValue => Symbol("InflowValue_" * string(seed)))

    return df
end

function evaluate_results_inflows()

    # CONFIG
    ############################################################################################################################################

    # Define seeds that should be considered
    seeds_to_consider = [11111, 22222, 33333, 444444, 55555]
    #seeds_to_consider = [55555]

    # Define runs that should be considered
    runs_to_consider = (
        #RunConfigInflows("bic", "in_sample", "SE", "mean", "red"),
        #RunConfigInflows("custom", "in_sample", "SE", "mean", "blue"),
        #RunConfigInflows("fitted", "in_sample", "SE", "mean", "green!70!black"),
        #RunConfigInflows("shapiro", "in_sample", "SE", "mean", "cyan"),
        #RunConfigInflows("bic", "bic", "SE", "mean", "red"),
        #RunConfigInflows("custom", "custom", "SE", "mean", "blue"),
        #RunConfigInflows("fitted", "fitted", "SE", "mean", "green!70!black"),
        #RunConfigInflows("shapiro", "shapiro", "SE", "mean", "cyan"),

        # RunConfigInflows("custom", "in_sample", "NE", "mean", "blue"),
        # RunConfigInflows("custom", "custom", "NE", "mean", "blue"),
        # RunConfigInflows("custom", "in_sample", "NE", "q1", "blue"),
        # RunConfigInflows("custom", "custom", "NE", "q1", "blue"),
        # RunConfigInflows("custom", "in_sample", "NE", "q2", "blue"),
        # RunConfigInflows("custom", "custom", "NE", "q2", "blue"),

        RunConfigInflows("shapiro", "in_sample", "NE", "mean", "blue"),
        RunConfigInflows("shapiro", "shapiro", "NE", "mean", "blue"),
        RunConfigInflows("shapiro", "in_sample", "NE", "q1", "blue"),
        RunConfigInflows("shapiro", "shapiro", "NE", "q1", "blue"),
        RunConfigInflows("shapiro", "in_sample", "NE", "q2", "blue"),
        RunConfigInflows("shapiro", "shapiro", "NE", "q2", "blue"),

        # RunConfigInflows("custom", "in_sample", "SE", "mean", "blue"),
        # RunConfigInflows("custom", "in_sample", "SE", "q1", "blue"),
        # RunConfigInflows("custom", "in_sample", "SE", "q2", "blue"),
        # RunConfigInflows("shapiro", "in_sample", "SE", "mean", "cyan"),
        # RunConfigInflows("shapiro", "in_sample", "SE", "q1", "cyan"),
        # RunConfigInflows("shapiro", "in_sample", "SE", "q2", "cyan"),
    )

    # Cut selection?
    cut_selection_flag = true

    # Considered number of stages
    number_of_stages = 60

    # Output file path for LaTeX plots
    file_path_latex = "C:/Users/cg4102/Documents/julia_plots/Cut_sharing 2025/Inflow_Comparison_NE_Lin_Sha.tex"
    create_latex_plots_starter(file_path_latex, number_of_stages)

    # Create DataFrame
    inflow_df = DataFrame()

    ############################################################################################################################################
    
    # Iterate over runs and read data from CSV files
    for run_index in eachindex(runs_to_consider)

        run = runs_to_consider[run_index]

        if cut_selection_flag
            input_dir_1 = "C:/Users/cg4102/Documents/julia_logs/Cut-sharing/Runs server 2025/Cut-sharing Re-Run (2h)/"
        else
            input_dir_1 = "C:/Users/cg4102/Documents/julia_logs/Cut-sharing/Runs server 2025/Cut-sharing Re-Run - NoCS (2h)/"
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

            input_file_name = input_dir_1 * input_dir_2 * run_identifier * "_" * sim_identifier * "_inflows_" * run.reservoir * ".txt"

            # Read input file
            df_seed = read_txt_file_inflows(input_file_name, run.statistic, seed)

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
        column_name = Symbol(run.model_name_run * "_" * run.model_name_sim * "_" * run.reservoir * "_" * run.statistic)

        # Compute average
        df_run[!, column_name] = Statistics.mean.(eachrow(df_run))
        df_run = select!(df_run, [column_name])
        
        # Add to plot
        create_plot(file_path_latex, run, df_run, number_of_stages)

        # Add to inflow_df
        if isempty(inflow_df)
            inflow_df = copy(df_run)
        else
            inflow_df = hcat(inflow_df, df_run)
        end
    end

    create_latex_plots_finisher(file_path_latex)
    Infiltrator.@infiltrate
end




function create_latex_header(io::Any, axis_limits::Vector{Float64})

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

    println(io, "xlabel={Stage}, ylabel={Inflows},")

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

function create_latex_plots_starter(file_path_latex::String, number_of_stages::Int)

    io = open(file_path_latex, "w")
	create_latex_header(io, [0.0, Float64(number_of_stages), 0.0, 100000.0])
    close(io)

    return
end

function create_latex_plots_finisher(file_path_latex::String)

    io = open(file_path_latex, "a")
	create_latex_footer(io)
    close(io)

    return
end

function create_plot(file_path_latex::String, run_config::RunConfigInflows, df::DataFrame, number_of_stages::Int)

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

evaluate_results_inflows()
