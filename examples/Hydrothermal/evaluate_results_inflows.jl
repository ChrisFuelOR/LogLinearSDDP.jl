using DataFrames
using Infiltrator
using CSV

struct RunConfigInflows
    seed::Int
    model_name_run::String
    model_name_sim::String
    reservoir::String
    stopping::String
    statistic::String
    color::String
end



function evaluate_results_inflows()

    # Define runs that should be considered
    runs_to_consider = (
        # RunConfigInflows(11111, "bic", "in_sample", "SE", "IterLimit", "mean", "red"),
        # RunConfigInflows(11111, "custom", "in_sample", "SE", "IterLimit", "mean", "blue"),
        # RunConfigInflows(11111, "fitted", "in_sample", "SE", "IterLimit", "mean", "green!70!black"),
        # RunConfigInflows(11111, "shapiro", "in_sample", "SE", "IterLimit", "mean", "cyan"),
        # RunConfigInflows(11111, "bic", "bic", "SE", "IterLimit", "mean", "red"),
        # RunConfigInflows(11111, "bic", "custom", "SE", "IterLimit", "mean", "blue"),
        # RunConfigInflows(11111, "bic", "fitted", "SE", "IterLimit", "mean", "green!70!black"),
        # RunConfigInflows(11111, "bic", "shapiro", "SE", "IterLimit", "mean", "cyan"),

        #RunConfigInflows(33333, "bic", "bic", "SE", "IterLimit", "mean", "blue"),
        #RunConfigInflows(33333, "custom", "custom", "SE", "IterLimit", "mean", "red"),
        RunConfigInflows(33333, "fitted", "fitted", "SE", "IterLimit", "mean", "green!70!black"),
        #RunConfigInflows(33333, "shapiro", "shapiro", "SE", "IterLimit", "mean", "cyan"),    

        #RunConfigInflows(33333, "bic", "in_sample", "SE", "IterLimit", "mean", "blue"),
        #RunConfigInflows(33333, "custom", "in_sample", "SE", "IterLimit", "mean", "red"),
        RunConfigInflows(33333, "fitted", "in_sample", "SE", "IterLimit", "mean", "green!70!black"),
        #RunConfigInflows(33333, "shapiro", "in_sample", "SE", "IterLimit", "mean", "blue"),  

        RunConfigInflows(55555, "fitted", "fitted", "SE", "IterLimit", "mean", "magenta"),    
        RunConfigInflows(55555, "fitted", "in_sample", "SE", "IterLimit", "mean", "red"),  


        # RunConfig(11111, "bic", "in_sample", "SE", "IterLimit", "q1", "red"),
        # RunConfig(11111, "custom", "in_sample", "SE", "IterLimit", "q1", "blue"),
        # RunConfig(11111, "fitted", "in_sample", "SE", "IterLimit", "q1", "green!70!black"),
        # RunConfig(11111, "shapiro", "in_sample", "SE", "IterLimit", "q1", "black"),
        # RunConfig(11111, "bic", "bic", "SE", "IterLimit", "q1", "magenta"),
        # RunConfig(11111, "bic", "custom", "SE", "IterLimit", "q1", "cyan"),
        # RunConfig(11111, "bic", "fitted", "SE", "IterLimit", "q1", "green"),
        # RunConfig(11111, "bic", "shapiro", "SE", "IterLimit", "q1", "gray"),

        # RunConfig(11111, "bic", "in_sample", "SE", "IterLimit", "q2", "red"),
        # RunConfig(11111, "custom", "in_sample", "SE", "IterLimit", "q2", "blue"),
        # RunConfig(11111, "fitted", "in_sample", "SE", "IterLimit", "q2", "green!70!black"),
        # RunConfig(11111, "shapiro", "in_sample", "SE", "IterLimit", "q2", "black"),
        # RunConfig(11111, "bic", "bic", "SE", "IterLimit", "q2", "magenta"),
        # RunConfig(11111, "bic", "custom", "SE", "IterLimit", "q2", "cyan"),
        # RunConfig(11111, "bic", "fitted", "SE", "IterLimit", "q2", "green"),
        # RunConfig(11111, "bic", "shapiro", "SE", "IterLimit", "q2", "gray"),
    )

    # Define max number of stages
    number_of_stages = 120

    # Dataframe to store inflow data
    inflow_df = DataFrame()

    # File path for LaTeX plots
    file_path_latex = "C:/Users/cg4102/Documents/julia_plots/Cut_sharing 2025/Inflows_33333.tex"
    create_latex_plots_starter(file_path_latex, number_of_stages)

    # Iterate over runs and read data from CSV files
    for run in runs_to_consider

        # Construct path and file name
        if run.stopping == "TimeLimit"
            dir_name = "Cut-sharing - Time"
        elseif run.stopping == "IterLimit"
            dir_name = "Cut-sharing"
        else
            Error("Stopping criterion not recognized.")
        end

        # if run.model_name_sim == "in_sample"
        #     aux_string2 = ""
        #     if run.model_name_run != "bic"
        #         aux_string1 = "_"
        #     else
        #         aux_string1 = ""
        #     end
        # else
        #     aux_string1 = ""
        #     aux_string2 = "_model"
        # end

        if run.model_name_sim == "in_sample"
            aux_string2 = ""
        else
            aux_string2 = "_model"
        end

        file_name = "C:/Users/cg4102/Documents/julia_logs/" * dir_name * "/Runs server 2025/Cut-sharing Iterations/Run_" * run.model_name_run * "_model_" * string(run.seed) * "/" * run.model_name_run * "_model_" * run.model_name_sim * aux_string2 * "_inflows_" * run.reservoir * ".txt"

        # Read data from CSV files
        inflow_df_run = read_txt_file_inflows(file_name, run.statistic)
        
        # Make column identifiable
        column_name = Symbol(run.model_name_run * "_" * run.model_name_sim * "_" * run.reservoir * "_" * run.statistic * "_" * string(run.seed))
        rename!(inflow_df_run, :InflowValue => column_name)
        
        # Add to plot
        create_plot(file_path_latex, run, inflow_df_run, number_of_stages)

        # Add to inflow_df
        if isempty(inflow_df)
            inflow_df = copy(inflow_df_run)
        else
            inflow_df = hcat(inflow_df, inflow_df_run)
        end
    end

    create_latex_plots_finisher(file_path_latex)
    Infiltrator.@infiltrate
end


function read_data(file_name::String, first_line_number::Int, number_of_rows::Int)

    df = CSV.read(file_name, header=false, skipto=first_line_number, limit=number_of_rows, delim=",", ignorerepeated=true, DataFrame)
    return df
end



function read_txt_file_inflows(file_name::String, statistic::String)

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

    return df
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

