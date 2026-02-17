using DataFrames
using Infiltrator
using CSV

struct RunConfigData
    seed::Int
    model_name_run::String
    model_name_sim::String
    stopping::String
    statistic::String
    color::String
end



function evaluate_results_data()

    # Define runs that should be considered
    runs_to_consider = (
        RunConfigData(11111, "bic", "shapiro", "IterLimit", "mean", "red"),
        RunConfigData(11111, "custom", "shapiro", "IterLimit", "mean", "blue"),
        RunConfigData(11111, "fitted", "shapiro", "IterLimit", "mean", "green!70!black"),
        RunConfigData(11111, "shapiro", "shapiro", "IterLimit", "mean", "cyan"),
        #RunConfigData(11111, "custom", "bic", "IterLimit", "mean", "blue"),
        #RunConfigData(11111, "custom", "custom", "IterLimit", "mean", "red"),
        #RunConfigData(11111, "custom", "fitted", "IterLimit", "mean", "green!70!black"),
        #RunConfigData(11111, "custom", "shapiro", "IterLimit", "mean", "cyan"),    
        #RunConfigData(11111, "custom", "in_sample", "IterLimit", "mean", "black"),    
    )

    # Define max number of stages
    number_of_stages = 60

    # Dataframe to store data
    data_df = DataFrame()

    # Iterate over runs and read data from CSV files
    for run_index in eachindex(runs_to_consider)

        run = runs_to_consider[run_index]

        # Construct path and file name
        if run.stopping == "TimeLimit"
            dir_name = "Cut-sharing - Time"
        elseif run.stopping == "IterLimit"
            dir_name = "Cut-sharing"
        else
            Error("Stopping criterion not recognized.")
        end

        if run.model_name_sim == "in_sample"
            aux_string2 = ""
        else
            aux_string2 = "_model"
        end

        file_name = "C:/Users/cg4102/Documents/julia_logs/" * dir_name * "/Runs server 2025/Cut-sharing New - Iterations/Run_" * run.model_name_run * "_model_" * string(run.seed) * "/" * run.model_name_run * "_model_" * run.model_name_sim * aux_string2 * "_data.txt"

        # Read data from CSV files
        df_run = read_txt_file_data(file_name, run.statistic)
       
        df_run = select!(df_run, [:DeficitCost])

        for name in ["DeficitCost"]
        #for name in ["Gen", "HydGen", "Deficit", "Exchange", "Spillage"]      
        # name in ["Gen", "HydGen", "Deficit", "DeficitCost", "Exchange", "Spillage"]                     

            # Make column identifiable
            column_name = Symbol(name * "_" * run.model_name_run * "_" * run.model_name_sim * "_" * run.statistic)
            rename!(df_run, Symbol(name) => column_name)
        
            # Get plot name
            file_path_latex = "C:/Users/cg4102/Documents/julia_plots/Cut_sharing 2025/" * lowercase(name) * "_All_Sha_mean.tex"

            # Get axis limits
            axis_limits = [0.0, Float64(number_of_stages), 0.0, 0.0]

            if name == "Gen"
                axis_limits[4] = 20000.0
            elseif name == "HydGen"
                axis_limits[3] = 40000.0
                axis_limits[4] = 70000.0
            elseif name == "Deficit"
                axis_limits[4] = 2000.0
            elseif name == "DeficitCost"
                axis_limits[4] = 10000000.0
            elseif name == "Exchange"
                axis_limits[3] = 5000.0
                axis_limits[4] = 15000.0
            elseif name == "Spillage"
                axis_limits[4] = 15000.0
            end

            # Start plot
            if run_index == 1
                create_latex_plots_starter(file_path_latex, axis_limits, name) 
            end

            # Add to respective plot
            create_plot(file_path_latex, run, df_run[:, column_name], number_of_stages)
        
            # End plot
            if run_index == last(eachindex(runs_to_consider))
                create_latex_plots_finisher(file_path_latex)
            end

        end

        # Add to overall df
        if isempty(data_df)
            data_df = copy(df_run)
        else
            data_df = hcat(data_df, df_run)
        end
    end   

end

function read_data(file_name::String, first_line_number::Int, number_of_rows::Int)

    df = CSV.read(file_name, header=false, skipto=first_line_number, limit=number_of_rows, delim=",", ignorerepeated=true, DataFrame)
    return df
end

function read_txt_file_data(file_name::String, statistic::String)

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
    df = select!(df, Not([:Column1]))

    # Generation
    rename!(df, :Column2 => :Gen)

    # Hydro Generation
    rename!(df, :Column3 => :HydGen)

    # Deficit
    rename!(df, :Column4 => :Deficit)

    # Deficit Cost
    rename!(df, :Column5 => :DeficitCost)

    # Exchange
    rename!(df, :Column6 => :Exchange)

    # Spillage
    rename!(df, :Column7 => :Spillage)
    df.Spillage .= replace.(df.Spillage, r"\)" => "")
    df.Spillage = parse.(Float64, df.Spillage)

    return df
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

function create_latex_plots_starter(file_path_latex::String, axis_limits::Vector{Float64}, label::String)

    io = open(file_path_latex, "w")
	create_latex_header(io, axis_limits, label)
    close(io)

    return
end

function create_latex_plots_finisher(file_path_latex::String)

    io = open(file_path_latex, "a")
	create_latex_footer(io)
    close(io)

    return
end

function create_plot(file_path_latex::String, run_config::RunConfigData, results::Vector{Float64}, number_of_stages::Int)

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
	for row in 1:number_of_stages
        print(io, "(   ")
        print(io, row)
        print(io, "   ,   ")
        print(io, results[row])
        print(io, ")   ")
        println(io)
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

