# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Copyright (c) 2023 Christian Fuellner <christian.fuellner@kit.edu>

# Note that this code reuses functions from SDDP.jl by Oscar Dowson,
# which are licensed under the Mozilla Public License, Version 2.0 as well. 
# Copyright (c) 2017-2023: Oscar Dowson and SDDP.jl contributors.
################################################################################

""" 
LOGGING
###########################################################################
At the beginning, the following things are logged.
    > BANNER
        >> see function print_banner()
    > MODEL RUN PARAMETERS
        >> file path, datetime and run description
        >> model run configuration, i.e. main model parameters (number of stages, realizations, used seeds),
            properties of the AR process, used solvers, algorithmic parameters (used stopping rule)
        >> see function print_parameters()
    > MODEL STATISTICS (borrowed from SDDP.jl)
        >> problem size, subproblem structure
        >> problem options
        >> numerical_stability report
    > HEADER FOR ITERATION LOGGING
        >> see function print_iteration_header()

After each iteration, the following things are logged.
    > ITERATION RESULTS
        >> iteration number
        >> lower bound, statistical upper bound / simulation result, gap (only for checks using deterministic models)
        >> total time, iteration time
        >> subproblem size (variables, constraints, added cuts) and finished solves
        >> see functions log_iteration() and print_iteration()

At the end, the following things are logged.
    > SDDP RESULT FOOTER
        >> result summary containing solution status, total time, lower bound, statistical upper bound / simulation value, numerical issue report
        >> see function print_footer()
    > TIMING SUMMARY
        >> table including the times and memory allocation for different subroutines of SDDP

"""

struct Log
    iteration::Int64
    bound::Float64
    simulation_value::Float64
    # current_state::Vector{Dict{Symbol,Float64}}
    time::Float64
    pid::Int64
    total_cuts::Int64
    active_cuts::Int64
    total_solves::Int64
    duality_key::String
    serious_numerical_issue::Bool
end

struct Results
    status::Symbol
    log::Vector{LogLinearSDDP.Log}
end


function print_helper(f, io, args...)
    f(stdout, args...)
    f(io, args...)
end


function print_banner(io)
    println(io)
    println(io)
    println(io,"#########################################################################################################################################",)
    println(io,"#########################################################################################################################################",)
    println(io, "LogLinearSDDP.jl (c) Christian Füllner, 2023")
    println(io, "re-uses code from SDDP.jl (c) Oscar Dowson, 2017-2023")
    flush(io)
end


function print_parameters(io, 
    algo_params::LogLinearSDDP.AlgoParams, 
    problem_params:: LogLinearSDDP.ProblemParams, 
    applied_solver:: LogLinearSDDP.AppliedSolver, 
    ar_process::LogLinearSDDP.AutoregressiveProcess,
    ) 

    # Printint the file name
    println(io, "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
    println(io, "PATH")
    println(io, "calling ")
    println(io, @__DIR__)
    println(io, Base.source_path())

    # Printing the time
    println(io, "DATETIME")
    println(io, Dates.now())

    # Printing the run description
    println(io, "RUN DESCRIPTION")
    println(io, algo_params.run_description)

    ############################################################################
    # PRINTING THE PARAMETERS USED
    ############################################################################
    println(io, "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
    println(io, "STOPPING RULES")
    if isempty(algo_params.stopping_rules)
        println(io, "No stopping rule defined.")
    else
        for stopping_rule in algo_params.stopping_rules
            println(io, stopping_rule)
        end
    end

    println(io, "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
    println(io, "APPLIED SOLVER")
    println(io, applied_solver)
    println(io, algo_params.solver_approach)
    println(io, "Numerical focus: ", algo_params.numerical_focus)
    println(io, "Silent: ", algo_params.silent)

    if !isnothing(algo_params.forward_pass_seed)
        println(io, "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        println(io, "SAMPLING")
        println(io, "Used seed for sampling scenarios (in forward pass): ")
        print(io, rpad(Printf.@sprintf("%s", algo_params.forward_pass_seed), 10))
        println(io)
    end

    println(io, "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
    println(io, "PROBLEM DESCRIPTION")
    println(io, "Number of stages: ", problem_params.number_of_stages)
    println(io, "Number of realizations per stage: ", problem_params.number_of_realizations)

    if !isnothing(problem_params.tree_seed)
            println(io, "Seed for scenario tree sampling: ", problem_params.tree_seed)
    end

    println(io, "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
    println(io, "AUTOREGRESSIVE PROCESS DESCRIPTION")
    println(io, "Model approach: ", algo_params.model_approach)
    println(io, "Lag order: ", ar_process.lag_order)
    println(io, "Max dimension of noises: ", LogLinearSDDP.get_max_dimension(ar_process))
    println(io)

    #TODO: What about the AR data (intercept, coefficients, eta)?
    #TODO: What about the initial process data?    

    flush(io)
end 


function print_iteration_header(io)

    rule = "─"
    rule_length = 200

    #total_table_width = sum(textwidth.((sec_ncalls, time_headers, alloc_headers))) + 3
    printstyled(io, "", rule^rule_length, "\n"; bold=true)

    header = "It.#"
    print(io, rpad(Printf.@sprintf("%s", header), 6))
    print(io, "  ")
    header = "LB"
    print(io, lpad(Printf.@sprintf("%s", header), 13))
    print(io, "  ")
    header = "Sim."
    print(io, lpad(Printf.@sprintf("%s", header), 13))
    print(io, "  ")
    header = "Gap"
    print(io, lpad(Printf.@sprintf("%s", header), 8))
    print(io, "  ")
    header = "Time"
    print(io, lpad(Printf.@sprintf("%s", header), 13))
    print(io, "  ")
    header = "It_Time"
    print(io, lpad(Printf.@sprintf("%s", header), 13))
    print(io, "  ")
    header = "#Tot. Cuts"
    print(io, lpad(Printf.@sprintf("%s", header), 16))
    print(io, "       ")
    header = "#Act. Cuts"
    print(io, lpad(Printf.@sprintf("%s", header), 16))
    print(io, "       ")
    println(io)

    header = ""
    print(io, rpad(Printf.@sprintf("%s", header), 53))
    header = "[%]"
    print(io, lpad(Printf.@sprintf("%s", header), 8))
    print(io, "  ")
    header = "[s]"
    print(io, lpad(Printf.@sprintf("%s", header), 13))
    print(io, "  ")
    print(io, "                ")
    print(io, "                ")
    header = "Total"
    print(io, lpad(Printf.@sprintf("%s", header), 13))
    print(io, "  ")
    header = "pid"
    print(io, lpad(Printf.@sprintf("%s", header), 9))
    print(io, "  ")
    # header = "Total"
    # print(io, lpad(Printf.@sprintf("%s", header), 7))
    # print(io, "  ")

    println(io)

    printstyled(io, "", rule^rule_length, "\n"; bold=true)

    flush(io)
end

function print_iteration(io, log::LogLinearSDDP.Log, start_time::Float64)
    print(io, rpad(Printf.@sprintf("%-5d", log.iteration), 6))
    print(io, "  ")
    print(io, lpad(Printf.@sprintf("%1.6e", log.bound), 13))
    print(io, "  ")
    print(io, lpad(Printf.@sprintf("%1.6e", log.simulation_value), 13))
    print(io, "  ")

    gap = abs(log.simulation_value - log.bound)/(max(log.simulation_value, log.bound + 1e-10))
    #TODO: only meaningful for deterministic problems

    print(io, lpad(Printf.@sprintf("%3.4f", gap), 8))
    print(io, "  ")
    print(io, lpad(Printf.@sprintf("%1.6e", log.time), 13))
    print(io, "  ")
    print(io, lpad(Printf.@sprintf("%1.6e", log.time - start_time), 13))
    print(io, "  ")
    print(io, lpad(Printf.@sprintf("%5d", log.total_cuts), 7))
    print(io, "  ")
    print(io, lpad(Printf.@sprintf("%5d", log.active_cuts), 7))
    print(io, "  ")
    print(io, lpad(Printf.@sprintf("%5d", log.total_solves), 7))
    print(io, "  ")
    print(io, log.serious_numerical_issue ? "†" : " ")
    print(io, log.duality_key)
    print(io, "  ")
    print(io, rpad(Printf.@sprintf("%-1d", log.pid), 6))

    println(io)

    flush(io)
end


function log_iteration(algo_params::LogLinearSDDP.AlgoParams, log_file_handle::Any, log::Vector{LogLinearSDDP.Log})
    if algo_params.print_level > 0 && mod(length(log), algo_params.log_frequency) == 0
        # Get time() after last iteration to compute iteration specific time
        if lastindex(log) > 1
            start_time = log[end-1].time
        else
            start_time = 0.0
        end

        print_helper(print_iteration, log_file_handle, log[end], start_time)
    end
end


function print_simulation(io, algo_params::LogLinearSDDP.AlgoParams, μ::Float64, ci::Float64, lower_bound::Float64, description::String)

    println(io)
    println(io, "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
    println(io, "SIMULATION RESULTS")
    println(io, algo_params.simulation_regime)
    println(io, description)
    println(io, "Lower bound: ", lower_bound)
    println(io, "Statistical upper bound (confidence interval): ", μ, " ± ", ci )
    println(io, "Pessimistic upper bound: ", μ + ci )
    flush(io)
end