# # How to use the code

# ## Setting up the repository

# Note that this code was developed a few years ago and only tested on Julia 1.9.2 and with SDDP.jl 1.6.6. From today’s perspective, the Julia version and many of the packages that are used may be outdated. 

# We have not managed yet to update the code in order to guarantee compatibility with all newer package versions. However, when we used `Pkg.instantiate()` to set up our repository on a different machine, we have only experienced some issues with newer versions of SDDP.jl. For SDDP.jl it seems that at least the `set_objective` function has changed compared to the one we use.

# Further note that running loglinearSDDP for our [hydrothermal scheduling problem](example/experiment_description.md) requires an installation of Gurobi together with a valid Gurobi license. If this requirement is not satisfied, the solver has to be changed in the run-files and a few adjustments have to be made in the code.


# ## Running the code

# Once the repository is set up, the code for our hydrothermal scheduling problem can be run using the following steps.

# * Open Julia (1.9).
# * Use `cd` to navigate to the repository directory.
# * Go to package mode and execute `activate .`.
# * It is possible that the Gurobi path has to be made known. If so, use `ENV["GUROBI_HOME] = [path]` where [path] is the file path of your Gurobi installation.
# * It is possible that the Gurobi package has to be built for first usage. If so, use package mode and `build Gurobi`.
# * Use `cd` to navigate to `examples/Hydrothermal`.
# * Execute `run_model.jl` (or other run-files) with `include("run_model.jl")`.

# Before using these steps, you have to make sure that the file paths in `run_model.jl`, `run_model_linearized.jl`, `run_model_markov.jl`, `markov.jl` and `create_inflows_for_Markov_loglinear.jl` are set as intended. The places are highlighted by TODO in the code.
