# # Running computations

# To run loglinearSDDP, we use file `run_model.jl`. It allows to specify several runs with different parameter configurations. Those runs are executed one after each other.

# For each run, we provide a random seed (used for the sampling in the forward pass), a specific log-linear model for the uncertain data and additional log-linear and linearized models that are used for the out-of-sample simulations after SDDP has terminated.

# The `run_model` function has the following general structure:

#  * Define main model parameters. For out tests, we always used 120 stage, 100 realizations per stage and 2000 scenarios in the simulations.
#  * Set the file path (this has to be adjusted to the user’s system when trying to reproduce our results).
#  * Use the previously defined parameters to set up structs of type `ProblemParams` and `AlgoParams` (see definitions in [Further parameters](../params.md)).
#  * Define a simulation regime that will be used for the in-sample simulation after SDDP has terminated.
#  * Set up the log-linear AR process by calling function `set_up_ar_process_loglinear` from `set_up_ar_process.jl`.
#  * Call `model_definition` from `hydrothermal_model.jl` to construct the multistage optimization problem. This requires to pass the previously defined process as an argument.
#  * Call the `train_loglinear` function from `algorithm.jl` to start running SDDP.
#  * After running SDDP, perform several simulations. For details, see [simulations](simulations.md).
#  * The function `extended_simulation_analysis` is used to analyze and log parts of the simulation output.

# !!! note "Remarks"
#     For running different variants of SDDP, the procedure is very similar (see `run_model_linearized.jl` or `run_model_markov.jl`).
