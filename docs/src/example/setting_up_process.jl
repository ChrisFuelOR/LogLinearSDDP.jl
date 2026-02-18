# # Setting up the autoregressive processes

# For both, log-linear and linearized AR processes, the processes are set up using functions in `set_up_ar_process.jl`.

# As inputs for this set-up three sources are used:

# *	For each reservoir system (N, S, NE, SE) and for each month, a log-linear AR model was fitted using the historical data (for more details, see [model fitting](fitting_process.md)). The estimates are stored in folder `AutoregressivePreparation` and then `bic_model` (or `custom_model`) in some txt-files (for the linear case it is `LinearizedAutoregressivePreparation`). Each txt-file contains the months, the lag order, the process intercept, the process coefficients, the factor multiplied with the error term and the standard error from the estimation.
# *	For each reservoir system, some historical inflow values are provided in the same folder in the txt-file `history_nonlinear.txt`. It contains a sufficient number of time steps for the lag order of the processes.
# *	For each reservoir system and each stage, 100 realizations of the stagewise independent noise term $\eta_t$ are provided in the same folder in the txt-file `scenarios_nonlinear.txt`.

# In the set-up process, first the above data is read from the source files. Then it is used to create structs of type `AutoregressiveProcessStage` and `AutoregressiveProcess`.

# As loglinearSDDP requires a constant lag order, the maximum over all reservoirs and months is used.

