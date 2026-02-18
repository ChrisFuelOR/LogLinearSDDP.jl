```@meta
EditURL = "logging.jl"
```

# Interpreting the logging results

For each run in our experiments, several outputs are logged. In general, there are three types of output files:

*	A main logging file called `LogLinearSDDP.log` (or `LinearSDDP.log` or `MC-SDDP.log` depending on the chosen variant of SDDP).
*	Some files which log simulation outputs (e.g. inflows, volumes, costs for different reservoirs). They are named something like `bic_model_custom_model_volumes_N.txt`. This means that the policy obtained using the `bic_model` is simulated with out-of-sample data from the `custom_model` and that the file contains the volumes of reservoir N. These outputs are required for some analyses in the paper.
*	A file `inflows.txt` that logs inflows values. This was used for (a) checks if inflows were indeed consistent between out-of-sample simulations for different policies and (b) to generate the inflow data input files for `run_markov.jl`.

We explain the main logging file `LogLinearSDDP.log` in more detail.
It consists of the following elements:

 * A section borrowed from SDDP.jl logging some general information about the multistage problem
 * A section printing the main model run parameters
     * the file path, the optional run description, the date and time of the run
     * the main parameters defined in the `ProblemParams` struct (problem size, number of realizations per stage)
     * the main parameters defined in the `AlgoParams` struct (sampling seed, stopping rules)
     * the main properties of the uncertainty model (the used model approach, the lag order, the dimension)
 * A section logging information from the SDDP iterations. Each row contains
     * the iteration number
     * the deterministic lower bound
     * the simulated upper bound and the gap (both are not relevant for our experiments)
     * the total time and the time for the specific iteration
     * the total number of cuts created and the active number of cuts (which differs if a cut selection scheme is used)
 * A section borrowed from SDDP.jl summarizing the SDDP results
     * total time
     * stopping status
     * best deterministic lower bound
     * simulated upper bound and confidence interval
 * A table summarizing timing and memory allocation information for different steps of the algorithm
 * A section containing the results of in-sample and out-of-sample simulations after SDDP has terminated. In each case the information contains
     * the used uncertainty model
     * the deterministic lower bound
     * the simulated upper bound and a confidence interval

Note that each simulation run accounts for two logs of simulation results, one including costs for all 120 stages and one including only costs for the first 60 stages (to remove the end-of-horizon effect). The latter are reported in the paper.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

