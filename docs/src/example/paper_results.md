```@meta
EditURL = "paper_results.jl"
```

# Results from our experiments

## Result data

The full results from our computational experiments are available in the folder `Computational_Results`.
For a description on how to interpret the output files, see section [logging](logging.md).

## Reproducing the results

To reproduce the results from our paper, it is simply required to run the files `run_model.jl`, `run_model_linearized.jl` and `run_model_markov.jl` on the current main branch.

Note that for loglinearSDDP, **cut selection** is not supported yet, so one run of `run_model.jl` is sufficient. For the other two variants, we executed two batches of runs, one with and one without cut selection. The default case is that SDDP.jl is run with cut selection in both `run_model_linearized.jl` and `run_model_markov.jl`.

Unfortunately, the `train` function in SDDP.jl does not have an argument to control the cut selection scheme. Therefore, if `run_model_linearized.jl` and `run_model_markov.jl` should be run without cut selection, it is required to develop the SDDP.jl package and to manually switch the `cut_selection` parameter from `True` to `False` in the file `bellman_functions.jl` for function `add_cut.jl`.

Keep in mind that we used Gurobi 11 and older versions of JuMP.jl, SDDP.jl and several other packages when performing our experiments, so the results that you obtain may slightly deviate from what we report. The same can be true if the code is executed on a different hardware than ours.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

