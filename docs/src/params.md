```@meta
EditURL = "params.jl"
```

# Further parameters

There are two more structs that are relevant for the user when running loglinearSDDP.

## ProblemParams

````@example params
struct ProblemParams
    number_of_stages::Int64
    number_of_realizations::Int64
    tree_seed::Union{Nothing,Int}
    gurobi_coupling_index_start::Union{Nothing,Int}
    gurobi_cut_index_start::Union{Nothing,Int}
    gurobi_fix_start::Union{Nothing,Int}
end
````

This struct is used to store some parameters of the test problem that is solved. This is mainly used for logging purposes.

Its fields are defined as follows:

 * `number_of_stages`: number of stages of the multistage problem
 * `numer_of_realizations`: number of realizations of the stagewise independent noise $\eta$
 * `tree_seed`: seed for the scenario tree generation
 * **gurobi parameters**: these parameters refer to Gurobi-internal indices of coupling constraints, cut constraints and cut intercept variables; if they are known, they can be provided by the user to speed up parts of the computations (see more details below)

Note that the size of `eta` in `AutoregressiveProcessStage` should match `number_of_realizations` defined in `ProblemParams`.

## AlgoParams

````@example params
using LogLinearSDDP, SDDP

mutable struct AlgoParams
    stopping_rules::Vector{SDDP.AbstractStoppingRule}
    simulation_regime::LogLinearSDDP.AbstractSimulationRegime
    cut_selection::Bool
    print_level::Int64
    log_frequency::Int64
    log_file::String
    run_numerical_stability_report::Bool
    numerical_focus::Bool
    silent::Bool
    forward_pass_seed::Union{Nothing,Int}
    run_description::String
    model_approach::Symbol
end
````

This struct is used to store some parameters that control the SDDP algorithm that is used to solve the model. Many of these parameters are the same that the `train` function in SDDP.jl uses as arguments.

Its fields are defined as follows:

 * `stopping_rules`: A vector of different SDDP stopping rules (same as for SDDP.jl).
 * `simulation_regime`: Defines which type of simulation should be used for the in-sample simulations (see [simulations](example/simulations.md)).
 * `cut_selection`: Controls if a cut selection scheme should be applied. Note that this is not supported yet but might be added in the future, so we added a placeholder.
 * `print_level`, `log_frequency`, `log_file`: As for `train` in SDDP.jl, these parameters control the logging of the SDDP output
 * `run_numerical_stability_report`: As for `train` in SDDP.jl, this parameter controls the generation of numerical stability reports. Note that this is only a placeholder so far and should always be set to `False`.
 * `numerical_focus`: If set to `True`, Gurobi will set its numerical_focus parameter to `True`, which may help in case of numerical issues but will also slow down the solver.
 * `silent`: Controls if the solver output is printed on the console.
 * `forward_pass_seed`: Seed that is used for sampling realizations of the stagewise independent noise $\eta$ in the forward pass of SDDP.
 * `run_description`: Allows the user to add some description to a specific model run that is used in the log-file.
 * `model_approach`: For logging purposes, stores which concrete model is used for the stochastic process (i.e. LOG-BIC, LOG-1, LIN-FIT or LIN-SHA for our hydrothermal scheduling problem, see [experimental set-up](example/experiment_description.md)).

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

