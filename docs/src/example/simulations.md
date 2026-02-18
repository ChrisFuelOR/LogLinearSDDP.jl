```@meta
EditURL = "simulations.jl"
```

# Simulations

We run several simulations (in-sample and out-of-sample) after SDDP has terminated.

Each simulation run consists of the following three ingredients:

 * Create a sampling scheme from the ones available in SDDP.jl (`InSampleMonteCarlo`, `OutOfSampleMonteCarlo`, `Historical`)
     * For `InSampleMonteCarlo`, we sample from the realizations passed in the model formulation.
     * For `OutOfSampleMonteCarlo`, we pass a function to the sampling scheme which defines how we sample realizations later.
     * For `Historical`, we pass a list of historical realizations to the sampling scheme from which we sample later.
 * Create an struct of type `LogLinearSDDP.Simulation` which stores this sampling scheme as well as the number of replications (and optionally a random seed).
 * Call a `simulation` function that uses the information from this struct to execute the simulation. It is responsible for calling the correct `sample_scenario` method and passing the correct sampling scheme information.

Importantly, for each variant of SDDP that we ran and the uncertainty model that we want to use for the simulation, these steps have to be adjusted accordingly.

For instance, if we want to use data from the lo-linear models to simulate the policy that we obtained when running standard SDDP with inflows from the linearized models, then we have to make sure that the out-of-sample inflows from the log-linear models are provided in a way that matches the model formulation and the parameterize function defined in `hydrothermal_model_linearized.jl`.

For this reason, our code contains a lot of variations of the same functions. An overview is given in the table below.

![Overview of simulation code](../assets/Simulations_table.PNG)

Note that for out-of-sample simulations of Markov-chain SDDP policies, the inflows are not sampled during the simulation but read from inflow files in folder `PreparationMarkov` which have been generated using the log-linear and the linearized AR processes in advance.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

