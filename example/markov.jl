# # Special remarks to MC-SDDP

# When using MC-SDDP, the model does not have to be defined manually (as in `hydrothermal_model.jl`), but is read from  `(12_0)_100.problem.json`. We have made sure that the model specified there exactly matches the model considered in `hydrothermal_model.jl` and `hydrothermal_model_linearized.jl`.

# Additionally, also the lattice (Markov chain) data is read from `(12_0)_100.lattice.json`. Reading this data can take some time, which is not included in the overall run time, though, to allow a fair comparison with other variants of SDDP.

# The problem and lattice data are taken from the [MSPLib](https://github.com/bonnkleiford/MSPLib-Library).

# Runs for MC-SDDP are conducted using `run_model_markov.jl`. The structure is very similar to `run_model.jl`. An important difference is that we can specify a `forward_pass_model`. When it is set to `lattice`, the scenario lattice itself is used for sampling within SDDP. Otherwise, we may also use inflows from the log-linear or linearized AR models (similar to the out-of-sample simulations). As these inflow values may not be included in the discrete lattice, we use function `closest_nod` from `markov.jl` to compute the closest lattice node to an inflow realization. 

# When solving a particular subproblem in the forward pass of SDDP, we use the exact sampled inflow value in the RHS. However, we then use the closest lattice node to decide which node is visited next in the forward pass. This is a standard approach in the literature when using scenario lattices combined with out-of-sample data.
