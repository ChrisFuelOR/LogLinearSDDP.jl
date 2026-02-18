# # Modeling log-linear AR processes

# The code is designed to deal with uncertainty in the RHS of the multistage problem that can be modeled by an autoregressive stochastic process that is described by formula

# ```math
# \begin{equation}
# \log(\xi_{t \ell}) = \gamma_{t \ell} + \sum_{k=1}^p \sum_{m=1}^{L_{t-k}} \phi^{(k)}_{t \ell m} \log(\xi_{t-k, m}) + \psi_{t \ell} \eta_{t \ell}.
# \end{equation}
# ```

# Here, $\gamma, \phi$ and $\psi$ are vectors and matrices of process coefficients that have to be estimated, $p$ is the lag order and $L$ is the dimension.
# We call these processes **log-linear** autoregressive processes, as they are linear functions in the natural logarithm of the process history.

# In LogLinearSDDP.jl, the definition of the log-linear AR process for a given model can be provided by the user using the `AutoregressiveProcess` and `AutoregressiveProcessStage` structs (both are defined in the file `src/typedefs.jl`).

# ## AutoregressiveProcess

using LogLinearSDDP

struct AutoregressiveProcess
    dimension::Int64
    lag_order::Int64
    parameters::Dict{Int64,LogLinearSDDP.AutoregressiveProcessStage}
    history::Dict{Int64,Vector{Float64}}
    simplified::Bool
end

# This struct can be used to store some general information on the stochastic process and its required history (lagged values $\xi_{t-k, m}$).

# The fields of `AutoregressiveProcess` are defined as follows:

# *	`dimension`:      
# Int64 which defines the dimension of the random process; denoted by $L$ in the paper
# *	`lag_order`:      
# Int64 which defines the lag order of the random process (same for each component and stage); denoted by $p$ in the paper
# *	`parameters`:     
# Dict containing the stage-specific data of the process. The key is the stage and the value is the actual data struct of type `AutoregressiveProcessStage`; one-dimensional with component $t$
# *	`history`:        
# Dict containing the historic values of the process (including stage 1). The key is the stage and the value is a vector of index $ℓ$ (alternatively, a tuple could be used).
# *	`simplified`:     
# If true, there are no dependencies between different process components (i.e. spatial dependencies). This allows to simplify some computations in our code. Referring to the paper, the process formula becomes

# ```math
# \begin{equation}
# \log(\xi_{t \ell}) = \gamma_{t \ell} + \sum_{k=1}^p \phi^{(k)}_{t \ell} \log(\xi_{t-k, \ell}) + \psi_{t \ell} \eta_{t \ell}.
# \end{equation}
# ```

# !!! note "Remarks"
#     1. We assume that the lag order $p$ is the same for all stages and components. Otherwise the cut formulas become way more sophisticated (see paper). In practice, different components and stages may require different lag orders, for instance in SPAR models. If a stage-component combination 
#     requires less lags than globally defined, we can set the `ar_coefficients` for excessive lags to 1, so that they do not have any effect.
#     2. In contrast to the paper - for this code we also assume that the dimension $L$ of the process is the same for all stages. This allows us to accelerate nested loops in the code with tools that do not  
#     allow for indices to be dependent on each other. This is a very natural assumption in practice. For instance, in our hydrothermal scheduling example, we have the same number of reservoirs for each stage.
#     3. We assume the first-stage data to be deterministic. Therefore, it should be included in `history` instead of `parameters`.


# ## AutoregressiveProcessStage

struct AutoregressiveProcessStage
    intercept::Vector{Float64}
    coefficients::Array{Float64,3}
    psi::Vector{Float64}
    eta::Vector{Any}
    probabilities::Vector{Float64}
end

# This struct can be used to store the specific parameters of the log-linear autoregressive process for a particular stage $t$. Note that parameters can be defined separately for each component $\ell$ (componentwise process definition, see our paper).

# The fields of `AutoregressiveProcess` are defined as follows:

# *	`intercept`:      
# Vector containing the intercepts of the process; one-dimensional with component $\ell$; denoted by $\gamma$ in the paper
# *	`coefficients`:   
# Array containing the coefficients of the process; three-dimensional with components $\ell$, $m$ and lag $k$; denoted by $\phi$ in the paper
# *	`psi`:            
# Vector containing the pre-factor for eta in the process formula; one-dimensional with component $\ell$; denoted by $\psi$ in the paper               
# *	`eta`:            
# Vector containing the stagewise independent realizations of the error term; denoted by $\eta$ in the paper
#     * Note that this information is merely stored in the struct for logging purposes and for setting up the optimization problem, but is not used in the actual algorithm.
#     * Each element of the vector should be a vector, tuple or named tuple of size $\ell$ in order to store values for different process components. This requirement is standard for SDDP.jl as well.
# *	`probabilities`:  
# Probabilities related to $\eta$ (optional); note that this information is merely stored in the struct for logging purposes

# We provide an example on how to set up a specific log-linear AR process for the hydrothermal scheduling problem from our paper on a different [page](example/setting_up_process.md).

# !!! note "Remark" 
#     Note that similar structs (`LinearAutoregressiveProcessStage` and `LinearAutoregressiveProcess`) are also defined in file `set_up_ar_process.jl` for our hydrothermal scheduling problem when  #     using linearized AR processes instead of log-linear AR processes.
