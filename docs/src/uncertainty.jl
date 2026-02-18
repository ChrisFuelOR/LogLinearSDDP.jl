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

# ## Autoregressive Process

struct AutoregressiveProcess
    dimension::Int64
    lag_order::Int64
    parameters::Dict{Int64,AutoregressiveProcessStage}
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
#     We we assume that the lag order is the same for all stages and components. Otherwise the cut  
#     formulas become way more sophisticated (see paper). In practice, different components and stages  #     may require different lag orders, for instance in SPAR models. If a stage-component combination 
#     requires less lags than globally defined, we can set the ar_coefficients corresponding to 
#     excessive lags to 1, so that they do not have any effect.

