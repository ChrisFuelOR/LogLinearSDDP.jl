# # How cuts are represented in the code

# As described in our paper, for any $t = 2,\ldots,T$ and any incumbent $(x_{t-1}, \xi_{[t-p:t-1]})$ cuts in loglinearSDDP can be expressed by formula 
# ```math
# \begin{equation}
# \begin{aligned}
# \mathcal{Q}_t(x_{t-1}, \xi_{[t-p:t-1]}) \geq \beta_{\bar{r} t}^\top x_{t-1} + \sum_{\tau = t}^T \sum_{\ell=1}^{L_\tau} \bigg( \alpha_{\bar{r} t \ell}^{(\tau)} \prod_{k=t-p}^{t-1} \prod_{m=1}^{L_k} \xi_{km}^{\Theta(t,\tau,\ell,m,k)} \bigg),
# \end{aligned}
# \end{equation}	
# ```
# with coefficients defined by
# ```math
# \begin{equation}
# \begin{aligned}
# \beta_{\bar{r} tj}^\top &:= - \big( \pi_{tj}^* \big)^\top T_{t-1}, \\
# \alpha_{\bar{r} t \ell j}^{(t)} &:= \pi_{t \ell j}^* e^{\gamma_{t \ell}} e^{\psi_{t \ell} \eta_{t \ell}^{(j)}}, \\
# \alpha_{\bar{r} t \ell j}^{(\tau)} &:= \Big( \sum_{r \in R_{t+1}} \rho^*_{rtj} \alpha_{r,t+1,\ell}^{(\tau)} \Big) \prod_{\nu=1}^{L_t} e^{\gamma_{t \nu} \Theta(t+1,\tau,\ell,\nu,t)} e^{\psi_{t \nu} \eta_{t \nu}^{(j)} \Theta(t+1,\tau,\ell,\nu,t)}, \\
# &\quad \tau = t+1,\ldots,T \\
# \end{aligned} 
# \end{equation}
# ```
# for $t=2,\ldots,T, \ell = 1,\ldots,L_\tau$ and $j=1,\ldots,q_t$,
# ```math
# \begin{equation}
# \begin{aligned}
# \beta_{\bar{r} t} := \sum_{j=1}^{q_t} p_{tj} \beta_{\bar{r} tj}, \quad \quad \alpha_{\bar{r} t \ell}^{(\tau)} := \sum_{j=1}^{q_t} p_{tj} \alpha_{\bar{r} t \ell j}^{(\tau)}, \quad \tau = t,\ldots,T,
# \end{aligned}
# \end{equation}
# ```
# and 
# ```math
# \begin{equation}
# \begin{aligned}
# \Theta(t,t,\ell,m,k) &:= \phi_{t \ell m}^{(t-k)} \\
# \Theta(t,\tau,\ell,m,k) &:= \begin{cases} \sum_{\nu=1}^{L_t} \big( \phi_{t \nu m}^{(t-k)} \Theta(t+1,\tau,\ell,\nu,t) \big) \\ \quad + \Theta(t+1,\tau,\ell,m,k), \quad & \text{if } k \geq t-p+1 \\ \sum_{\nu=1}^{L_t} \big( \phi_{t \nu m}^{(p)} \Theta(t+1,\tau,\ell,\nu,t) \big), & \text{if } k = t-p \end{cases} \\
# \end{aligned}
# \end{equation} 
# ```
# for $t=2,\ldots,T$, $\tau = t+1,\ldots,T$, $k=t-p,\ldots,t-1$, $\ell=1,\ldots,L_{\tau}$ and $m=1,\ldots,L_k$. $\bar{r}$ denotes the index of the new cut.

# For detailed explanations of all the terms in (1)-(4) and how they are derived, we refer to our paper.

# When a subproblem for a specific scenario is solved within loglinearSDDP, the cut intercepts of all existing cuts are adapted to said scenario by re-evaluating the second summand in (1) accordingly.

# In the code, the required information to handle these types of cuts is stored in structs of type `Cut`, as known from SDDP.jl. However, the struct is adjusted to satisfy the special form of the cuts above.

using JuMP

mutable struct Cut
    coefficients::Dict{Symbol,Float64}
    deterministic_intercept::Float64
    stochastic_intercept_tight::Float64
    intercept_factors::Array{Float64,2}
    trial_state::Dict{Symbol,Float64}
    constraint_ref::Union{Nothing,JuMP.ConstraintRef}
    cut_intercept_variable::Union{Nothing,JuMP.VariableRef}
    non_dominated_count::Int64
    iteration::Int64
end

# Its fields are defined as follows:
# *	`coefficients`: This is the cut gradient vector $\beta$. It can be computed using the values of certain dual variables.
# *	`deterministic_intercept`: This value is used to account for the contribution of deterministic constraints to the intercept. Handling them as coupling constraints is unnecessary from a memory perspective, but not taking them into account leads to a wrong intercept overall.
# *	`stochastic_intercept_tight`: This is the value of the full intercept at the incumbent, i.e. where the cut is constructed and tight. This is merely used for checks and to compute other values. It may also be used for cut selection purposes in the future.
# *	`intercept_factors` is not a scalar intercept as in standard SDDP, but a matrix of intercept factors $\alpha_{\bar{r} t \ell}^{(\tau)}$ for each $\tau=t,\ldots,T$ and each component $\ell$ of the process (see (3)).
# *	`trial state`; This is the incumbent where the cut is constructed.
# *	`cut_constraint`: This refers to the cut constraint in the JuMP model.
# *	`cut_intercept_variable`: This refers to an artificial variable in the JuMP model which is fixed to the cut intercept for a given, specific scenario.
# *	`non_dominated_count` is required for cut selection purposes. Cut selection is not supported yet but might be considered in the future.
# *	`iteration`: This value stores the iteration number in which the cut was constructed, which often coincides with index $\bar{r}$ above. This is used for logging and analyses.

# !!! note "Remark"
#     1. The exponents in (4) only have to computed once as they do not change between cuts, 
#     scenarios and iterations. 
#     2. Once the $\alpha$ factors in (2) and (3) are computed when a new cut is generated, they are fixed cut coefficients.
#     3. When cut intercepts are re-evaluated for a given scenario, then the scenario-specific factors 
#     $\prod_{k=t-p}^{t-1} \prod_{m=1}^{L_k} \xi_{km}^{\Theta(t,\tau,\ell,m,k)}$ in (2) are 
#     the same for all cuts, so they only have to be re-evaluated once for all cuts. Only the factors $\alpha_{\bar{r} t \ell}^{(\tau)}$ are specific to each cut.
