```@meta
EditURL = "model_assumptions.jl"
```

# Model assumptions

loglinearSDDP is designed for multistage stochastic linear problems with log-linear autoregressive uncertainty in the right-hand side (RHS). For simplicity, we assume that we face only uncertainty of data in the RHS, even though in theory also stagewise independent uncertainty in other problem components is allowed.

As described in our paper, this type of multistage problem can be described by coupled subproblems for stages $t=1,\ldots,T$, each of them of the form
```math
\begin{aligned}
&Q_t(x_{t-1}, \xi_t) :=
\begin{cases}
\begin{aligned}
\min_{x_t}& &&c_t^\top x_t + \mathcal{Q}_{t+1}(x_t, \xi_{[t]}) \\
\text{s.t.}& &&T_{t-1} x_{t-1} + W_t x_t = \xi_t \\
&&&x_t \geq 0,
\end{aligned}
\end{cases}
\end{aligned}
```
where $x_t$ is the decision variable, $\xi_t$ denotes the uncertain data, $Q_t$ is the so-called **value function** for stage $t$ and $\mathcal{Q}_{t+1}$ is the so-called **expected value function** for stage $t+1$.

Currently, the theory and the code are restricted to problems satisfying the following properties:
*	finitely many stages $T$
*	linear constraints and objective
*	continuous decision variables
*	log-linear autoregressive uncertainty in the RHS
*	finite support of the uncertainty
*	deterministic first stage
*	uncertainty is exogeneous (not decision-dependent)
*	expected value is considered in the objective (no other risk measures)
*	no usage of believe or objective states compared to SDDP.jl

Additionally, so far our code is restricted to using a single-cut approach (see parameter `SINGLE_CUT` in the code), where the expected value functions $\mathcal{Q}_t$ are approximated by one set of cuts per stage. A multi-cut approach (`MULTI_CUT`), where for each stage value functions $Q_t$ for different scenarios are approximated by separate sets of cuts, is not supported yet.

Most of these assumptions are in line with what is known from using SDDP.jl. The major difference is that in our case the uncertainty has to be modeled differently to account for its autoregressive character. More details on this are provided in [Modeling log-linear AR processes](uncertainty.md)

Importantly, contrary to other approaches in the literature, loglinearSDDP does not require the user to model an explicit state expansion for the given problem to take the history of the AR process into account. Instead, tailored cut formulas are used to adapt the cut intercept to a scenario at hand. We provide an example for a hydrothermal scheduling problem below to further clarify this [TODO].

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

