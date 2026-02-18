```@meta
EditURL = "../src/model_assumptions.jl"
```

# Model assumptions

_This tutorial was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl)._
[_Download the source as a `.jl` file_](model_assumptions.jl).
[_Download the source as a `.ipynb` file_](model_assumptions.ipynb).

Our version of SDDP is designed for multistage stochastic linear problems with log-linear autoregressive uncertainty in the right-hand side (RHS). For simplicity, we assume that we face only uncertainty of data in the RHS, even though in theory also stagewise independent uncertainty in other problem components is allowed.

Currently, the code is restricted to problems satisfying the following properties:
*	finitely many stages $T$
*	linear constraints and objective
*	continuous decision variables
*	log-linear autoregressive uncertainty in the RHS
*	finite support of the uncertainty
*	deterministic first stage
*	uncertainty is exogeneous (not decision-dependent)
*	expected value is considered in the objective (no other risk measures)
*	no usage of believe or objective states compared to SDDP.jl

````@example model_assumptions
```math
\begin{aligned}
\min_{x_t}& &&c_t^\top x_t + \evfttwo[x_t][\xi_{[t]}]{t+1} \\
\text{s.t.}& &&T_{t-1} x_{t-1} + W_t x_t = h_t(\xi_t) = \xi_t \\
&&&x_t \geq 0,
\end{aligned}
```
````

Additionally, this version of SDDP is restricted to using a single-cut approach (see parameter `SINGLE_CUT` in the code), where the expected value functions are approximated by one set of cuts per stage. A multi-cut approach (`MULTI_CUT`), where for each stage value functions for different scenarios are approximated by separate sets of cuts, is not supported yet.

TODO: Maybe add the problem formulation from the paper

In general, the multistage problems can be formulated in the same way as when using SDDP.jl (for a general introduction, see TODO). The only difference is that the uncertainty has to be modeled differently to account for its stagewise dependent character. We explain in more detail below how the uncertainty should be defined (TODO).

Importantly, contrary to other approaches in the literature, our version of SDDP does not require the user to model an explicit state expansion for the given problem to take the history of the AR process into account. Instead, tailored cut formulas are used to adapt the cut intercept to a scenario at hand. We provide an example for a hydrothermal scheduling problem below to further clarify this.

