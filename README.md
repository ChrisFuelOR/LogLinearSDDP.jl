# LogLinearSDDP

This code is intended for using a special version of SDDP for multistage stochastic linear problems with log-linear autoregressive uncertainty in the right-hand side.

For simplicity, it is assumed that this uncertainty in the RHS is the only uncertainty in the problem, even though in theory also stagewise independent uncertainty in other problem elements is allowed.

The definition of the autoregressive process has to be provided by the user by means of the autoregressive_data_stage struct.

For now, this version of SDDP is restricted to problems satisfying the following properties
1.) finitely many stages
2.) linear constraints and objective
3.) continuous decision variables
4.) log-linear autoregressive uncertainty in the RHS
5.) finite support of the uncertainty
6.) uncertainty is exogeneous
7.) expectations considered in the objective (no other risk measures)
8.) no usage of believe or objective states

Additionally, this version of SDDP is restricted to using a SINGLE_CUT approach.
A MULTI_CUT approach is not supported yet.

Importantly, this package does not require the user to model an explicit state 
expansion for the given problem to take the history of the AR process into account.
Instead, tailored cut formulas are used to adapt the cut intercept to a scenario at hand.