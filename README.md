# LogLinearSDDP - Stochastic Dual Dynamic Programming for Log-linear Autoregressive Uncertainty in the Right-hand Side

The software and data in this repository are a snapshot of the software and data that were used in the research reported on in the [paper]() ``Stochastic Dual Dynamic Programming for Log-linear Autoregressive Uncertainty in the Right-hand Side'' by C. Füllner and S. Rebennack. 

The code contains a special version of the stochastic dual dynamic programming (SDDP) algorithm designed for multistage stochastic linear problems with log-linear autoregressive uncertainty in the right-hand side.
It is based on earlier versions of the package [SDDP.jl](https://github.com/odow/SDDP.jl) from O. Dowson.


## Cite

TODO


## Description

The goal of this software is to demonstrate the effect of our proposed feasibility verification and upper bound computation procedure 
for global optimization with continuous variables and possibly non-convex inequality and equality constraints.
The procedure is integrated into a spatial branch-and-bound method. 

The feasibility verification method has two main ingredients: The first one is a reformulation of inequality constraints based on
so-called approximate active index sets. The second one is the Miranda theorem based feasibility verification method for purely box- and
equality-constrained problems that was presented in an earlier [paper](https://link.springer.com/article/10.1007/s10107-020-01493-2) by Füllner, Kirst and Stein.

The main contribution is that under certain assumptions, the proposed method is sufficient to
guarantee convergence of the spatial branch-and-bound method.

For comparison, alternative feasibility verification procedures based on interval Newton methods can be used.






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







## Building

Our experiments where conducted using Python 3.7 with the package versions as specified in the
requirements.txt file.

To replicate the experiments or test the code on different problems, make sure that you run

```
pip install -r requirements.txt
```

to install the required packages.

## Running the code

The code can be run by executing the files "exe.py" (for our proposed method) or "exe_newton.py" (for comparison with different methods)
in the "scripts" directory.

The directory also contains a README file providing more details on how to prepare experiments and which parameters
can be set by the user.

## Results

The results of our computational tests as presented in the paper are stored in the "results" directory.
It contains a README file providing more details on the interpretation and replication of the results.

## Test Problems

The data for the considered test problems is stored in the "data" directory.
It contains a README file providing more details on the structure of the problems and how to create new ones.


## Support

For support in using this software, submit an issue in this repository.