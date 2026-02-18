# An introduction to LogLinearSDDP.jl

This code implements a special version of **stochastic dual dynamic programming (SDDP)** and is based on (an earlier version of) [SDDP.jl](https://github.com/odow/SDDP.jl) from Oscar Dowson.

The special version of SDDP, which we refer to as **loglinearSDDP**, is tailored to solve multistage stochastic linear problems with **log-linear autoregressive uncertainty** in the right-hand side.
By this we mean that the uncertain data in the right-hand side is stagewise dependent and described by an autoregressive process that is linear in the natural logarithm of its lagged variables.


## License

`LogLinearSDDP.jl` is licensed under the [MPL 2.0 license](https://github.com/ChrisFuelOR/LogLinearSDDP.jl/blob/main/LICENSE).

## Documentation

You can find the documentation at [https://chrisfuelor.github.io/LogLinearSDDP.jl/](https://chrisfuelor.github.io/LogLinearSDDP.jl/).

## Help

If you need help, please [open a GitHub issue](https://github.com/ChrisFuelOR/LogLinearSDDP.jl/issues).
