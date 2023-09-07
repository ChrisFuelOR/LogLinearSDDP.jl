module LogLinearSDDP

import JuMP
import MathOptInterface
import SDDP
import Revise
import TimerOutputs
import GAMS
import Gurobi
import Printf
import Infiltrator

const MOI = MathOptInterface #TODO

const GURB_ENV = Ref{Gurobi.Env}()
#const ws = GAMS.GAMSWorkspace()

function __init__()
    GURB_ENV[] = Gurobi.Env()
    return
end

# Write your package code here.
include("typedefs.jl")
include("solver_handling.jl")
include("logging.jl")
include("cut_computations.jl")
include("ar_preparations.jl")
include("sampling_schemes.jl")
include("algorithm.jl")
include("duals.jl")
include("bellman.jl")

end
