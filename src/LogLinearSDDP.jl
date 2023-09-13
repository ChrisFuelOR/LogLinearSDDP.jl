module LogLinearSDDP

import SDDP
import JuMP
import MathOptInterface
import Revise
import TimerOutputs
import GAMS
import Gurobi
import Printf
import Infiltrator
import Dates
import Distributed

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
include("bellman_redefine.jl")
include("bellman.jl")

end
