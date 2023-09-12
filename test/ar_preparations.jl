module TestCutComputations

using LogLinearSDDP
using Test
using Infiltrator
using SDDP
using JuMP

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

function create_model()

    model = SDDP.LinearPolicyGraph(
        stages = 3,
        optimizer = GLPK.Optimizer,
        lower_bound = 0.0,
    ) do sp, t
        JuMP.@variable(sp, x[i=1:2] >= 0, SDDP.State, initial_value = 0)
        JuMP.@constraint(sp, ubound[i=1:2], x[i].out <= 6)

        if t == 1
            JuMP.@variable(sp, w)
            JuMP.@stageobjective(sp, w)
            JuMP.@constraint(sp, -w <= x[1].out - 4)
            JuMP.@constraint(sp, w >= x[1].out - 4)

        elseif t == 2
            


        else

        end


    end
   


    v∗ = min
    n
    |x1 − 4| + θ1 : x1 ∈ [0, 6], θ1 ≥ 0
    o
    ,
    Q2(x1, ξ2) = min
    n
    2y2 + x2 + θ2 : y2 − x2 = ξ2 − x1, x2, y2 ∈ [0, 6], θ2 ≥ 0
    o
    ,
    Q3(x2, ξ3) = min
    n
    x31 + x32 : x31 − x32 = ξ3 − x2, x31, x32 ∈ [0, 6]
    o
    with the RHS uncertainty described by
    ξ1 = 3, ξt = eηtξ
    1
    4
    t−1, ηt ∈ {−1, 1} , for t = 2, 3.


end