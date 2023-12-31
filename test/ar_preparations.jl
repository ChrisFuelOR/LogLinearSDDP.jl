module TestArPreparations

using LogLinearSDDP
using Test
using Infiltrator
using JuMP
using SDDP
using Gurobi

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

function create_autoregressive_data_1D()
    stages = 3
    realizations = 2
    lag_order = 1
    dim = 1

    ar_history = Dict{Int64,Any}()
    ar_history[1] = [3.0] 

    ar_parameters  = Dict{Int64, LogLinearSDDP.AutoregressiveProcessStage}()

    intercept = zeros(dim)
    coefficients = 1/4 * ones(dim, dim, lag_order)
    eta = [-1.0, 1.0]
    ar_parameters[2] = LogLinearSDDP.AutoregressiveProcessStage(dim, intercept, coefficients, eta)

    intercept = zeros(dim)
    coefficients = 1/4 * ones(dim, dim, lag_order)
    eta = [-1.0, 1.0]
    ar_parameters[3] = LogLinearSDDP.AutoregressiveProcessStage(dim, intercept, coefficients, eta)

    ar_process = LogLinearSDDP.AutoregressiveProcess(lag_order, ar_parameters, ar_history)

    return ar_process, stages, realizations
end    


function create_autoregressive_data_2D()
    stages = 3
    realizations = 3
    lag_order = 2
    dim = 2

    ar_history = Dict{Int64,Any}()
    ar_history[0] = [4.0, 5.0]
    ar_history[1] = [4.0, 5.0] 
    
    ar_parameters  = Dict{Int64, LogLinearSDDP.AutoregressiveProcessStage}()

    intercept = [0.0, 0.0]
    coefficients = zeros(dim, dim, lag_order)
    coefficients[1,1,1] = 1/5
    coefficients[2,2,1] = 2/3
    coefficients[2,2,2] = 1/3   
    eta_1 = [-4.0, 3/4, 2.0]
    eta_2 = [-0.5, 0.0, 0.5]
    eta = vec(collect(Iterators.product(eta_1, eta_2)))

    ar_parameters[2] = LogLinearSDDP.AutoregressiveProcessStage(dim, intercept, coefficients, eta)
    ar_parameters[3] = LogLinearSDDP.AutoregressiveProcessStage(dim, intercept, coefficients, eta)

    ar_process = LogLinearSDDP.AutoregressiveProcess(lag_order, ar_parameters, ar_history)

    return ar_process, stages, realizations
end   

function create_model_1D(ar_process::LogLinearSDDP.AutoregressiveProcess)

    model = SDDP.LinearPolicyGraph(
        stages = 3,
        optimizer = Gurobi.Optimizer,
        sense = :Min,
        lower_bound = 0.0,
    ) do sp, t

        JuMP.@variable(sp, x >= 0, SDDP.State, initial_value = 0)
        JuMP.@constraint(sp, ubound, x.out <= 6)
        JuMP.@variable(sp, ξ)

        if t == 1
            JuMP.@variable(sp, w)
            SDDP.@stageobjective(sp, w)
            JuMP.@constraint(sp, -w <= x.out - 4)
            JuMP.@constraint(sp, w >= x.out - 4)

            # Parameterize the RHS
            SDDP.parameterize(sp, [[0.0]]) do ω
                JuMP.fix(ξ, ω[1])
            end

        elseif t == 2
            JuMP.@variable(sp, y >= 0)
            JuMP.@constraint(sp, y <= 6)
            SDDP.@stageobjective(sp, 2*y + x.out)
            coupling_ref = JuMP.@constraint(sp, y - x.out == ξ - x.in)

            # Store coupling constraint reference to access dual multipliers LATER
            coupling_refs = sp.ext[:coupling_constraints] = Vector{JuMP.ConstraintRef}(undef, ar_process.parameters[t].dimension)
            coupling_refs[1] = coupling_ref
            
            realizations = ar_process.parameters[t].eta
            # Parameterize the RHS
            SDDP.parameterize(sp, realizations) do ω
                JuMP.fix(ξ, ω[1])
            end
        else
            JuMP.@variable(sp, y >= 0)
            JuMP.@constraint(sp, y <= 6)
            SDDP.@stageobjective(sp, y + x.out)
            coupling_ref = JuMP.@constraint(sp, y - x.out == ξ - x.in)

            # Store coupling constraint reference to access dual multipliers LATER
            coupling_refs = sp.ext[:coupling_constraints] = Vector{JuMP.ConstraintRef}(undef, ar_process.parameters[t].dimension)
            coupling_refs[1] = coupling_ref
            
            realizations = ar_process.parameters[t].eta
            # Parameterize the RHS
            SDDP.parameterize(sp, realizations) do ω
                JuMP.fix(ξ, ω[1])
            end
        end        
    end

    return model
end

function create_model_2D(ar_process::LogLinearSDDP.AutoregressiveProcess)

    model = SDDP.LinearPolicyGraph(
        stages = 3,
        optimizer = Gurobi.Optimizer,
        sense = :Min,
        lower_bound = 0.0,
    ) do sp, t

        JuMP.@variable(sp, 5 <= x <= 15, SDDP.State, initial_value = 10)
        JuMP.@variable(sp, g_t >= 0)
        JuMP.@variable(sp, g_h >= 0)
        JuMP.@variable(sp, s >= 0)
        JuMP.@variable(sp, demand)
        JuMP.@variable(sp, inflow)
        coupling_ref_1 = JuMP.@constraint(sp, water_bal, x.out - x.in + g_h + s == inflow)
        JuMP.@constraint(sp, demand_bal, g_h + g_t == demand)
        SDDP.@stageobjective(sp, s + t * g_t)

        # Parameterize inflow and demand
        if t == 1
            realizations = [[0.0, 0.0]]
        else
            realizations = ar_process.parameters[t].eta

            # Store coupling constraint reference to access dual multipliers LATER
            coupling_refs = sp.ext[:coupling_constraints] = Vector{JuMP.ConstraintRef}(undef, ar_process.parameters[t].dimension)
            coupling_refs[1] = coupling_ref_1
        end
        
        SDDP.parameterize(sp, realizations) do ω
            JuMP.fix(inflow, ω[1])
            JuMP.fix(demand, ω[2])
        end
    end

    return model

end

function test_get_max_dimension()

    # 1D example
    ar_process, stages, realizations = create_autoregressive_data_1D()
    L = LogLinearSDDP.get_max_dimension(ar_process)
    @test L == 1

    # 2D example
    ar_process, stages, realizations = create_autoregressive_data_2D()
    L = LogLinearSDDP.get_max_dimension(ar_process)
    @test L == 2

    return
end

function test_get_lag_dimensions()

    # 1D example
    ar_process, stages, realizations = create_autoregressive_data_1D()
    @test LogLinearSDDP.get_lag_dimensions(ar_process, 2) == [1]
    @test LogLinearSDDP.get_lag_dimensions(ar_process, 3) == [1]

    # 2D example
    ar_process, stages, realizations = create_autoregressive_data_2D()
    L = LogLinearSDDP.get_max_dimension(ar_process)
    @test LogLinearSDDP.get_lag_dimensions(ar_process, 2) == [2, 2]
    @test LogLinearSDDP.get_lag_dimensions(ar_process, 3) == [2, 2]

    return
end


function test_initialize_process_state()

    # 1D example
    ar_process, stages, realizations = create_autoregressive_data_1D()
    model = create_model_1D(ar_process)
    LogLinearSDDP.initialize_process_state(model, ar_process)

    @test length(model.ext[:initial_process_state]) == 1
    @test model.ext[:initial_process_state][0] == [1.0]
    @test length(model.nodes[1].ext[:process_state]) == 1
    @test model.nodes[1].ext[:process_state][0] == [1.0] 

    # 2D example
    ar_process, stages, realizations = create_autoregressive_data_2D()
    model = create_model_2D(ar_process)
    LogLinearSDDP.initialize_process_state(model, ar_process)    

    @test length(model.ext[:initial_process_state]) == 2
    @test model.ext[:initial_process_state][-1] == [1.0, 1.0]
    @test model.ext[:initial_process_state][0] == [4.0, 5.0]
    @test length(model.nodes[1].ext[:process_state]) == 2
    @test model.nodes[1].ext[:process_state][-1] == [1.0, 1.0] 
    @test model.nodes[1].ext[:process_state][0] == [4.0, 5.0] 

    return
end


end

TestArPreparations.runtests()