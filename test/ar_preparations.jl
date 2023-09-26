module TestArPreparations

using LogLinearSDDP
using Test
using Infiltrator

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

function _create_autoregressive_data_1D()
    stages = 3
    realizations = 2
    lag_order = 1
    dim = 1

    ar_history = Dict{Int64,Any}()
    ar_history[1] = [3.0] 

    ar_parameters  = Dict{Int64, LogLinearSDDP.AutoregressiveProcessStage}()

    intercept = zeros(dim)
    coefficients = 1/4 * ones(lag_order, dim, dim)
    eta = [-1.0, 1.0]
    ar_parameters[2] = LogLinearSDDP.AutoregressiveProcessStage(dim, intercept, coefficients, eta)

    intercept = zeros(dim)
    coefficients = 1/4 * ones(lag_order, dim, dim)
    eta = [-1.0, 1.0]
    ar_parameters[3] = LogLinearSDDP.AutoregressiveProcessStage(dim, intercept, coefficients, eta)

    ar_process = LogLinearSDDP.AutoregressiveProcess(lag_order, ar_parameters, ar_history)

    return ar_process, stages, realizations
end    


function _create_autoregressive_data_2D()
    stages = 3
    realizations = 3
    lag_order = 2
    dim = 2

    ar_history = Dict{Int64,Any}()
    ar_history[0] = [4.0, 5.0]
    ar_history[1] = [4.0, 5.0] 
    
    ar_parameters  = Dict{Int64, LogLinearSDDP.AutoregressiveProcessStage}()

    intercept = [0.0, 0.0]
    coefficients = zeros(lag_order, dim, dim)
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

function test_get_max_dimension()

    # 1D example
    ar_process, stages, realizations = _create_autoregressive_data_1D()
    L = LogLinearSDDP.get_max_dimension(ar_process)
    @test L == 1

    # 2D example
    ar_process, stages, realizations = _create_autoregressive_data_2D()
    L = LogLinearSDDP.get_max_dimension(ar_process)
    @test L == 2

    return
end

function test_get_lag_dimensions()

    # 1D example
    ar_process, stages, realizations = _create_autoregressive_data_1D()
    @test LogLinearSDDP.get_lag_dimensions(ar_process, 2) == [1]
    @test LogLinearSDDP.get_lag_dimensions(ar_process, 3) == [1]

    # 2D example
    ar_process, stages, realizations = _create_autoregressive_data_2D()
    L = LogLinearSDDP.get_max_dimension(ar_process)
    @test LogLinearSDDP.get_lag_dimensions(ar_process, 2) == [2, 2]
    @test LogLinearSDDP.get_lag_dimensions(ar_process, 3) == [2, 2]

    return
end


end

TestArPreparations.runtests()