module TestCutComputations

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

function _create_autoregressive_data_simple()
    stages = 3
    lag_order = 1
    realizations = 2
    data = Vector{LogLinearSDDP.AutoregressiveDataStage}(undef, 3)

    dim_1 = 1
    intercept_1 = zeros(dim_1)
    coefficients_1 = ones(lag_order, dim_1, dim_1)
    eta_1 = 3 * ones(dim_1, 1)
    data[1] = LogLinearSDDP.AutoregressiveDataStage(intercept_1, coefficients_1, eta_1, dim_1)

    dim_2 = 1
    intercept_2 = zeros(dim_1)
    coefficients_2 = 1/4 * ones(lag_order, dim_1, dim_1)
    eta_2 = [-1 1]
    data[2] = LogLinearSDDP.AutoregressiveDataStage(intercept_2, coefficients_2, eta_2, dim_2)

    dim_3 = 1
    intercept_3 = zeros(dim_1)
    coefficients_3 = 1/4 * ones(lag_order, dim_1, dim_1)
    eta_3 = [-1 1]
    data[3] = LogLinearSDDP.AutoregressiveDataStage(intercept_3, coefficients_3, eta_3, dim_3)

    autoregressive_data = LogLinearSDDP.AutoregressiveData(lag_order, data)

    return autoregressive_data, stages, realizations
end    

function _create_autoregressive_data_medium()
    stages = 3
    lag_order = 2
    realizations = 3
    data = Vector{LogLinearSDDP.AutoregressiveDataStage}(undef, 3)

    dim_1 = 2
    intercept_1 = zeros(dim_1)
    coefficients_1 = ones(lag_order, dim_1, dim_1)
    eta_1 = 3 * ones(dim_1, 1)
    data[1] = LogLinearSDDP.AutoregressiveDataStage(intercept_1, coefficients_1, eta_1, dim_1)

    dim_2 = 2
    intercept_2 = zeros(dim_1)
    coefficients_2 = 1/4 * ones(lag_order, dim_1, dim_1)
    eta_2 = [-1 1]
    data[2] = LogLinearSDDP.AutoregressiveDataStage(intercept_2, coefficients_2, eta_2, dim_2)

    dim_3 = 2
    intercept_3 = zeros(dim_1)
    coefficients_3 = 1/4 * ones(lag_order, dim_1, dim_1)
    eta_3 = [-1 1]
    data[3] = LogLinearSDDP.AutoregressiveDataStage(intercept_3, coefficients_3, eta_3, dim_3)

    autoregressive_data = LogLinearSDDP.AutoregressiveData(lag_order, data)

    return autoregressive_data, stages, realizations
end    

function test_compute_cut_exponents_simple()

    autoregressive_data, stages, realizations = _create_autoregressive_data_simple()
    cut_exponents = LogLinearSDDP.compute_cut_exponents(LogLinearSDDP.ProblemParams(stages, realizations), autoregressive_data)

    # Infiltrator.@infiltrate
    @test length(cut_exponents) == 3
    @test size(cut_exponents[1], 1) == 3
    @test size(cut_exponents[1], 2) == 1
    @test size(cut_exponents[1], 3) == 1
    @test size(cut_exponents[1], 4) == 1
   # @test cut_exponents[3][1,1,1,1] < 1e-300
   # @test cut_exponents[1][2,1,1,1] < 1e-300
    @test cut_exponents[3][3,1,1,1] == 1/4
    @test cut_exponents[2][3,1,1,1] == 1/16
    @test cut_exponents[2][2,1,1,1] == 1/4

    return
end

function test_compute_cut_exponents_medium()

    autoregressive_data, stages, realizations = _create_autoregressive_data_medium()
    cut_exponents = LogLinearSDDP.compute_cut_exponents(LogLinearSDDP.ProblemParams(stages, realizations), autoregressive_data)



end




end
