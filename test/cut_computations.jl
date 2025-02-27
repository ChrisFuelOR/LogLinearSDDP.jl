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

function test_compute_cut_exponents_1D()

    autoregressive_data, stages, realizations = create_autoregressive_data_1D()
    cut_exponents = LogLinearSDDP.compute_cut_exponents(LogLinearSDDP.ProblemParams(stages, realizations), autoregressive_data)

    @test length(cut_exponents) == 3
    @test size(cut_exponents[1], 1) == 3
    @test size(cut_exponents[1], 2) == 1
    @test size(cut_exponents[1], 3) == 1
    @test size(cut_exponents[1], 4) == 1
    @test cut_exponents[3][3,1,1,1] == 1/4
    @test cut_exponents[2][3,1,1,1] == 1/16
    @test cut_exponents[2][2,1,1,1] == 1/4

    return
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

function test_compute_cut_exponents_2D()

    autoregressive_data, stages, realizations = create_autoregressive_data_2D()
    cut_exponents = LogLinearSDDP.compute_cut_exponents(LogLinearSDDP.ProblemParams(stages, realizations), autoregressive_data)

    @test cut_exponents[3][3,1,1,2] == 0.0
    @test cut_exponents[3][3,1,2,2] == 0.0
    @test cut_exponents[3][3,1,1,1] == 0.2
    @test cut_exponents[3][3,1,2,1] == 0.0
    @test cut_exponents[3][3,2,1,2] == 0.0
    @test cut_exponents[3][3,2,2,2] == 1/3
    @test cut_exponents[3][3,2,1,1] == 0.0
    @test cut_exponents[3][3,2,2,1] == 2/3

    @test cut_exponents[2][2,1,1,2] == 0.0
    @test cut_exponents[2][2,1,2,2] == 0.0
    @test cut_exponents[2][2,1,1,1] == 0.2
    @test cut_exponents[2][2,1,2,1] == 0.0
    @test cut_exponents[2][2,2,1,2] == 0.0
    @test cut_exponents[2][2,2,2,2] == 1/3
    @test cut_exponents[2][2,2,1,1] == 0.0
    @test cut_exponents[2][2,2,2,1] == 2/3

    @test cut_exponents[2][3,1,1,2] == 0.0
    @test cut_exponents[2][3,1,2,2] == 0.0
    @test cut_exponents[2][3,1,1,1] ≈ 0.04
    @test cut_exponents[2][3,1,2,1] == 0.0
    @test cut_exponents[2][3,2,1,2] == 0.0
    @test cut_exponents[2][3,2,2,2] == 2/9
    @test cut_exponents[2][3,2,1,1] == 0.0

    @test cut_exponents[2][3,2,2,1] ≈ 7/9

    return
end

function test_compute_scenario_factors_1D()

    ar_process, stages, realizations = create_autoregressive_data_1D()
    problem_params = LogLinearSDDP.ProblemParams(stages, realizations)
    cut_exponents = LogLinearSDDP.compute_cut_exponents(problem_params, ar_process)

    # Case 1
    t = 3
    process_state = Dict{Int64, Any}()
    process_state[0] = [1.0]
    process_state[1] = [3.0]
    process_state[2] = [exp(-1)*3^0.25]
    cut_exponents_stage = cut_exponents[t]
    scenario_factors = LogLinearSDDP.compute_scenario_factors(t, process_state, problem_params, cut_exponents_stage, ar_process)
    @test scenario_factors[3,1] ≈ (exp(-1)*3^0.25)^0.25

    # Case 2
    t = 3
    process_state = Dict{Int64, Any}()
    process_state[0] = [1.0]
    process_state[1] = [3.0]
    process_state[2] = [exp(1)*3^0.25]
    cut_exponents_stage = cut_exponents[t]
    scenario_factors = LogLinearSDDP.compute_scenario_factors(t, process_state, problem_params, cut_exponents_stage, ar_process)
    @test scenario_factors[3,1] ≈ (exp(1)*3^0.25)^0.25

    # Case 3
    t = 2
    process_state = Dict{Int64, Any}()
    process_state[0] = [1.0]
    process_state[1] = [3.0]
    cut_exponents_stage = cut_exponents[t]
    scenario_factors = LogLinearSDDP.compute_scenario_factors(t, process_state, problem_params, cut_exponents_stage, ar_process)
    @test scenario_factors[2,1] ≈ 3.0^(1/4)
    @test scenario_factors[3,1] ≈ 3.0^(1/16)

    return
end

function test_compute_intercept_value_1D()

    ar_process, stages, realizations = create_autoregressive_data_1D()
    problem_params = LogLinearSDDP.ProblemParams(stages, realizations)
    T = problem_params.number_of_stages
    L = ar_process.dimension

    # Stage 3 - Realization 1
    t = 3
    cut = LogLinearSDDP.Cut(Dict(:x => 1.0), Inf, [-0.5*(exp(-1)+exp(1));;], Dict(:x => 3.5158434275747794), nothing, nothing, nothing, nothing, 1, 1)
    scenario_factors = Array{Float64,2}(undef, T, L)

    scenario_factors[3,1] = (exp(-1)*3^0.25)^0.25
    intercept_value = LogLinearSDDP.compute_intercept_value(t, cut, scenario_factors, problem_params, ar_process)
    @test intercept_value ≈ -0.5*(exp(-1)+exp(1)) * (exp(-1)*3^0.25)^0.25

    # Stage 3 - Realization 2
    scenario_factors = Array{Float64,2}(undef, T, L)
    scenario_factors[3,1] = (exp(1)*3^0.25)^0.25
    intercept_value = LogLinearSDDP.compute_intercept_value(t, cut, scenario_factors, problem_params, ar_process)
    @test intercept_value ≈ -0.5*(exp(-1)+exp(1)) * (exp(1)*3^0.25)^0.25

    # Stage 2 - Realization 1
    t = 2
    scenario_factors = Array{Float64,2}(undef, T, L)
    cut = LogLinearSDDP.Cut(Dict(:x => 0.5), Inf, [-0.5*exp(-1);1/4*(exp(-3/4)+exp(5/4));;], Dict(:x => 4.0), nothing, nothing, nothing, nothing, 1, 1)
   
    scenario_factors[2,1] = 3.0^(1/4)
    scenario_factors[3,1] = 3.0^(1/16)
    intercept_value = LogLinearSDDP.compute_intercept_value(t, cut, scenario_factors, problem_params, ar_process)
    @test intercept_value ≈ -0.5*exp(-1) * 3.0^(1/4) + 1/4*(exp(-3/4)+exp(5/4)) * 3.0^(1/16)
    # intercept_check_value = 0.8190119645169287

    return
end

function test_compute_scenario_factors_2D()

    ar_process, stages, realizations = create_autoregressive_data_2D()
    problem_params = LogLinearSDDP.ProblemParams(stages, realizations)
    cut_exponents = LogLinearSDDP.compute_cut_exponents(problem_params, ar_process)

    # Case 1
    t = 3
    process_state = Dict{Int64, Any}()
    process_state[0] = [4.0, 5.0]
    process_state[1] = [4.0, 5.0]
    process_state[2] = [exp(3/4)*4.0^(1/5), exp(-1/2)*5.0^(2/3)*5.0^(1/3)]
    cut_exponents_stage = cut_exponents[t]
    scenario_factors = LogLinearSDDP.compute_scenario_factors(t, process_state, problem_params, cut_exponents_stage, ar_process)
    @test scenario_factors[3,1] ≈ (exp(3/4)*4.0^(1/5))^(1/5)
    @test scenario_factors[3,2] ≈ (exp(-1/2)*5.0^(2/3)*5.0^(1/3))^(2/3) * 5.0^(1/3)

    # Case 2
    t = 3
    process_state = Dict{Int64, Any}()
    process_state[0] = [4.0, 5.0]
    process_state[1] = [4.0, 5.0]
    process_state[2] = [exp(3/4)*4.0^(1/5), exp(1/2)*5.0^(2/3)*5.0^(1/3)]
    cut_exponents_stage = cut_exponents[t]
    scenario_factors = LogLinearSDDP.compute_scenario_factors(t, process_state, problem_params, cut_exponents_stage, ar_process)
    @test scenario_factors[3,1] ≈ (exp(3/4)*4.0^(1/5))^(1/5)
    @test scenario_factors[3,2] ≈ (exp(1/2)*5.0^(2/3)*5.0^(1/3))^(2/3) * 5.0^(1/3)

    # Case 3
    t = 2
    process_state = Dict{Int64, Any}()
    process_state[0] = [4.0, 5.0]
    process_state[1] = [4.0, 5.0]
    cut_exponents_stage = cut_exponents[t]
    scenario_factors = LogLinearSDDP.compute_scenario_factors(t, process_state, problem_params, cut_exponents_stage, ar_process)
    @test scenario_factors[2,1] ≈ 4.0^(1/5)
    @test scenario_factors[2,2] ≈ 5.0^(2/3) * 5.0^(1/3)
    @test scenario_factors[3,1] ≈ 4.0^(0.04)
    @test scenario_factors[3,2] ≈ 5.0^(7/9) * 5.0^(2/9)

    return
end

function test_compute_intercept_value_2D()

    ar_process, stages, realizations = create_autoregressive_data_2D()
    problem_params = LogLinearSDDP.ProblemParams(stages, realizations)
    T = problem_params.number_of_stages
    L = ar_process.dimension

    ####################################################
    # Stage 3
    t = 3
    cut = LogLinearSDDP.Cut(Dict(:x => -2/9), Inf, [1/9*(-3*exp(-4)+exp(2)) 1/9*(3*exp(0.5)-exp(-0.5))], Dict(:x => 8.760744970463605), nothing, nothing, nothing, nothing, 1, 1)
    scenario_factors = Array{Float64,2}(undef, T, L)
    ξ0D = 5.0
    ξ1I = 4.0
    ξ1D = 5.0

    # Realization 1
    η2I = 3/4
    η2D = 1/2
    ξ2I = exp(η2I) * ξ1I^(1/5)
    ξ2D = exp(η2D) * ξ1D^(2/3) * ξ0D^(1/3)
    scenario_factors[3,1] = (ξ2I)^(1/5)
    scenario_factors[3,2] = (ξ2D)^(2/3) * (ξ1D)^(1/3)
    intercept_value = LogLinearSDDP.compute_intercept_value(t, cut, scenario_factors, problem_params, ar_process)
    @test intercept_value ≈ 1/9*(-3*exp(-4)+exp(2)) * (ξ2I)^(1/5) + 1/9*(3*exp(0.5)-exp(-0.5)) * (ξ2D)^(2/3) * (ξ1D)^(1/3)

    # Realization 2
    η2I = -4
    η2D = 0
    ξ2I = exp(η2I) * ξ1I^(1/5)
    ξ2D = exp(η2D) * ξ1D^(2/3) * ξ0D^(1/3)
    scenario_factors[3,1] = (ξ2I)^(1/5)
    scenario_factors[3,2] = (ξ2D)^(2/3) * (ξ1D)^(1/3)
    intercept_value = LogLinearSDDP.compute_intercept_value(t, cut, scenario_factors, problem_params, ar_process)
    @test intercept_value ≈ 1/9*(-3*exp(-4)+exp(2)) * (ξ2I)^(1/5) + 1/9*(3*exp(0.5)-exp(-0.5)) * (ξ2D)^(2/3) * (ξ1D)^(1/3)
    
    ####################################################
    # Stage 2
    t = 2

    a = 1/9*(-2/9*exp(-4) - 2*exp(-4) - 2*exp(-4) - 2/9*exp(3/4) - 2/9*exp(3/4) - 2*exp(3/4) + 1*exp(2) - 2/9*exp(2) - 2/9*exp(2))
    b = 1/9*(2/9*exp(-1/2) + 2*exp(0) + 2*exp(1/2) + 2/9*exp(-1/2) + 2/9*exp(0) + 2*exp(1/2) - 1*exp(-1/2) + 2/9*exp(0) + 2/9*exp(1/2))
    c = 1/9*1/9*(exp(2)-3*exp(-4)) * (3*exp(-4*1/5) + 3*exp(3/4*1/5) + 2*exp(2*1/5))
    d = 1/9*1/9*(-exp(-1/2)+3*exp(1/2)) * (2*exp(-1/2*2/3) + 3*exp(0*2/3) + 3*exp(1/2*2/3))

    cut = LogLinearSDDP.Cut(Dict(:x => -0.6790123456790124), Inf, [a b; c d], Dict(:x => 9.0), nothing, nothing, nothing, nothing, 1, 1)
    scenario_factors = Array{Float64,2}(undef, T, L)
    ξ0D = 5.0
    ξ1I = 4.0
    ξ1D = 5.0

    scenario_factors[2,1] = ξ1I^(1/5)
    scenario_factors[2,2] = ξ1D^(2/3) * ξ0D^(1/3)
    scenario_factors[3,1] = ξ1I^(0.04)
    scenario_factors[3,2] = ξ1D^(7/9) * ξ0D^(2/9)
    intercept_value = LogLinearSDDP.compute_intercept_value(t, cut, scenario_factors, problem_params, ar_process)
    @test intercept_value ≈ a * ξ1I^(1/5) + b * ξ1D^(2/3) * ξ0D^(1/3) + c * ξ1I^(0.04) + d * ξ1D^(7/9) * ξ0D^(2/9)

    return
end


end

TestCutComputations.runtests()