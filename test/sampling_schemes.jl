module TestSamplingSchemes

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
    eta = [0.5]
    ar_parameters[2] = LogLinearSDDP.AutoregressiveProcessStage(intercept, coefficients, eta)

    intercept = zeros(dim)
    coefficients = 1/4 * ones(dim, dim, lag_order)
    eta = [1.0]
    ar_parameters[3] = LogLinearSDDP.AutoregressiveProcessStage(intercept, coefficients, eta)

    ar_process = LogLinearSDDP.AutoregressiveProcess(dim, lag_order, ar_parameters, ar_history, false)

    return ar_process, stages, realizations
end    


function create_autoregressive_data_2D()
    stages = 3
    realizations = 1
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
    eta_1 = [-1.0]
    eta_2 = [1.0]
    eta = vec(collect(Iterators.product(eta_1, eta_2)))

    ar_parameters[2] = LogLinearSDDP.AutoregressiveProcessStage(intercept, coefficients, eta)
    ar_parameters[3] = LogLinearSDDP.AutoregressiveProcessStage(intercept, coefficients, eta)

    ar_process = LogLinearSDDP.AutoregressiveProcess(dim, lag_order, ar_parameters, ar_history, false)

    return ar_process, stages, realizations
end   

function test_set_noise_terms_1D()

    ar_process, stages, realizations = create_autoregressive_data_1D()

    # Stage 2
    t = 2
    ps_array = Array{Float64,2}(undef, ar_process.dimension, ar_process.lag_order)
    process_state = Dict{Int64, Any}()
    process_state[1] = ar_process.history[1]
    LogLinearSDDP.process_state_to_array!(ps_array, process_state, t)

    noise_term = Vector{Float64}(undef, length(ar_process.parameters[t].eta))
    LogLinearSDDP.set_noise_terms!(noise_term, t, ar_process.dimension, ar_process.lag_order, ar_process.parameters[t], Float64.(ar_process.parameters[t].eta), ps_array)

    @test length(noise_term) == 1
    @test noise_term[1] == exp(0.5)*3^(0.25)

    # Stage 3
    t = 3
    ps_array = Array{Float64,2}(undef, ar_process.dimension, ar_process.lag_order)
    process_state[2] = noise_term[1]
    LogLinearSDDP.process_state_to_array!(ps_array, process_state, t)

    noise_term = Vector{Float64}(undef, length(ar_process.parameters[t].eta))
    LogLinearSDDP.set_noise_terms!(noise_term, t, ar_process.dimension, ar_process.lag_order, ar_process.parameters[t], Float64.(ar_process.parameters[t].eta), ps_array)

    @test length(noise_term) == 1
    @test isapprox(noise_term[1], exp(9/8)*3^(1/16))

    return
end

function test_set_noise_terms_2D()

    ar_process, stages, realizations = create_autoregressive_data_2D()

    # Stage 2
    t = 2
    ps_array = Array{Float64,2}(undef, ar_process.dimension, ar_process.lag_order)
    process_state = Dict{Int64, Any}()
    process_state[1] = ar_process.history[1]
    process_state[0] = ar_process.history[0]
    LogLinearSDDP.process_state_to_array!(ps_array, process_state, t)

    realization = ar_process.parameters[t].eta[1]
    noise_term = Vector{Float64}(undef, ar_process.dimension)
    LogLinearSDDP.set_noise_terms!(noise_term, t, ar_process.dimension, ar_process.lag_order, ar_process.parameters[t], collect(realization), ps_array)

    @test length(noise_term) == 2
    @test noise_term[1] == exp(-1)*4^(0.2)
    @test noise_term[2] == exp(1)*5

    return
end


end

TestSamplingSchemes.runtests()