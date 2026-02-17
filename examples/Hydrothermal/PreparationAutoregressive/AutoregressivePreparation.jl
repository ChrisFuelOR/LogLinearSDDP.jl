module PreparationAutoregressive

import CSV
import Distributions
import DataFrames
using GLM
import HypothesisTests
using Plots; pgfplotsx()#; pgfplotsx() gr()
import Statistics
import StatsBase
import StatsPlots
import Infiltrator
import Printf

struct MonthlyModelStorage
    detrending_mean::Float64
    detrending_sigma::Float64
    lag_order::Int64
    fitted_model::Any
    regressors::Matrix{Float64}
    year_offset::Int64

    function MonthlyModelStorage(
        detrending_mean,
        detrending_sigma,
        fitted_model,
        lag_order,
        regressors,
        year_offset
    )
        return new(
            detrending_mean,
            detrending_sigma,
            lag_order,
            fitted_model,
            regressors,
            year_offset,
        )
    end
end

include("data_plots.jl")
include("data_preparation.jl")
include("data_analysis.jl")
include("box_jenkins.jl")
include("periodic_box_jenkins.jl")
include("forecasting.jl")
include("ar_model_generation.jl")
include("validation_cross_models.jl")

end