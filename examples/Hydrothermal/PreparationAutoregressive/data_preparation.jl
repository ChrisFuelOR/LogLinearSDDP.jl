# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Copyright (c) 2026 Christian Fuellner <christian.fuellner@kit.edu>
################################################################################

""" Reads time series data from csv files."""
function read_raw_data(file_name::String)
    raw_data = CSV.read(file_name, DataFrames.DataFrame, header=false, delim=";")
    DataFrames.rename!(raw_data, ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])    

    return raw_data
end

""" Transforms dataframe with monthly columns to a vector of full time series."""
function data_frame_to_vector(data_frame::DataFrames.DataFrame)
    full_vector = Float64[]
    for row in eachrow(data_frame)
        append!(full_vector, Vector(row))
    end

    return full_vector
end

""" Deseasonalizes/Detrends the data in DataFrame df. If with_sigma is true, then mean and standard deviation
are used for the detrending, otherwise only the mean is used."""
function detrend_data(df::DataFrames.DataFrame, with_sigma::Bool, with_plot::Bool)

    μ = Statistics.mean.(eachcol(df))
    if with_sigma
        σ = Statistics.std.(eachcol(df))
        residuals = copy(df)
        for col_name in names(df)
            residuals[!, col_name] = (df[!, col_name] .- μ[parse(Int, col_name)]) ./ σ[parse(Int, col_name)]
        end
    else
        σ = ones(length(μ))
        residuals = copy(df)
        for col_name in names(df)
            residuals[!, col_name] = (df[!, col_name] .- μ[parse(Int, col_name)])
        end
    end

    # Plot detrended and non-detrended data if required
    if with_plot
        detrending_plots(df, residuals)
    end

    return μ, σ, residuals
end

""" Split the data in a training set and a test set according to split_percentage."""
function split_data(df::DataFrames.DataFrame, split_bool::Bool, split_percentage::Float64)

    if split_bool
        split_bound = Int(floor(split_percentage * DataFrames.nrow(df)))
        training_df = df[1:split_bound, :]
        test_df = df[split_bound+1:DataFrames.nrow(df), :]
        return training_df, test_df
    else
        return df, df
    end
end