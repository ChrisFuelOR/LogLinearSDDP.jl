# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Copyright (c) 2026 Christian Fuellner <christian.fuellner@kit.edu>
################################################################################

import Statistics
import DataFrames
import Infiltrator

""" Compute the sample autocovariance for a given time series (DataFrame) for all months and lags.
This is required to compute periodical ACFs, for instance. """
function compute_sample_autocovariance(df::DataFrames.DataFrame)

    # Get monthly means from dataframe (approximately 0 due to deseasonalization)
    monthly_means = [0.0 for i in 1:12] #Statistics.mean.(eachcol(df))

    # Get maximum number of lags to consider, number_of_years/4 according to Box & Jenkins
    n_years = DataFrames.nrow(df)
    max_lag = Int(floor(n_years/4)) # 12

    # Create a DataFrame to store the autocovariance (rows: months, columns: lags, lag 0 => variance)
    acv_df = DataFrames.DataFrame()
    for k in 0:max_lag    
        DataFrames.insertcols!(acv_df, Symbol(k) => [0.0 for i in 1:12])
    end

    # Iterate over all months
    for month in 1:12
        # Iterate over all lags
        for lag in 0:max_lag
            # Identify the correct month for the given lag
            lag_month = month - lag > 0 ? month - lag : Int(12 - mod((lag - month), 12))
            lag_year_offset = Int(ceil((lag - month + 1)/12))

            # Compute the sample autocovariance for the given month and lag
            acv = 1/(n_years-1) * sum((df[year,month]-monthly_means[month]) * (df[year-lag_year_offset,lag_month]-monthly_means[lag_month]) for year in 1+lag_year_offset:n_years)
            acv_df[month, Symbol(lag)] = acv
        end
    end

    return acv_df
end    
    