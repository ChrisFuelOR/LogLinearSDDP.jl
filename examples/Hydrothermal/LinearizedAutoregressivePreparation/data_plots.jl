""" Plotting detrended and non-detrended data for comparison."""
function detrending_plots(df::DataFrames.DataFrame, residuals::DataFrames.DataFrame)

    # Plot of the original time series which illustrates the seasonality
    ts_plot = Plots.plot(data_frame_to_vector(df), legend=false, color=:red, lw=2)
    Plots.display(ts_plot)
    
    # Compare distribution of monthly data before and after deseasonalization (box-plots)
    bx_plot_1 = StatsPlots.@df df StatsPlots.boxplot(cols(), legend=false, xticks=1:12)
    bx_plot_2 = StatsPlots.@df residuals StatsPlots.boxplot(cols(), legend=false, xticks=1:12)
    Plots.display(bx_plot_1)
    Plots.display(bx_plot_2)

    return
end

""" Plotting the (P)ACF for two different cases in comparsion. n is the number of years
and used to compute the 95% confidence interval.
These plots may give a hint on the required lag, even though we use a monthly model.
A quick decline in PACF for the detrended data indicates stationarity."""
function pacf_comparison_plot(pacf_1::Vector{Float64}, pacf_2::Vector{Float64}, n::Int64)

    conf_value = 1.96 * 1/sqrt(n)
    p1 = Plots.plot(Plots.bar(pacf_1))
    Plots.hline!([-conf_value, conf_value], color=:black, lw=2, ls=:dash)
    p2 = Plots.plot(Plots.bar(pacf_2))
    Plots.hline!([-conf_value, conf_value], color=:black, lw=2, ls=:dash)
    pfull = Plots.plot(p1, p2, ylims=(-1,1), legend=false)
    Plots.display(pfull)

    return
end

""" Plotting the (P)ACF for a given vector (e.g. residuals)."""
function pacf_plot(pacf::Vector{Float64}, standard_errors::Vector{Float64})

    plot_pacf = Plots.plot(Plots.bar(pacf), ylims=(-1,1), legend=false)
    Plots.plot!(-1.96*standard_errors, color=:black, lw=2, ls=:dash)
    Plots.plot!(1.96*standard_errors, color=:black, lw=2, ls=:dash) 
    #Plots.hline!([-conf_value, conf_value], color=:black, lw=2, ls=:dash)
    Plots.display(plot_pacf)
    return
end

""" Plotting the time series as a simple graph."""
function time_series_plot(series::Vector{Float64})
    plot = Plots.plot(series, lw=2, legend=false)
    Plots.display(plot)
end

""" Plotting a vector (e.g. residuals) as a scatter plot."""
function scatter_plot(series::Vector{Float64})
    plot_scat = Plots.scatter(series, legend=false)
    Plots.display(plot_scat)
end

""" Plotting histograms of the data in a DataFrame."""
function histogram_plot(df::DataFrames.DataFrame)
    plot_hist = Plots.histogram(data_frame_to_vector(df), color=:gray, legend=false, nbins=30)
    Plots.display(plot_hist)
end

""" Plotting the forecasts for detrended, logarithmized and original data (with comparison to the actual data)."""
function plot_forecasts(df::DataFrames.DataFrame)      
    plot_fc_3 = Plots.plot(df[:,:orig],legend=false, color=:blue)
    Plots.plot!(df[:,:fc_orig],color=:red, lw=2)
    Plots.display(plot_fc_3)

    return
end

""" Making box-plot and scatter diagrams for the mean and std of the original historical data
vs the artificial scenario data on a monthly level."""
function plot_scenario_statistics(
    all_means::DataFrames.DataFrame,
    all_stds::DataFrames.DataFrame,
    df::DataFrames.DataFrame
    )

    bx_plot_mean = StatsPlots.@df all_means StatsPlots.boxplot(cols(), legend=false, xticks=1:12)
    for col_name in names(df)
        Plots.scatter!([parse(Int, col_name)], [Statistics.mean(df[:,col_name])], color = "black", label = "", markersize = 5, markershape = :x)
    end
    Plots.display(bx_plot_mean)

    bx_plot_std = StatsPlots.@df all_stds StatsPlots.boxplot(cols(), legend=false, xticks=1:12)
    for col_name in names(df)
        Plots.scatter!([parse(Int, col_name)], [Statistics.std(df[:,col_name])], color = "black", label = "", markersize = 5, markershape = :x)
    end
    Plots.display(bx_plot_std)

    return
end

""" Making a q-q-plot for comparison of the yearly means in the original historical time series
and the artificially generated scenario data."""
function plot_yearly_qq(
    yearly_means::Vector{Float64},
    historic_df::DataFrames.DataFrame,
    )

    # Get yearly_means from historic df
    historic_means = DataFrames.reduce(+, DataFrames.eachcol(historic_df)) ./ DataFrames.ncol(historic_df)

    # Make a quantile-quantile plot
    plot_qq = Plots.plot(StatsPlots.qqplot(historic_means, yearly_means, qqline = :identity))
    Plots.display(plot_qq)

    return
end