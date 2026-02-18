# # Model fitting and data preparation

# The folders `PreparationAutoregressive` and `PreparationAutoregressiveLinearized` contain files for the preparation of the AR processes and inflow realizations if either a log-linear model or a linearized model is used. In particular, they contain code to fit the models LOG-BIC, LOG-1 and LIN-FIT that we used in our experiments, whereas for LIN-SHA data from the literature is used. 

# For details on the methodology that is used, we refer to the electronic companion supplementing our paper. Here, we simply focus on explaining the structure of the code. We restrict to `PreparationAutoregressive`. For `PreparationAutoregressiveLinearized` the structure is very similar.

# !!! note "Remarks"
#     If you simply want to reproduce the SDDP results given the pre-fitted AR models that we used, then you do not have to deal with this section at all.

# The `PreparationAutoregressive` contains the following files:

#  * `run_file.jl`: Using this file, we can run the preprocessing of the loglinear AR models.
#  * `AutoregressivePreparation.jl`: Contains the module definition; also defines a `MonthlyModelStorage` struct in which information is stored during the fitting process
#  * `data_plots.jl`: Contains various functions for creating plots, e.g. plotting the original or detrended inflow data, plotting (partial) autocorrelation functions, creating boxplots for model validation
#  * `data_preparation.jl`: Contains functionality to read the historical inflow data (which is stored in files `hist1.csv`, `hist2.csv`, `hist3.csv` and `hist4.csv` in folder `historical_data`), to detrend it and split it into training and validation sets.
#  * `data_analysis.jl`: Contains a function to compute the sample autocovariance for a given time series (in dataframe format) for all months and lags. This is required to compute periodical AFCs, for instance.
#  * `box_jenkins.jl`: Contains functionality to perform the Box-Jenkins method to fit an AR model to the time series data. The steps are
#      * Analyzing the (P)AFC to detect autocorrelation
#      * Perform an augmented Dickey-Fuller test to test for stationarity
#      * Fit an AR model to the time series
#      * Use the fitted model and the residuals for validations (analyzing the goodness of fit, performing significance tests, analyzing autocorrelation and heteroscedasticity in the residuals, testing for normal distribution of the residuals).
#  * `periodic_box_jenkins.jl`: Similar to box_jenkins.jl but more advanced as it allows us to fit periodic models were the coefficients differ between months. This is used to fit the models that we use within SDDP.
#      * The lag order can either be identified by using periodic AFCs or by fitting several models with different lag order (from 1 to 12) and choosing the lag order that yields the best AIC or BIC measure.
#  * `forecasting.jl`: Contains functionality to create point forecasts as well as full scenarios (over the whole time horizon) using the fitted models. This is done for further model validation
#  * `ar_model_generation.jl`: Main file for preparing the AR model data, as it uses functions from all the previous files. The procedure works as follows:
#      * Iterate over all 4 reservoirs
#      * Read the relevant data
#      * Analyze the data using plots
#      * Convert it to logarithmic data
#      * Detrend the data and split it into training and validation data
#      * Perform a (periodic) Box-Jenkins analysis to fit an AR model to the data and validate it.
#      * Create point forecasts using the fitted model for further model validation.
#      * Create full scenarios (over the whole time horizon) using the fitted model for further model validation.
#      * Prepare the model output in the correct form so that it can be used as an input in our version of SDDP as well as out-of-sample simulations. Also checks are performed to assert that this is done correctly.
#  * `scenario_generation.jl`:
#      * Once `ar_model_generation.jl` is finished and the process coefficients are stored correctly, this file can be used to generate a predefined number of realizations for the stagewise independent term in the AR model. These realizations are then used in `hydrothermal_model.jl` to parameterize the uncertain data.
#      * Additionally, this file can be used to prepare the history of the stochastic process that is required in our version of SDDP.

# Both `ar_model_generation.jl` (executed using `run-file.jl`) and `scenario_generation.jl` were executed before running our experiments in SDDP. Once we started our experiments in SDDP, all the inputs were fixed.

# The folder `PreparationHistorical` contains files to prepare the historical data for the out-of-sample simulations.