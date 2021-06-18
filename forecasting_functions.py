# Other
import warnings
import itertools

# Data manipulation
import pandas as pd
import numpy as np

# Statistical computing
import statsmodels.api as sm

# Data Plotting and Visualisation
import matplotlib.pyplot as plt

# Plot Styling
plt.style.use('fivethirtyeight')

# ARIMA parameters range: ARIMA(p,d,q)(P,D,Q)s
p_range = [0, 1, 2, 3, 4, 7]
d_range = [0, 1, 2]
q_range = [0, 1, 2]

# Generating all different combinations of p, q and q triplets
pdq = list(itertools.product(p_range, d_range, q_range))

# Generate all different combinations of seasonal p, q and q triplets
# The term 4(or s) is the periodicity of the time series, 4 represents a month (roughly) for weekly series
seasonal_pdq = [(x[0], x[1], x[2], 7) for x in list(itertools.product(p_range, d_range, q_range))]


# Grid Search
# Finding best model fit, utilising AIC (Akaike Information Criterion)
def find_best_fit_parameters(data, params, seasonal_params):
    # To store parameters and AIC values to find the best fit
    parameters = []
    aic_values = []

    # Specifying to ignore warnings, because there will be a lot when the model won't fit :P
    warnings.filterwarnings("ignore")

    for param in params:
        for seasonal_param in seasonal_params:
            try:
                model = sm.tsa.statespace.SARIMAX(data,
                                                  order=param,
                                                  seasonal_order=seasonal_param,
                                                  enforce_stationarity=False,
                                                  enforce_invertibility=False)
                result = model.fit()
                parameters.append(str(param) + " x " + str(seasonal_param))
                aic_values.append(result.aic)
                print('ARIMA{} x {} - AIC:{}'.format(param, seasonal_param, result.aic))
            except:
                continue

    # Printing minimum AIC parameters
    min_index = aic_values.index(min(aic_values))
    best_parameters = parameters[min_index]
    print(f"Parameters with minimum index are: {best_parameters}")
    return best_parameters


def sarima_forecast(data, param, seasonal_param):
    model = sm.tsa.statespace.SARIMAX(data,
                                      order=param,
                                      seasonal_order=seasonal_param,
                                      enforce_stationarity=False,
                                      enforce_invertibility=False)
    results = model.fit()

    # The coef column shows the weight (i.e. importance) of each feature and how each one impacts the time series.
    # The P>|z| column informs us of the significance of each feature weight.
    print(results.summary().tables[1])

    # Visual diagnostics of our model
    # Our primary concern is to ensure that the residuals of our model
    # are uncorrelated and normally distributed with zero-mean.
    results.plot_diagnostics()
    plt.show()

    # Predictions
    # The dynamic=False argument ensures that we produce one-step ahead forecasts
    # meaning that forecasts at each point are generated using the full history up to that point.
    pred = results.get_prediction(start=pd.to_datetime('2021-05-16'), end=pd.to_datetime('2021-05-31'), dynamic=False)
    pred_ci = pred.conf_int()

    # Plotting
    ax = data.plot(label="actual data")
    pred.predicted_mean.plot(ax=ax, label="One-step ahead Forecast", alpha=.7)
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    plt.legend()

    plt.show()

    return results


def check_mse(data_actual, data_predicted):
    mse = ((data_predicted - data_actual) ** 2).mean()
    print(f"The Mean Squared Error is: {round(mse, 2)}")
    return mse


def forecast_steps(data, fitted_model, steps):
    # Get forecast of steps ahead in future
    pred_uc = fitted_model.get_forecast(steps=steps)

    # Get confidence intervals of forecasts
    pred_ci = pred_uc.conf_int()

    ax = data.plot(label='observed', figsize=(20, 15))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('CO2 Levels')

    plt.legend()
    plt.show()
