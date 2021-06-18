import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.api as sm

# Plot Styling
plt.style.use('fivethirtyeight')


# Seasonality: Does the data display a clear periodic pattern?
# Trend: Does the data follow a consistent upwards or downward slope?
# Noise: Are there any outlier points or missing values that are not consistent with the rest of the data?
def show_data(sales_data):
    sales_data.plot(figsize=(15, 6))
    plt.show()


# Time Series Decomposition
# 1. Check for trend
# 2. Check for seasonality, should have cyclic pattern
# 3. Check for residual, pattern (of any kind) should not be there in residual
def time_series_decomposition(sales_data):
    decomposition = sm.tsa.seasonal_decompose(sales_data, model='additive')
    fig = decomposition.plot()
    plt.show()
