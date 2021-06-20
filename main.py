# Data manipulation
import pandas as pd
import numpy as np

# Statistical computing
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# Data Plotting and Visualisation
import matplotlib.pyplot as plt
import seaborn as sns  # Advanced visualization based on matplotlib

# Custom functions
import constants
import utility_functions
import data_functions
import visual_functions
import forecasting_functions
import market_basket_analysis

# Combining and cleaning data
sales_data = data_functions.combine_data(constants.raw_data_folder, constants.sales_data_csv_list)
sales_data = data_functions.clean_zoho_data(sales_data)

utility_functions.save_df_to_csv(sales_data, constants.calculated_data_folder, "sales_data")

# Extracting products data
# Product ID, Product Name, Product Unit, Usage Starting Date, Usage Ending Date,
# Total Sold Qty, Total Orders, Avg. Qty Sold/Order, Avg. Order Frequency,
# Min. Price, Max. Price, Avg. Price
products_list = sales_data["Item Name"].unique()
print(products_list)
products_data = 0

# Extracting customers data
# Customer ID, Customer Name, Order Starting Date, Order Ending Date,
# Total Sold Qty, Total Orders, Avg. Qty Sold/Order, Avg. Order Frequency,
# Min. Price, Max. Price, Avg. Price
customers_data = 0

# Sales data: Daily, Weekly, Monthly
daily_sales_data = sales_data["Item Total"].resample("D").sum()  # D for day
weekly_sales_data = sales_data["Item Total"].resample("W-MON").sum()  # W for week, MON for Monday as start of the week
monthly_sales_data = sales_data["Item Total"].resample("MS").sum()  # M for month, S for starting date of the month

# Visual data
# visual_functions.show_data(daily_sales_data)
# visual_functions.show_data(weekly_sales_data)
# visual_functions.show_data(monthly_sales_data)

# Time Series Decomposition
# Data must have 2 complete cycles, or minimum 104 observations in the time series
# visual_functions.time_series_decomposition(daily_sales_data)
# visual_functions.time_series_decomposition(weekly_sales_data) # 26 observations only
# visual_functions.time_series_decomposition(monthly_sales_data) # 5 observations only

# Finding SARIMA parameters
# best_fit_parameters = forecasting_functions.find_best_fit_parameters(daily_sales_data,
#                                                                      forecasting_functions.pdq,
#                                                                      forecasting_functions.seasonal_pdq)
# forecasting_model = forecasting_functions.sarima_forecast(daily_sales_data,
#                                                           best_fit_parameters.split(" x ")[0],
#                                                           best_fit_parameters.split(" x ")[1])
# forecasted_data = forecasting_functions.forecast_steps(daily_sales_data, 15)

# Finding similar products
only_dry_data = sales_data[(sales_data["CF.Material Type"] == "Dry Goods")]
rules = market_basket_analysis.define_frequent_items(only_dry_data)
utility_functions.save_df_to_csv(rules, constants.calculated_data_folder, "apriori_rules")