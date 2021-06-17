# Data manipulation
import pandas as pd

# Statistical computing
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# Data Plotting and Visualisation
import matplotlib.pyplot as plt
import seaborn as sns  # Advanced visualization based on matplotlib

# Custom files
import data_functions
import visual_functions
import forecasting_functions

# Data path and links
calculated_data_folder = "calculated_data/"
raw_data_folder = "raw_data/"
sales_data_csv_list = [
    "burgrill_sales_2020_12.csv",
    "burgrill_sales_2021_01.csv",
    "burgrill_sales_2021_02.csv",
    "burgrill_sales_2021_03.csv",
    "burgrill_sales_2021_04.csv",
    "burgrill_sales_2021_05.csv",
]
purchase_data_csv_list = [
    "burgrill_purchase_2020_12_2021_05.csv",
]

# Combining data
sales_data = data_functions.combine_data(raw_data_folder, sales_data_csv_list)

# Cleaning data
sales_data = data_functions.clean_zoho_data(sales_data)

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

# Sales data
daily_sales_data = sales_data["Item Total"].resample("D").sum()
weekly_sales_data = sales_data["Item Total"].resample("W-MON").sum()
monthly_sales_data = sales_data["Item Total"].resample("MS").sum()

# Visual data
visual_functions.show_data(daily_sales_data)
# visual_functions.show_data(weekly_sales_data)
# visual_functions.show_data(monthly_sales_data)

# TSD
visual_functions.time_series_decomposition(daily_sales_data)


def daily_sales_data(df):
    daily_sales_df = df.copy()

    # Sum of sales per day
    daily_sales_df = daily_sales_df.groupby("Invoice Date")["Item Total"].sum().reset_index()

    return daily_sales_df


def weekly_sales_data(df):
    weekly_sales_df = df.copy()

    # Adding "Invoice Week" column
    weekly_sales_df["Invoice Week"] = pd.to_datetime("1-" + weekly_sales_df["Invoice Date"].dt.strftime("%W-%Y"),
                                                     format="%w-%W-%Y")

    # Sum of sales per week
    weekly_sales_df = weekly_sales_df.groupby("Invoice Week")["Item Total"].sum().reset_index()

    # Adding percentage change
    weekly_sales_df["Weekly Growth"] = weekly_sales_df["Item Total"].pct_change()

    return weekly_sales_df


def monthly_sales_data(df):
    monthly_sales_df = df.copy()

    # Adding month column
    monthly_sales_df["Invoice Month"] = pd.to_datetime(monthly_sales_df["Invoice Date"].dt.strftime("%b %Y"),
                                                       format="%b %Y")

    # Sum of sales per month
    monthly_sales_df = monthly_sales_df.groupby("Invoice Month")["Item Total"].sum().reset_index()

    return monthly_sales_df


def get_diff_data(df):
    df["Item Total Diff"] = df["Item Total"].diff()
    df = df.dropna()
    return df


def time_plot(df, x_col, y_col, title):
    fig, ax = plt.subplots(figsize=(15, 5))


# Generating separate dataframes for difference months
sales_dfs = []
for file_link in sales_data_csv_list:
    sales_dfs.append(load_data(raw_data_folder + file_link))

# Combine multiple dataframes into one
burgrill_df = pd.concat(sales_dfs)

cleaned_burgrill_df = clean_zoho_data(burgrill_df)
daily_sales_burgrill_df = daily_sales_data(cleaned_burgrill_df)
weekly_sales_burgrill_df = weekly_sales_data(cleaned_burgrill_df)
monthly_sales_burgrill_df = monthly_sales_data(cleaned_burgrill_df)

print(weekly_sales_burgrill_df)
print(monthly_sales_burgrill_df)

# weekly_sales_burgrill_diff_df = get_diff_data(weekly_sales_burgrill_df)
# weekly_sales_burgrill_diff_df.head()

# weekly_sales_burgrill_df.plot(x="Invoice Week", y="Weekly Growth", kind="line")
# plt.show()

# North Star Metric is the single metric that best captures the core value that your product delivers to customers
# Our North Star Metric: Revenue
# Revenue = Active Customers Count * Order Count * Avg. Revenue per Order

# Per month & Per week analytics
# For Existing Customers and for new Customers
# Active Customers, Orders Count, Revenue, Avg. Revenue per Customer, Avg. Revenue per Order

# New Customers Ratio

# Monthly Retention Rate
# Monthly Retention Rate = Retained Customers From Prev. Month/Active Customers Total

# TIME LINES (MONTHLY,
# Sales by Region

# Geographical Revenue Distribution
# TABLE SCHEMA
# Customer ID, Customer Name, Order Count, Revenue, % Share

product_list = cleaned_burgrill_df["Item Name"].unique()
print(len(product_list))


# Customer Based Revenues
def customer_revenue(df):
    customer_revenue_df = df.copy()
    customer_revenue_df = customer_revenue_df.groupby("Customer Name")["Item Total"].sum().reset_index()
    customer_revenue_df = customer_revenue_df.sort_values("Item Total", ascending=False)

    print(customer_revenue_df)


def state_revenue(df):
    state_revenue_df = df.copy()
    state_revenue_df = state_revenue_df.groupby("Shipping State")["Item Total"].sum().reset_index()
    state_revenue_df = state_revenue_df.sort_values("Item Total", ascending=False)

    print(state_revenue_df)


def city_revenue(df):
    city_revenue_df = df.copy()
    city_revenue_df = city_revenue_df.groupby("Shipping City")["Item Total"].sum().reset_index()
    city_revenue_df = city_revenue_df.sort_values("Item Total", ascending=False)

    print(city_revenue_df)


customer_revenue(cleaned_burgrill_df)
state_revenue(cleaned_burgrill_df)
city_revenue(cleaned_burgrill_df)


def customer_list(data):
    customer_ids = data["Customer ID"].unique()
    customer_list_df = pd.DataFrame(customer_ids,
                                    columns=["CustomerID", "CustomerName", "FirstOrder", "LastOrder", "Age"])
    print(customer_list_df)

    # Finding first and last order dates


# customer_list(cleaned_burgrill_df)
# Schema
# Customer ID, Customer Name, First Order Date, Last Order Date

def get_product_data(data):
    product_ids = data["Product ID"].unique()
    print(product_ids)

    # PRODUCT TABLE
    # 1. Basic Product Information (Unique Combination)
    # ID, Name, Description, Tax, HSN, Unit, CP, SP

    # 2. Purchase Information
    # Bill Count, Purchase Qty, Purchase Amount, Purchase Frequency Days, First Bill Date, Last Bill Date

    # 3. Selling Information
    # Invoice Count, Sold Qty, Sold Amount, Invoice Frequency Days, First Order Date, Last Order Date

    # 4. Margin Details
    # Margin %, Margin Amount, Total Margin Amount


def get_customer_data(data):
    product_ids = data["Product ID"].unique()
    print(product_ids)

    # CUSTOMER TABLE
    # 1. Basic Customer Information (Unique Combination)
    # ID, Name, GST, Billing State, Shipping Address, Shipping Area, Shipping City, Shipping State, Shipping Code

    # 2. Invoicing Information
    # Count, Subtotal Amount, Total Amount, Frequency Days, First Order Date, Last Order Date


def get_ordering_behaviour(data):
    pass

    # CUSTOMER ORDERING BEHAVIOUR TABLE
    # 1. Basic Customer Information (Unique Combination)
    # ID, Name, GST, Billing State, Shipping Address, Shipping Area, Shipping City, Shipping State, Shipping Code

    # 2. Invoicing Information
    # Count, Subtotal Amount, Total Amount, Frequency Days, First Order Date, Last Order Date


def get_product_data(cleaned_burgrill_df):
    pass


def get_weekly_revenue_data(data):
    # Making new dataframe
    weekly_revenue_df = data.copy()

    # Customising dataframe
    weekly_revenue_df["Week"] = weekly_revenue_df["Invoice Date"].dt.strftime("%W %Y")
    weekly_revenue_df["Week"] = pd.to_datetime("1 " + weekly_revenue_df["Week"], format="%w %W %Y")
    weekly_revenue_df = weekly_revenue_df.groupby("Week")["Item Total"].sum().reset_index()

    # Saving values to CSV file
    weekly_revenue_df.to_csv(calculated_data_folder + "weekly_revenue.csv")

    # Printing values
    print("Weekly Revenue Data:")
    print(weekly_revenue_df)

    # Plotting values
    # plot_line_graph(weekly_revenue_df, "Week", "Item Total", "Weekly Revenue Data")

    return weekly_revenue_df


# get_weekly_revenue_data(cleaned_burgrill_df)


def define_arima_parameters(y, test_p, test_d, test_q, training_portion):
    # ARIMA: AutoRegressive Integrated Moving Average

    # "p" is the order of the ‘Auto Regressive’ (AR) term.
    # It refers to the number of lags of Y to be used as predictors.

    # "d" is the minimum number of differencing needed to make the series stationary

    # "q" is the order of the ‘Moving Average’ (MA) term.
    # It refers to the number of lagged forecast errors that should go into the ARIMA Model.

    # Data split to be used for training and for testing
    y_len = len(y)
    y_len_training = int(y_len // (1 / training_portion))
    y_len_testing = y_len - y_len_training

    print(f"Data length: {y_len}, training length: {y_len_training}, testing length: {y_len_testing}")

    errors = []
    parameters = []
    for p in test_p:
        for d in test_d:
            for q in test_q:
                print(f"Trying order (p, d, q) values: ({p}, {d}, {q})")
                try:
                    preds = y[0:y_len_training]
                    for i in range(0, y_len_testing):
                        model = ARIMA(preds, order=(p, d, q))
                        fit = model.fit()
                        pred = fit.predict(y_len_training + i, y_len_training + i, typ='levels')
                        preds = preds.append(pred)
                    error = sm.tools.eval_measures.meanabs(y, preds)
                    errors.append(error)
                    parameters.append((p, d, q))
                except Exception as ex:
                    print(f"Exception: {ex}")
    print(f"Differences:\n{errors}")
    print(f"ARIMA order parameters:\n{parameters}")
    index = errors.index(min(errors))
    parameters = parameters[index]
    print(f"The optimal (p,d,q) values are {parameters} with MAE {errors[index]}")
    return parameters


def arima(y, arima_order, training_portion, pred_count=None):
    # Data split to be used for training and for testing
    y_len = len(y)
    y_len_training = int(y_len // (1 / training_portion))

    if pred_count is None:
        y_len_testing = y_len - y_len_training
    else:
        y_len_testing = y_len - y_len_training + pred_count

    print(f"Data length: {y_len}, training length: {y_len_training}, testing length: {y_len_testing}")

    preds = y[0:y_len_training]
    model = ARIMA(y, order=arima_order)
    fit = model.fit()
    for i in range(0, y_len_testing):
        pred = fit.predict(y_len_training + i, y_len_training + i, typ='levels')
        preds = preds.append(pred)

    plt.plot(np.arange(0, y_len_training + y_len_testing), preds, y)
    plt.show()


weekly_revenue_df = get_weekly_revenue_data(cleaned_burgrill_df)

test_p = [1, 2, 3, 4, 6, 8]
test_d = [0, 1, 2, 3]
test_q = [0, 1, 2, 3]

# Testing our ARIMA values
arima_order = define_arima_parameters(weekly_revenue_df["Item Total"], test_p, test_d, test_q, 0.72)
arima(weekly_revenue_df["Item Total"], arima_order, 0.72)

# Real Prediction
arima_order = define_arima_parameters(weekly_revenue_df["Item Total"], test_p, test_d, test_q, 0.96)
arima(weekly_revenue_df["Item Total"], arima_order, 0.96, pred_count=4)
