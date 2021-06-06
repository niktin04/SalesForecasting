import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

# Data links
raw_data_folder = "raw_data/"
sales_data_csv = ["burgrill_sales_2020_12.csv",
                  "burgrill_sales_2021_01.csv",
                  "burgrill_sales_2021_02.csv",
                  "burgrill_sales_2021_03.csv",
                  "burgrill_sales_2021_04.csv",
                  "burgrill_sales_2021_05.csv"]


# Loading data into pandas dataframe
def load_data(data_link):
    print(f"Generating pandas dataframe from: {raw_data_folder + data_link}")
    df = pd.read_csv(data_link)
    print(df.info())
    return df


def clean_zoho_data(df):
    cleaned_df = df.copy()

    # Change dates to datetime object
    cleaned_df["Invoice Date"] = pd.to_datetime(cleaned_df["Invoice Date"], format="%Y-%m-%d")
    cleaned_df["Due Date"] = pd.to_datetime(cleaned_df["Due Date"], format="%Y-%m-%d")

    # Remove "Draft" and "Void" data
    # print(cleaned_burgrill_df["Invoice Status"].unique())
    cleaned_df = cleaned_df[(cleaned_df["Invoice Status"] != "Draft") & (cleaned_df["Invoice Status"] != "Void")]

    # Remove non-burgrill customers
    cleaned_df = cleaned_df[cleaned_df["Customer Name"].str.contains("DBH -")]
    cleaned_df = cleaned_df[cleaned_df["Customer Name"] != "DBH - Damaged Products - Okhla, Delhi"]

    # Remove NaN values
    cleaned_df = cleaned_df.fillna("")

    return cleaned_df


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
for file_link in sales_data_csv:
    sales_dfs.append(load_data(raw_data_folder + file_link))

# Combine multiple dataframes into one
burgrill_df = pd.concat(sales_dfs)

cleaned_burgrill_df = clean_zoho_data(burgrill_df)
daily_sales_burgrill_df = daily_sales_data(cleaned_burgrill_df)
weekly_sales_burgrill_df = weekly_sales_data(cleaned_burgrill_df)
monthly_sales_burgrill_df = monthly_sales_data(cleaned_burgrill_df)
print(weekly_sales_burgrill_df)
weekly_sales_burgrill_diff_df = get_diff_data(weekly_sales_burgrill_df)
print(weekly_sales_burgrill_diff_df)
# weekly_sales_burgrill_df.plot(x="Invoice Week", y="Item Total", kind="line")
# plt.show()
