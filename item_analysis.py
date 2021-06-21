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


def clean_zoho_item_data(df):
    # Printing Information
    if constants.debug:
        print("Cleaning zoho item data...")

    # Copying zoho item's database for analysis
    cleaned_df = df.copy()

    # Remove "Inactive" data
    # print(cleaned_burgrill_df["Invoice Status"].unique())
    cleaned_df = cleaned_df[(cleaned_df["Status"] != "Inactive")]

    # Printing Information
    if constants.debug:
        print(f"Removed data with 'Inactive' status, remaining types:\n{cleaned_df['Status'].unique()}")

    # Remove non-burgrill items
    cleaned_df = cleaned_df[cleaned_df["Item Name"].str.contains("DBH -")]

    # Printing Information
    if constants.debug:
        print(f"Removed non-burgrill items, remaining items:\n{cleaned_df['Item Name'].unique()}")

    # Remove NaN values
    cleaned_df = cleaned_df.fillna("")

    # Printing Information
    if constants.debug:
        print(f"Removed NaN values from dataframe")

    # Change index to "Invoice Date"
    cleaned_df.reset_index(drop=True, inplace=True)

    # Printing Information
    if constants.debug:
        print(f"Index reset")
        print(cleaned_df.head())

    # Saving item_list as csv for future reference
    utility_functions.save_df_to_csv(cleaned_df, constants.calculated_data_folder, "item_list")

    # Printing Information
    if constants.debug:
        print(f"Saved cleaned zoho item data to: {constants.calculated_data_folder + 'item_list.csv'}")

    return cleaned_df


def generate_product_report(item_list, sales_data, purchase_data):
    # Copying item_list database for analysis
    item_report = item_list.copy()

    # Dropping unwanted columns and resetting index
    columns_to_drop = ["Sales Account", "Is Returnable Item", "Package Weight", "Package Length",
                       "Package Width", "Package Height", "Taxable", "Exemption Reason", "Source", "Reference ID",
                       "Last Sync Time", "Status", "SKU", "UPC", "EAN", "ISBN", "Part Number", "Purchase Account",
                       "Inventory Account", "Reorder Level", "Is Combo Product", "Item Type", "Intra State Tax Name",
                       "Intra State Tax Type", "Inter State Tax Name", "Inter State Tax Type", "CF.Purchase MOQ",
                       "CF.Purchase Increment Value", "CF.TotalPurchaseQty", "CF.Total Purchase Cost",
                       "Outsourcing Brand", "Opening Stock", "Opening Stock Value"]
    item_report = item_report.drop(columns_to_drop, axis=1).reset_index(drop=True)

    # Saving item_report as csv for future reference
    utility_functions.save_df_to_csv(item_report, constants.calculated_data_folder, "item_report")

    return item_report
