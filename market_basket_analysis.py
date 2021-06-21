# Data manipulation
import pandas as pd
import numpy as np

# Statistical computing
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Data Plotting and Visualisation
import matplotlib.pyplot as plt
import seaborn as sns  # Advanced visualization based on matplotlib

import constants
import utility_functions


# MARKET BASKET ANALYSIS: APRIORI ALGORITHM

# Data structuring for Apriori analysis
def structure_data(sales_data):
    # Adding a column with constant value 1
    sales_data["value"] = sales_data.apply(lambda x: 1, axis=1)

    # Dropping non-useful columns
    print(sales_data.columns.values)
    columns_to_drop = ['Invoice Number', 'Invoice Status', 'Invoice Type', 'Customer Name', 'Customer ID', 'Branch ID',
                       'Branch Name', 'Place of Supply', 'GST Identification Number (GSTIN)', 'Due Date',
                       'PurchaseOrder', 'Account', 'Item Name', 'SKU', 'Item Desc', 'Quantity', 'Usage unit',
                       'Warehouse Name', 'Item Price', 'HSN/SAC', 'Sales Order Number', 'Item Tax %', 'Item Tax Amount',
                       'Item Type', 'Item Total', 'SubTotal', 'Total', 'Adjustment', 'Round Off', 'Billing Address',
                       'Billing City', 'Billing State', 'Billing Code', 'Shipping Address', 'Shipping City',
                       'Shipping State', 'Shipping Code', 'CF.Material Type', 'CF.Driver, Vehicle & Helper Information']
    only_product_data = sales_data.drop(columns_to_drop, axis=1).reset_index(drop=True)
    print(only_product_data.head())

    # Reshaping dataframe to pivot form
    order_product_data = pd.pivot_table(only_product_data, index="Invoice ID", columns="Product ID", values="value")
    order_product_data.fillna(0, inplace=True)
    print(order_product_data)

    return order_product_data


def define_frequent_items(sales_data):
    structured_data = structure_data(sales_data)
    frequent_item_sets = apriori(structured_data, min_support=0.2, use_colnames=True)
    utility_functions.save_df_to_csv(frequent_item_sets, constants.calculated_data_folder, "frequent_itemsets")

    rules = association_rules(frequent_item_sets, metric="lift", min_threshold=1)

    return rules
