# Data manipulation
import pandas
import pandas as pd
import numpy as np

# Statistical computing
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# Data Plotting and Visualisation
import matplotlib.pyplot as plt
import seaborn as sns  # Advanced visualization based on matplotlib


def generate_product_report(sales_data):
    product_data = pandas.DataFrame()
    # Product ID, Product Name, Product Unit, Usage Starting Date, Usage Ending Date,
    # Total Sold Qty, Total Orders, Avg. Qty Sold/Order, Avg. Order Frequency,
    # Min. Price, Max. Price, Avg. Price
    sales_data.columns()

    # Drop unwanted columns

    # List unique products
    product_data["Product ID"] = sales_data["Product ID"].unique()

    # List names corresponding to "Product ID"

    pass


def market_basket_analysis():
    pass
