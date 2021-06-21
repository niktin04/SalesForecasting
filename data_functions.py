# Data manipulation
import pandas as pd

# Debug
debug = True


# COMBINING DATA
def combine_data(data_folder, files_list):
    dfs = []
    for file in files_list:
        df = pd.read_csv(data_folder + file)
        dfs.append(df)

        if debug:
            print(f"Creating dataframe for: {data_folder + file}")
            print(df.head())

    if debug:
        print("Combining databases...")

    combined_df = pd.concat(dfs)

    if debug:
        print("Combined database")
        print(combined_df)

    return combined_df


# CLEANING DATA
def clean_zoho_sales_data(df):
    if debug:
        print("Cleaning dataframe...")

    cleaned_df = df.copy()

    # Change dates to datetime object
    cleaned_df["Invoice Date"] = pd.to_datetime(cleaned_df["Invoice Date"], format="%Y-%m-%d")
    cleaned_df["Due Date"] = pd.to_datetime(cleaned_df["Due Date"], format="%Y-%m-%d")

    if debug:
        print(f"Changed column 'Invoice Date' data type to: {cleaned_df.dtypes['Invoice Date']}")
        print(f"Changed column 'Due Date' data type to: {cleaned_df.dtypes['Invoice Date']}")

    # Remove "Draft" and "Void" data
    # print(cleaned_burgrill_df["Invoice Status"].unique())
    cleaned_df = cleaned_df[(cleaned_df["Invoice Status"] != "Draft") & (cleaned_df["Invoice Status"] != "Void")]

    if debug:
        print(f"Removed data with 'Draft' and 'Void' status, remaining types:\n{cleaned_df['Invoice Status'].unique()}")

    # Remove non-burgrill customers
    cleaned_df = cleaned_df[cleaned_df["Customer Name"].str.contains("DBH -")]
    cleaned_df = cleaned_df[cleaned_df["Customer Name"] != "DBH - Damaged Products - Okhla, Delhi"]

    if debug:
        print(f"Removed non-burgrill customers, remaining customers:\n{cleaned_df['Customer Name'].unique()}")

    # Remove NaN values
    cleaned_df = cleaned_df.fillna("")

    if debug:
        print(f"Removed NaN values from dataframe")

    # Change index to "Invoice Date"
    cleaned_df.set_index("Invoice Date", inplace=True)

    if debug:
        print(f"Changed index of df to 'Invoice Date' column")
        print(cleaned_df.head())

    return cleaned_df
