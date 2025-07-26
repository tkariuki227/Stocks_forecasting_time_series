import pandas as pd

merged_df = pd.read_pickle('merged_nse_df.pkl')

def get_stock(merged_df, stock_name):
    individual_stock = merged_df[merged_df['Name'] == stock_name].copy()
    individual_stock = individual_stock.sort_values('Date')
    return individual_stock