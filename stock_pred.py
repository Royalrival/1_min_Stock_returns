import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

quote_dir = '2024_04_01_Q'
trade_dir = '2024_04_01_T'

# Debugging for directories
# if not os.path.exists(quote_dir):
#     raise FileNotFoundError(f"Quote directory not found: {quote_dir}")
# if not os.path.exists(trade_dir):
#     raise FileNotFoundError(f"Trade directory not found: {trade_dir}")

def process_trade_file(file_path):
    df = pd.read_csv(file_path, header=None)
    df.columns = [
        'Date', 'Time', 'Price', 'Volume', 'Aggressor Side', 'Trade Period', 
        'Trade Id', 'Buyer', 'Buy Algo Type', 'Buy Order Capacity', 
        'Seller', 'Sell Algo Type', 'Sell Order Capacity'
    ]
    df['Time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.drop(columns=['Date'])
    df.set_index('Time', inplace=True)

    # Aggregating trade data to minute level
    df_minute = df.resample('min').agg({
        'Price': ['count', 'first', 'max', 'min', 'last'],
        'Volume': 'sum',
        'Aggressor Side': lambda x: (x == 'B').sum() / (x == 'S').sum() if (x == 'S').sum() != 0 else np.nan,
        'Buy Order Capacity': 'sum',
        'Sell Order Capacity': 'sum',
        'Trade Period': 'first',
    })
    df_minute.columns = ['_'.join(col).strip() for col in df_minute.columns.values]
    
    df['Weighted Price'] = df['Price'] * df['Volume']
    weighted_price = df['Weighted Price'].resample('min').sum() / df['Volume'].resample('min').sum()
    df_minute['weighted_price'] = weighted_price

    # Feature Engineering for Trade Data
    df_minute['num_trades'] = df_minute['Price_count']
    df_minute['o'] = df_minute['Price_first']
    df_minute['h'] = df_minute['Price_max']
    df_minute['l'] = df_minute['Price_min']
    df_minute['c'] = df_minute['Price_last']
    df_minute['total_volume'] = df_minute['Volume_sum']
    df_minute['trade_imbalance_ratio'] = df_minute['Aggressor Side_<lambda>']
    df_minute['order_cap_imbalance_ratio'] = (df_minute['Buy Order Capacity_sum'] != df_minute['Sell Order Capacity_sum']).astype(int)

    return df_minute

def process_quote_file(file_path):
    df = pd.read_csv(file_path, header=None)
    df.columns = ['Date', 'Time', 'Bid Price', 'Bid Size', 'Ask Price', 'Ask Size']
    df['Time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.drop(columns=['Date'])
    df.set_index('Time', inplace=True)

    # Aggregating quote data to minute level
    df_minute = df.resample('min').agg({
        'Bid Price': ['mean', 'max', 'min'],
        'Ask Price': ['mean', 'max', 'min'],
        'Bid Size': 'sum',
        'Ask Size': 'sum'
    })
    df_minute.columns = ['_'.join(col).strip() for col in df_minute.columns.values]
    
    # Feature Engineering for Quote Data
    df_minute['avg_spread'] = df_minute['Ask Price_mean'] - df_minute['Bid Price_mean']
    df_minute['max_spread'] = df_minute['Ask Price_max'] - df_minute['Bid Price_min']
    df_minute['min_spread'] = df_minute['Ask Price_min'] - df_minute['Bid Price_max']
    df_minute['total_bid_size'] = df_minute['Bid Size_sum']
    df_minute['total_ask_size'] = df_minute['Ask Size_sum']
    df_minute['weighted_avg_bid_price'] = df_minute['Bid Price_mean'] * df_minute['Bid Size_sum'] / df_minute['Bid Size_sum'].sum()
    df_minute['weighted_avg_ask_price'] = df_minute['Ask Price_mean'] * df_minute['Ask Size_sum'] / df_minute['Ask Size_sum'].sum()

    return df_minute

# Function to process all files in the directory
def process_directory(directory_path, is_trade_data):
    all_data = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.asc'):
            file_path = os.path.join(directory_path, file_name)
            if is_trade_data:
                df = process_trade_file(file_path)
            else:
                df = process_quote_file(file_path)
            company_name = file_name.split('_')[0]
            df['Company'] = company_name  
            all_data.append(df)
    return pd.concat(all_data)

trade_data_all = process_directory(trade_dir, is_trade_data=True)
quote_data_all = process_directory(quote_dir, is_trade_data=False)

# Merging trade and quote data on timestamp
merged_data = pd.merge(trade_data_all, quote_data_all, left_index=True, right_index=True, suffixes=('_trade', '_quote'))

# Dropping redundant company columns if any
if 'Company_trade' in merged_data.columns:
    merged_data = merged_data.drop(columns=['Company_trade'])
if 'Company_quote' in merged_data.columns:
    merged_data = merged_data.drop(columns=['Company_quote'])

# Encoding 'Trade Period' categorical column as numerical
merged_data['Trade Period_first'] = merged_data['Trade Period_first'].map({'O': 0, 'T': 1, '-': 2})

# Defining target variable: 1-minute return based on close price
merged_data['return'] = merged_data['c'].pct_change(fill_method=None).shift(-1)
merged_data = merged_data.dropna()

X = merged_data.drop(columns=['return'])
y = merged_data['return']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model evaluation using Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plotting actual vs predicted returns over time
plt.figure(figsize=(10, 6))
plt.plot(X_test.index, y_test, label='Actual Returns')
plt.plot(X_test.index, y_pred, label='Predicted Returns')
plt.xlabel('Time')
plt.ylabel('Returns')
plt.title('Actual vs Predicted Returns Over Time')
plt.legend()
plt.show()

# Plotting correlation of features with 1-minute return across companies
correlation_with_return = merged_data.groupby('Company').apply(lambda x: x.corr()['return'].drop('return'))
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_with_return, cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5)
plt.title('Correlation of Features with 1-Minute Return Across Companies')
plt.xlabel('Features')
plt.ylabel('Company')
plt.xticks(rotation=90)
plt.show()