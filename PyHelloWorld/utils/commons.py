import os
import pandas as pd
import datetime
from pandas_datareader import data as pdr
import yfinance as yfin

from consts import NSE_BSE, OUT_DIR, HOLDING_FILE


def save_csv(df, f_prefix):
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    f_name = os.path.join(OUT_DIR, f'{f_prefix}_{now}.csv')

    df.to_csv(f_name)
    print(f'Saved: {f_name}')

def read_xls(file_name, tab=0):
    xls = pd.ExcelFile(file_name)
    df = xls.parse(tab)

    return df

def get_curr_prices_from_holdings():
    df = read_xls(HOLDING_FILE)
    df = df[['Symbol', 'Qty', 'Curr Price']]
    df = df[df['Qty'] > 0] 
    df = df.rename(columns={'Curr Price': 'currPrice', 'Symbol': 'index'})
    df = df[['index', 'currPrice']]
    
    return df

def get_holding_quantities():
    df = read_xls(HOLDING_FILE, tab=0)
    df = df[['Symbol', 'Qty', 'Curr Val']]
    df = df.sort_values('Curr Val', ascending=False)
    df = df[df['Qty'] > 0][['Symbol', 'Qty']]
#     dir(df)
#     df.to_json()
    list(df.values)
    df_dict = df.to_dict('list') # https://stackoverflow.com/questions/52547805/how-to-convert-dataframe-to-dictionary-in-pandas-without-index
    qtys = {}
    for c, v in zip(df_dict['Symbol'], df_dict['Qty']):
        qtys[c] = v
        
    return qtys

def get_stock_data(stock_list=None, 
                   start=datetime.datetime.today().date(), 
                   end=datetime.datetime.today().date(), 
                   interval='1d',
                   print_data=None):
    if not stock_list:
        stock_list = [NSE_BSE[n] for n in get_holding_quantities().keys()] # list(NSE_BSE.values()) # ['BAJFINANCE.BO', 'DMART.NS']

    # d = web.DataReader(stocks, 'yahoo', start, end)
    # d = data.DataReader("BAJFINANCE.BO",'yahoo', start='2021-09-10', end='2022-10-09')

    yfin.pdr_override()
    print(f'****** Getting stock data between {start} and {end} *******')
    df = pdr.get_data_yahoo(stock_list, start=start, end=end, interval=interval) #['Adj Close']
    print(f'****** Got stock data between {start} and {end} *******')
    
    if interval == '1wk':
        df = df.asfreq('W-FRI', method='pad')
    df = df['Adj Close']
    df = df.rename(columns={NSE_BSE[n]:n for n in get_holding_quantities().keys()})

    if print_data:
        print(df)
    return df

def min_consecutive_sum_kadane(arr, dates=None):
    max_so_far = float('-inf')
    max_ending_here = 0
    _start, start, end = 0, 0, -1

    for i in range(0, len(arr)):
        arr[i] = arr[i] * -1
        max_ending_here = max_ending_here + arr[i]

        if max_ending_here < 0:
            max_ending_here = 0
            _start = i+1

        if max_so_far < max_ending_here:
            max_so_far = max_ending_here
            start = _start
            end = i

    if dates and start <= end and start <= len(dates) and len(dates) >= end:
        if isinstance(dates[start], pd.Timestamp):
            start = dates[start].date().strftime('%d%b%Y')
            end = dates[end].date().strftime('%d%b%Y')

    return [max_so_far * -1, start, end]



# print(min_consecutive_sum_kadane(arr=[10, -15, 2, 25, -16, -53, 22, 2, 1, -3, 60]))  # -69
# print(min_consecutive_sum_kadane(arr=[10, -15, -2, 10, -16, -53, 22, 2, 1, -3, 60]) * -1) # -76
# print(get_stock_data())

