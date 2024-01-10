import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
from time import sleep
import re
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
warnings.filterwarnings("ignore")


from consts import PROFIT_LOSS_YEARS, SCREENER_ROW_PL, CURRENT_HIGH_THRESHOLD_PERCENT, BASE_URL
from consts import SCREENER_TOP_DATA, CONSOLIDATED_NOT_AVAILABLE_ON_SCREENER, NBFC_KEY_MAP
from commons import get_holding_quantities

def floatify(val):
    if not val:
        return None
    
    if isinstance(val, str): 
        if ',' in val:
            val = val.replace(',', '')
        if '%' in val:
            val = float(val.split('%')[0]) * .01

    return float(val) 

def extract_compound_tables(soup, section_id, class_name):
    section_html = soup.find('section',{'id': section_id})

    data = {}
    for contents in section_html.find_all('table', class_=class_name):
        for block in contents.find_all('tr'):
            if block.th:
                row_name = block.th.text
            row_data = block.find_all('td')
            if row_data:
                row = [tr.text.strip() for tr in row_data]
                # print(row)
                gain = row[1].split('%')[0]
                
                data[f"[{row_name}][{row[0].replace(':', '')}]%"] = float(gain)*.01 if gain else None
      
    return data      


def extract_table_by_class(soup, section_id, class_name):
    section_html = soup.find('section',{'id': section_id})
    table_html = section_html.find('table',{'class': class_name})

    headers = []
    for header in table_html.find_all('th'):
        headers.append(  header.text or 'Type')

    table_df = pd.DataFrame(columns = headers)

    for row_element in table_html.find_all('tr')[1:]:
            row_data = row_element.find_all('td')
            row = [tr.text.strip() for tr in row_data]
            length = len(table_df)
            table_df.loc[length] = row 
            
    return table_df
    
def fetch_number_span(list_element):
    num_span = list_element.find('span',{'class':'number'})

    num_span = num_span.text.replace(',', '')
    return float(num_span) if (num_span != '') else 0.0
    
def extract_scrip_ratios(soup,div_class, ul_id):
    div_html = soup.find('div',{'class': div_class})
    ul_html = div_html.find('ul',{'id': ul_id})
     
    scrip_data = pd.Series()
    for d in SCREENER_TOP_DATA:
        for li in ul_html.find_all("li"):
            name_span = li.find('span',{'class':'name'})
            
            if d in name_span.text: 
                scrip_data[d] = fetch_number_span(li)

    return scrip_data


def fetch_scrip_data(scrip, consolidated=True):
    link = f'{BASE_URL}{scrip}'
    link += '/consolidated' if consolidated else ''
    
    hdr = {'User-Agent':'Mozilla/5.0'}
    req = Request(link,headers=hdr)
    
    profit_loss_df = None
    scrip_data = pd.Series()
    try:
        page=urlopen(req)
        soup = BeautifulSoup(page)
        scrip_data = extract_scrip_ratios(soup,'company-ratios', 'top-ratios')
        profit_loss_df = extract_table_by_class(soup, 'profit-loss', 'data-table responsive-text-nowrap')
    except:
        print(f'EXCEPTION THROWN: UNABLE TO FETCH DATA')

    return scrip_data, profit_loss_df

def extract_last_n_years_pl(pl_df, n_years):
    # Extract data for all years from the column names
    mon_year_regex = re.compile('([A-Z][a-z]{2}) (\d{4})')
    years = {}
    for col in list(pl_df.columns):
        res = re.search(mon_year_regex,col)
        if res:
            years[res.group(2)] = col

    # Get only the last n (PROFIT_LOSS_YEARS) years for checking the P&L 
    years_list = sorted(years.keys())
    years_list = years_list[-n_years:]
    cols = [years[year] for year in years_list]
    print(pl_df[cols])
    
    pl_values = pl_df[cols].iloc[0, :].values.tolist()
    pl_values = [float(x.replace(',', '')) for x in pl_values] 
    return pl_values

def check_current_below_high_threshold(current,high, threshold_percent):
    '''
        Check if current price is below the 52-week high with a certain threshold
        Eg: If current price is 100, 52-week high is 120, threshold is 10%, then return True
        If current price is 100, 52-week high is 105, threshold is 10%, then return False
    '''
    below_threshold = False
    if ((current < high) & ((high-current)/high*100 > threshold_percent)):
        below_threshold = True
    return below_threshold   

def apply_pl_strategy(current_price, scrip_high, profit_loss_df, high_threshold_percent, scrip=None):   
#     STRATEGY:
#     BUY recommendation if:
#         1. Profit/Loss for the company has been increasing consistently in the last few years.
#         2. Current market price is below 10% of 52-week high

    # SET DEFAULT TO STOCK AS NO-ACTION
    strategy_result = 'WAIT'
    try: 

        # CHECK IF REQUIRED VALUES COULD BE SCRAPED
        if (current_price is None or current_price == 0.0 or 
            scrip_high is None or scrip_high == 0.0):
            strategy_result = 'NOT FOUND'

        else:
            profit_loss_df = profit_loss_df[profit_loss_df['Type'] == SCREENER_ROW_PL]
            last_pl_list = extract_last_n_years_pl(profit_loss_df, PROFIT_LOSS_YEARS)
            print(f'Profit/Loss for last {PROFIT_LOSS_YEARS} years:{last_pl_list}')
            print(f'Current Price:{current_price}, 52-week High:{scrip_high}, Threshold%: {high_threshold_percent}%')

            # CHECK IF PROFIT-LOSS IS CONSISTENTLY INCREASING
            if(last_pl_list == sorted(last_pl_list)):
                # IF YES, CHECK IF CURRENT MARKET VALUE IS NOT AT ALL TIME HIGH
                if check_current_below_high_threshold(current_price, scrip_high, high_threshold_percent):
                    # BUY RECOMMENDATION
                    strategy_result = 'BUY'
    except Exception as e:
        print(f"UNABLE TO APPLY PROFIT-LOSS STRATEGY ON {scrip}. Exception: {e}")

    return strategy_result

def process_key(key):
    for k, v in NBFC_KEY_MAP.items():
        if k in key:
            key = key.replace(k, v) 

    # key = re.sub(r'\W+', '', key) #Remove everything except alpha-numeric
    key = re.sub(r'[^a-zA-Z0-9\[\]]+', '', key)

    return key


def flatten_df(df, append_name=None):
    df_dict = df.to_dict()
    data = {}
    types = df_dict['Type']
    del(df_dict['Type'])

    gain = None
    for i in range(len(types)):
        for k, vals in df_dict.items():
            if append_name:
                key_name = f'[{types[i]}][{append_name}][{k}]'
            else:
                key_name = f'[{types[i]}][{k}]'
            
            data[process_key(key_name)] = floatify(vals[i])
    
    return data

def get_scrip_data(scrip):
    print(f'********** {scrip}')
    URL = f'https://www.screener.in/company/{scrip}/'

    if scrip not in CONSOLIDATED_NOT_AVAILABLE_ON_SCREENER:
        URL += 'consolidated/'

    hdr = {'User-Agent':'Mozilla/5.0'}
    req = Request(URL, headers=hdr)
    page=urlopen(req)
    soup = BeautifulSoup(page)
    data = {'Scrip': scrip} 
    data.update(extract_scrip_ratios(soup,'company-ratios', 'top-ratios'))
    data.update(flatten_df(extract_table_by_class(soup, 'profit-loss', 'data-table responsive-text-nowrap')))
    data.update(flatten_df(extract_table_by_class(soup, 'quarters', 'data-table responsive-text-nowrap'), append_name='Qtr'))
    data.update(extract_compound_tables(soup, 'profit-loss', 'ranges-table'))
    data.update(flatten_df(extract_table_by_class(soup, 'ratios', 'data-table responsive-text-nowrap')))
    data.update(flatten_df(extract_table_by_class(soup, 'shareholding', 'data-table')))
    
    return data

def get_screener_data(scrips=[]):
    scrips = get_holding_quantities().keys() if not scrips else scrips
    
    data_list = []
    for scrip in scrips:
        data_list.append(get_scrip_data(scrip))
        sleep(1)

    return pd.DataFrame(data_list).set_index('Scrip')


# print(get_scrip_data('BAJFINANCE'))

# URL = 'https://www.screener.in/company/DMART/consolidated/'
# hdr = {'User-Agent':'Mozilla/5.0'}
# req = Request(URL,headers=hdr)
# page=urlopen(req)
# soup = BeautifulSoup(page)
# shareholders = extract_table_by_class(soup, 'shareholding', 'data-table')
# ratios = extract_table_by_class(soup, 'ratios', 'data-table responsive-text-nowrap')

# profit_loss_df = extract_table_by_class(soup, 'profit-loss', 'data-table responsive-text-nowrap')
# quarters_df = extract_table_by_class(soup, 'quarters', 'data-table responsive-text-nowrap')
# box = extract_compound_tables(soup, 'profit-loss', 'ranges-table')

# print(profit_loss_df)
# print(flatten_df(quarters_df))
# print(box)
# print(flatten_df(ratios))
# print(flatten_df(shareholders))

