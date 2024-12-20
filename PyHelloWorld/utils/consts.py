OUT_DIR = '/Users/imran/Documents/apps/docs/outputs/'
HOLDING_FILE = '~/Documents/apps/docs/Holdings_20Dec24.xlsx'
### TODO: Get this data from holdings.googlesheet directly

NSE_BSE = {
            'BAJFINANCE': 'BAJFINANCE.BO',
            'DMART': 'DMART.NS',
            'DEEPAKNTR': 'DEEPAKNTR.BO',
            'SONACOMS': 'SONACOMS.BO',
            'POLYCAB': 'POLYCAB.BO',
            'FINEORG': 'FINEORG.BO',
            'KPITTECH': 'KPITTECH.BO',
            'DIVISLAB': 'DIVISLAB.BO',
            'NAUKRI': 'NAUKRI.BO',
            'AFFLE': 'AFFLE.BO',
            'HAPPSTMNDS': 'HAPPSTMNDS.BO',
            'TATACHEM': 'TATACHEM.BO',
            'RAJRATAN': 'RAJRATAN.BO',
            'PRINCEPIPE': 'PRINCEPIPE.BO',
            'IONEXCHANG': 'IONEXCHANG.BO',
            'BERGEPAINT': 'BERGEPAINT.BO',
            'RELAXO': 'RELAXO.BO',
            'DIXON': 'DIXON.BO',
            'PRAJIND': 'PRAJIND.BO',
            'LAURUSLABS': 'LAURUSLABS.BO',
            'BEL': 'BEL.BO',
            'DEVYANI': 'DEVYANI.BO',
            'GARFIBRES': 'GARFIBRES.BO',
            'MOLDTKPAC': 'MOLDTEK.BO',
            'RBA': 'RBA.BO'
}

# BSE_NSE = {}
# for k, v in zip(NSE_BSE.values(), NSE_BSE.keys()):
#     BSE_NSE[k] = v

# QTYS = {
#             'BAJFINANCE': 43,
#             'DMART': 79,
#             'DEEPAKNTR': 132,
#             'SONACOMS': 457,
#             'POLYCAB': 48,
#             'FINEORG': 52,
#             'KPITTECH': 143,
#             'DIVISLAB': 55,
#             'NAUKRI': 45,
#             'AFFLE': 169,
#             'HAPPSTMNDS': 225,
#             'TATACHEM': 183,
#             'RAJRATAN': 231,
#             'PRINCEPIPE': 213,
#             'IONEXCHANG': 243,
#             'BERGEPAINT': 243,
#             'RELAXO': 128,
#             'DIXON': 20,
#             'PRAJIND': 167,
#             'LAURUSLABS': 224,
#             'NEOGEN': 55,
#             'BEL': 570,
#             'BORORENEW': 135,
#             'DEVYANI': 281,
#             'GARFIBRES': 13,
#             'MOLDTKPAC': 36,
#             'RBA': 245,
# }

BASE_URL = 'https://www.screener.in/company/'

# PROFIT-LOSS STRATEGY - CONFIG
PROFIT_LOSS_YEARS = 3
SCREENER_ROW_PL = 'Net Profit' # or can be 'Profit before tax'
CURRENT_HIGH_THRESHOLD_PERCENT = 10

CONSOLIDATED_NOT_AVAILABLE_ON_SCREENER = ['PRINCEPIPE', 'RELAXO']
SCREENER_TOP_DATA = ['Market Cap', 'Current Price', 'High / Low', 'Stock P/E', 'Book Value',
                     'Dividend Yield', 'ROCE', 'ROE', 'Face Value', 'ROIC', 'PEG Ratio',
                     'Debt to equity', 'Financial leverage', 'EVEBITDA', 'Pledged percentage',
                     'CMP / FCF']

NBFC_KEY_MAP = {
                    'Revenue': 'Sales',
                    'Financing Profit': 'Operating Profit',
                    'Financing Margin': 'OPM',
                    
               } 
