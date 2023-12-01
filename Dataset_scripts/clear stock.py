import pandas as pd
from tqdm import tqdm
from yahooquery import search

error = 0

def lister_stock(stock):
    global error
    result = search(query=stock, country='brazil', quotes_count=1)
    try:
        if result['count'] != 0:
            return result['quotes'][0].get('symbol', False)
        else:
            return False
    except Exception as e:
        print(f'Erro: {e} - {stock} / {error}')
        error += 1
        return False

def validade(stocks, filename, amount=None):
    df = pd.read_csv(stocks)
    remove = []
    result = []

    if amount is not None:
        df = df.sample(n=amount, random_state=1)

    for stock in tqdm(df['ticker'].values):
        stock_info = lister_stock(stock)

        if not stock_info:
            remove.append(stock)
        else:
            result.append(stock_info)

    if result:
        result_df = pd.DataFrame(result, columns=['ticker'])
        result_df.to_csv(filename, mode='w', index=False, header=False)

    if remove:
        remove_df = pd.DataFrame({'ticker': remove})
        remove_df.to_csv('remove.csv', mode='a', index=False, header=False)

validade('Data/stocks_list.csv', 'Dataset/Tickers.csv')
