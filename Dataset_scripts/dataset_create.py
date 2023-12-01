import pandas as pd
from tqdm import tqdm
from yahooquery import Ticker
from datetime import datetime, timedelta
import numpy as np

import csv


class Stocks:
    def __init__(self, stock_list: list, dataset_name: str, period: int = 2, amount: int = None, regularize: bool = False):
        # Seu código de inicialização aqui
        self.stock_list = stock_list
        self.regularize = regularize
        self.dataset = dataset_name
        self.period = period
        self.amount = amount
        self.data = None
        self.data_view = None
        self.price_history = None
        self.end = datetime.now().strftime("%Y-%m-%d")
        self.start = (datetime.now() - timedelta(days=(365 * self.period))).strftime("%Y-%m-%d")
        

    def _is_float(self, value):
        return isinstance(value, float)

    def get_price_history(self, price_history):
        stock_history = price_history.copy()
        # transforma o dataframe em uma tabela
        stock_history = stock_history['adjclose'].reset_index().drop(columns=['symbol']).iloc[::-1]
        header = stock_history.iloc[0:, 0].to_list()
        header = [ date.strftime('%Y-%m-%d') for date in header]
    
        data = stock_history.iloc[0:, 1].to_list()
        stock_history = pd.DataFrame([data], columns=header)

        date = datetime.now()

        # pega a mesma data 1 ano no passado 
        date_pass = date - pd.DateOffset(years=1)
        # pega a coluna da data 1 ano no passado e se n achar avanca1 dia ate achar    
        while date_pass.date().strftime('%Y-%m-%d') not in stock_history.columns:
            date_pass = date_pass + pd.DateOffset(days=1)
            
        col = stock_history.columns.get_loc(date_pass.date().strftime('%Y-%m-%d'))

        #calcula a valorizacao em um ano
        valozacao_1_ano = (stock_history.iloc[0, 0] - stock_history.iloc[0, col]) / stock_history.iloc[0, col]

        #valorizacao total em n anos
        valorizacao_total = (stock_history.iloc[0, 0] - stock_history.iloc[0,-1]) / stock_history.iloc[0 ,-1]

        history = stock_history

        # adiciona estatisticas
        # preco medio
        
        stock_history.insert(0, 'price max', history.max(axis=1))
        stock_history.insert(0, 'price med', history.mean(axis=1))
        stock_history.insert(0, 'price min', history.min(axis=1))
        stock_history.insert(0, 'valuation ', valorizacao_total)
        stock_history.insert(0, 'valuation 12M', valozacao_1_ano)
        
        return  stock_history
            
    def get_stock(self, stock):
        data = []
        price_history = None
        simbol = stock.symbols[0]
        
        try:
            # caso que não existe o summary_profile
            sector = stock.summary_profile[simbol].get('sector', 'N/A')
            industry = stock.summary_profile[simbol].get('industry', 'N/A')
        except:
            sector = 'N/A'
            industry = 'N/A'
        try:
            date_time = datetime.strptime(stock.price[simbol]['regularMarketTime'], '%Y-%m-%d %H:%M:%S')
            date = int(date_time.strftime('%Y%m%d'))
            open = stock.summary_detail[simbol].get('open', 'N/A')
            price = stock.price[simbol].get('regularMarketPrice', 'N/A')
            volume = stock.summary_detail[simbol].get('volume', 'N/A')
            dividend = stock.summary_detail[simbol].get('dividendRate', 'N/A')
            dividend = dividend if dividend != {} else 0
            dy = stock.summary_detail[simbol].get('dividendYield', 'N/A')
            dy = dy if dy != {} else 0
            pl = stock.summary_detail[simbol].get('trailingPE', 'N/A')
        except:
            date = 'N/A'
            open = 'N/A'
            price = 'N/A'
            volume = 'N/A'
            dividend = 'N/A'
            dy = 'N/A'
            pl = 'N/A'
        try:
            # caso que não existe o key_stats
            pvp = stock.key_stats[simbol].get('priceToBook', 'N/A')
            lpa = stock.key_stats[simbol].get('trailingEps', 'N/A')
            vpa = stock.key_stats[simbol].get('bookValue', 'N/A')
            graham = (22.5 * lpa * vpa * 1.1) ** (1 / 2) if self._is_float(lpa) and self._is_float(vpa) and (lpa > 0 and vpa > 0) else 'N/A'
            enterprise_value = stock.key_stats[simbol].get('enterpriseValue', 'N/A')
        except:
            pvp = 'N/A'
            lpa = 'N/A'
            vpa = 'N/A'
            graham = 'N/A'
            enterprise_value = 'N/A'
        try:
            market_cap = stock.summary_detail[simbol].get('marketCap', 'N/A')
            price_history = self.get_price_history(stock.history(start=self.start, end=self.end, interval='1d'))
        except:
            market_cap = 'N/A'
            
        data_stock = np.array([simbol, sector, industry, date, open, price, volume, dividend, dy, pl, pvp, lpa, vpa, graham, enterprise_value, market_cap])
        data_stock = np.where(data_stock == '{}', 'N/A', data_stock)
        
        data.append(data_stock)
           
        header = ["Ticker", "Sector", "Industry", "Date", "Open", "Price", "Volume", "Dividend", "Dividend Yield", "P/L", "P/VP", "LPA", "VPA", "Graham", "Enterprise Value", "Market Cap"]
        data = pd.DataFrame(data, columns=header)
        if price_history is not None:
            data = pd.concat([data, price_history], axis=1)
        return data
    
    def dataset_save(self, dataview: str = "data_view.csv"):
        
        try:
            data_view = pd.read_csv(dataview, low_memory=False)
            self.data = data_view.copy()
        except FileNotFoundError:
            print('Não existe o arquivo data_view.csv')
        
        try:
            # separa os index
            sector = pd.DataFrame(data_view['Sector'].unique(), columns=['Sector'])
            industry = pd.DataFrame(data_view['Industry'].unique(), columns=['Industry'])
            stocks = pd.DataFrame(data_view['Ticker'].unique(), columns=['Ticker'])
               
            # cria um arquivo CSV com os dados ou adicione a um arquivo existente
            sector.to_csv('Dataset_b3/sector.csv', mode='w', index=False, header=True)
            industry.to_csv('Dataset_b3/industry.csv', mode='w', index=False, header=True)
            stocks.to_csv('Dataset_b3/stocks.csv', mode='w', index=False, header=True)
    
            # troca os valores de sector, ticker e industry por index
            self.data['Sector'] = self.data['Sector'].replace(sector['Sector'].tolist(), sector.index.tolist())
            self.data['Industry'] = self.data['Industry'].replace(industry['Industry'].tolist(), industry.index.tolist())
            self.data['Ticker'] = self.data['Ticker'].replace(stocks['Ticker'].tolist(), stocks.index.tolist())
            
            # salva os dados no arquivo CSV
            self.data.to_csv(self.dataset, mode='w', index=False, header=True)
            return self.data
        
        except Exception as e:
            print('Error: ', e )
            pass
        
    
    def _regularize(self):
        print('Regularizando os dados ....')
        data = pd.read_csv('data_view.csv', low_memory=False)
        data = data.drop_duplicates()
        columns_to_check = data.iloc[:, 20:]
        # Verificar se as colunas estão com valor NaN ou {} e salva o index
        index_to_drop = []
        for index, row in columns_to_check.iterrows():
            if row.isnull().any() or row.isin(['{}']).any():
                index_to_drop.append(index)
                
        data = data.drop(index_to_drop)
        data.to_csv('data_view_regularized.csv', index=False)
        print('Dados regularizados com sucesso!')

    def dataset_view_save(self):
        print('Criando dataset ....')
        stock_list = pd.read_csv(self.stock_list, low_memory=False)
        stocks = stock_list['Ticker'].to_list()
        data_view = None
        
        try:
            data_view = pd.read_csv('data_view.csv', low_memory=False)
        except FileNotFoundError:
            print('Não existe o arquivo data_view.csv')
        
        if self.amount is not None:
            stocks = stocks[:self.amount]
        
        for ticker in tqdm(stocks, desc="Processing", ncols=100):
            stock = Ticker(ticker)
            stock = self.get_stock(stock)
            if data_view is None:
                data_view = stock
            else:
                data_view = pd.concat([data_view, stock], ignore_index=True, axis=0)
            
            data_view.to_csv('data_view.csv', mode='w', index=False, header=True)
            
            # Remove o ticker do arquivo Tickers.csv
            stock_list = stock_list[stock_list['Ticker'] != ticker]
            stock_list.to_csv(self.stock_list, mode='w', index=False, header=True)
    
        if self.regularize:
            self._regularize()
                
        print('Dataset criado com sucesso!')

teste = Stocks('Dataset_b3/Tickers.csv', 'Dataset_b3/dataset_b3_3y.csv', period=5, regularize=True)
teste.dataset_save('data_view_3y.csv')
