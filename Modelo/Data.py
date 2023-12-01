import pandas as pd
from sklearn.model_selection import train_test_split

# Configuração para exibir até 10 colunas no DataFrame
pd.set_option('display.max_columns', 10)

# Carregar o conjunto de dados
dataset = pd.read_csv('../Dataset_b3/data_view_3y.csv')

# Filtrar linhas com cota superior a 500
selected_rows = dataset[(dataset.iloc[:, 22:] > 500).any(axis=1)]
dataset = dataset.drop(selected_rows.index, axis=0)

# Selecionar os preços das ações
prices = dataset.iloc[:, 22] # o valor da ações no dia 2023-11-01

# Remover colunas de informações das ações (primeiros 21 índices representam informações das ações)
dataset = dataset.drop(dataset.columns[0:292], axis=1) # o historico de valores a partir de 2022-09-30 a 2020-04-07

print(dataset.head())

# Exibir informações sobre o conjunto de dados
print('Número de ações: ', len(dataset))
print('Número de dias: ', len(dataset.columns))

# Separar labels (preços) e features (dados)
labels = prices
features = dataset

# Dividir os dados em conjuntos de treinamento e teste
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25, random_state=50)





