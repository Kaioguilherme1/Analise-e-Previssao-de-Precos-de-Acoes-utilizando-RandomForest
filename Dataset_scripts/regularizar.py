
import pandas as pd

# Carregar data
data = pd.read_csv('data_view.csv', low_memory=False).drop_duplicates()
print(data.shape)

col_index = data.columns.get_loc('2020-04-06')

data = data.iloc[:, :col_index]

columns_to_check = data.iloc[:, 21:]
# Verificar se as colunas estão com valor NaN ou {} e salva o index
index_to_drop = []
for index, row in columns_to_check.iterrows():
    if row.isnull().any() or row.isin(['{}']).any():
        index_to_drop.append(index)

data = data.drop(index_to_drop)

print(data)
data.to_csv('data_view_3y.csv', index=False)


