import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score, make_scorer
from sklearn.model_selection import train_test_split, learning_curve

# limpar o dataset
dataset = pd.read_csv('../Dataset_b3/data_view_3y.csv')

# Filtrar linhas com cota superior a 500
selected_rows = dataset[(dataset.iloc[:, 22:] > 500).any(axis=1)]
dataset = dataset.drop(selected_rows.index, axis=0)



# limpar o dataset para o treinamento limitando a data de 2022-11-01
features = dataset.drop(dataset.columns[0:292], axis=1)


# selecionar os meses para avaliação através do indice
meses = [43, 63, 86, 107, 128, 150, 168, 190, 208, 230, 251, 271]

hyperparameter = [3, 4, 29, 14, 21]
predicoes = []
real = []

# Definir o modelo
# model = RandomForestRegressor(n_estimators=hyperparameter[0] + 1,
#                               max_depth=hyperparameter[1] + 1,
#                               min_samples_split=hyperparameter[2] + 2,
#                               min_samples_leaf=hyperparameter[3] + 1,
#                               max_features=hyperparameter[4] + 1,
#                               random_state=42)

model = RandomForestRegressor()

labels = pd.DataFrame(dataset.iloc[:, meses]) #a cada mes
# days = list(range(43, 271, 10)) # a cada 10 dias
# labels = pd.DataFrame(dataset.iloc[:, days]) # o valor da ações no dia 2023-11-01

# separa os dados de teste e de avaliação
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25, random_state=50)

# treina o modelo
model.fit(features_train, labels_train)
predictions = model.predict(features_test)

# transforma os dados em dataframes
predictions_data = pd.DataFrame(predictions).iloc[:, ::-1]
real_data = pd.DataFrame(labels_test).iloc[:, ::-1]

r2_values = []

# Calcular métricas de avaliação
for i in range(predictions_data.shape[1]):
    r2 = r2_score(real_data.iloc[:, i], predictions_data.iloc[:, i])
    r2_values.append(r2)
    r2_values = sorted(r2_values)

# Encontrar o índice do valor máximo (melhor precisão)
index_max_precision = r2_values.index(max(r2_values))

index_med_precision = len(r2_values) // 2

# Encontrar o índice do valor mínimo (pior precisão)
index_min_precision = r2_values.index(min(r2_values))

Data = labels_test.columns.tolist()[::-1]

r2 =  r2_score(labels_test, predictions)
mse = mean_squared_error(labels_test, predictions)
mae = mean_absolute_error(labels_test, predictions)
evs = explained_variance_score(labels_test, predictions)
num_test = len(predictions)
num_train = len(labels_train)
mse_quadratico = np.sqrt(mse)

# Plotar o gráfico
# Configurar o estilo do seaborn para melhorar a estética
sns.set(style="whitegrid")

# mean relative erro 
def std_percent_diff(real_values, predicted_values):
    # Converte as listas para arrays numpy
    real_values = np.array(real_values)
    predicted_values = np.array(predicted_values)

    # Calcule a diferença absoluta entre os valores reais e previstos
    absolute_differences = np.abs(real_values - predicted_values)

    # Calcule a diferença percentual
    percent_differences = (absolute_differences / np.abs(real_values)) * 100

    # Calcule o desvio padrão da diferença percentual
    std_percent_difference = np.std(percent_differences)

    return std_percent_difference

# mean relative erro
max_standard_deviation = std_percent_diff(real_data.iloc[index_max_precision], predictions_data.iloc[index_max_precision])
med_standard_deviation = std_percent_diff(real_data.iloc[index_med_precision], predictions_data.iloc[index_med_precision])
min_standard_deviation = std_percent_diff(real_data.iloc[index_min_precision], predictions_data.iloc[index_min_precision])

# Tamanho da figura
plt.figure(figsize=(15, 12))

# Gráfico para a melhor precisão
plt.subplot(3, 1, 1)  # 3 linhas, 1 coluna, primeiro gráfico
plt.plot(Data, real_data.iloc[index_max_precision], label='Real', marker='o', color='blue')
plt.plot(Data, predictions_data.iloc[index_max_precision], label='Melhor Previsão', marker='o', color='green')
plt.title(f'Gráfico para a Melhor Precisão (R²={max(r2_values):.4f} | Desvio padrão: {max_standard_deviation:.4f}%', fontsize=16, color='blue')
plt.xlabel('Data', fontsize=14)
plt.ylabel('Valor', fontsize=14)
textstr = f"R²: {r2:.4f}\nEVS: {evs:.4f}\nMSE: {mse:.4f}\nRaiz MSE: {mse_quadratico:.4f}\nMAE: {mae:.4f} \nTrain: {num_train} \nTests: {num_test}"
plt.text(0.05, 0.85, textstr, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black', alpha=0.7))
plt.legend()
plt.grid(True)



# Grafico para a precisão média
plt.subplot(3, 1, 2)  # 3 linhas, 1 coluna, segundo gráfico
plt.plot(Data, real_data.iloc[index_med_precision], label='Real', marker='o', color='blue')
plt.plot(Data, predictions_data.iloc[index_med_precision], label='Previsão Média', marker='o', color='orange')
plt.title(f'Gráfico para a Precisão Média (R²={np.mean(r2_values):.4f} | Desvio padrão: {med_standard_deviation:.4f}%)', fontsize=16, color='orange')
plt.xlabel('Data', fontsize=14)
plt.ylabel('Valor', fontsize=14)
plt.legend()
plt.grid(True)

# Gráfico para a pior precisão
plt.subplot(3, 1, 3)  # 3 linhas, 1 coluna, terceiro gráfico
plt.plot(Data, real_data.iloc[index_min_precision], label='Real', marker='o', color='blue')
plt.plot(Data, predictions_data.iloc[index_min_precision], label='Pior Previsão', marker='o', color='red')
plt.title(f'Gráfico para a Pior Precisão (R²={min(r2_values):.4f} | Desvio padrão: {min_standard_deviation:.4f}%)', fontsize=16, color='red')
plt.xlabel('Data', fontsize=14)
plt.ylabel('Valor', fontsize=14)
plt.legend()
plt.grid(True)

plt.tight_layout()  # Ajusta automaticamente a disposição dos gráficos para evitar sobreposição
plt.show()


# imprime as estatisticas do modelo
print(f"R² no conjunto de teste: {r2:.4f}")
print(f"Raiz do Erro médio quadrático no conjunto de teste: {mse_quadratico:.4f}")
print(f"Erro médio absoluto no conjunto de teste: {mae:.4f}")
print(f"Explained Variance Score: {evs:.4f}")

def plot_learning_curve(model, Title, cv=5):
    """
    Plota a curva de aprendizado para o modelo, exibindo as pontuações de MSE e R² em conjuntos de treinamento e teste.

    Parâmetros:
    - model: objeto do modelo
        O modelo a ser avaliado.
    - Title: str
        Título usado nos gráficos.
    - cv: int, opcional
        Número de dobras para a validação cruzada. Padrão é 5.
    """
    # Crie um gráfico de aprendizado para o MSE
    train_sizes, train_scores_mse, test_scores_mse = learning_curve(
        model, features, labels, cv=cv, scoring='neg_mean_squared_error')

    # Calcule as médias e desvios padrão das pontuações do MSE
    train_scores_mean_mse = -train_scores_mse.mean(axis=1)
    train_scores_std_mse = train_scores_mse.std(axis=1)
    test_scores_mean_mse = -test_scores_mse.mean(axis=1)
    test_scores_std_mse = test_scores_mse.std(axis=1)

    # Plote o gráfico de aprendizado para o MSE
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.title('Curva de Aprendizado (MSE): ' + Title)
    plt.xlabel('Tamanho do Conjunto de Treinamento')
    plt.ylabel('MSE (Erro Quadrático Médio)')
    plt.grid(True)

    plt.fill_between(train_sizes, train_scores_mean_mse - train_scores_std_mse, train_scores_mean_mse + train_scores_std_mse, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_scores_mean_mse - test_scores_std_mse, test_scores_mean_mse + test_scores_std_mse, alpha=0.1, color='g')

    plt.plot(train_sizes, train_scores_mean_mse, 'o-', color='r', label='Treinamento')
    plt.plot(train_sizes, test_scores_mean_mse, 'o-', color='g', label='Avaliação')

    plt.legend(loc='best')

    # Crie um gráfico de aprendizado para o R²
    train_sizes, train_scores_r2, test_scores_r2 = learning_curve(
        model, features, labels, cv=cv, scoring=make_scorer(r2_score))

    # Calcule as médias e desvios padrão das pontuações do R²
    train_scores_mean_r2 = train_scores_r2.mean(axis=1)
    train_scores_std_r2 = train_scores_r2.std(axis=1)
    test_scores_mean_r2 = test_scores_r2.mean(axis=1)
    test_scores_std_r2 = test_scores_r2.std(axis=1)

    # Plote o gráfico de aprendizado para o R²
    plt.subplot(1, 2, 2)
    plt.title('Curva de Aprendizado (R²): ' + Title)
    plt.xlabel('Tamanho do Conjunto de Treinamento')
    plt.ylabel('R²')
    plt.grid(True)

    plt.fill_between(train_sizes, train_scores_mean_r2 - train_scores_std_r2, train_scores_mean_r2 + train_scores_std_r2, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_scores_mean_r2 - test_scores_std_r2, test_scores_mean_r2 + test_scores_std_r2, alpha=0.1, color='g')

    plt.plot(train_sizes, train_scores_mean_r2, 'o-', color='r', label='Treinamento')
    plt.plot(train_sizes, test_scores_mean_r2, 'o-', color='g', label='Availiação')

    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


plot_learning_curve(model, 'Random Forest 2Y - saida multipla | sem hyperparametro')
