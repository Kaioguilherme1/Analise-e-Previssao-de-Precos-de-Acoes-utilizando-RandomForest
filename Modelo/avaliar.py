import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from Data import features, labels, features_test, labels_train, labels_test

# Função para avaliar o modelo
from sklearn.metrics import mean_absolute_error, explained_variance_score

def evaluate(predictions, Title):
    """
    Avalia o desempenho do modelo usando métricas como R², MSE, MAE, e Explained Variance Score.
    Também exibe gráficos de dispersão e curva de aprendizado.

    Parâmetros:
    - predictions: array-like
        As previsões feitas pelo modelo.
    - Title: str
        Título usado nos gráficos.

    Retorna:
    - results_df: DataFrame
        DataFrame contendo os valores reais, previstos e a primeira coluna de features dos dados de teste.
    """
    # Calcular métricas de avaliação
    r2 = r2_score(labels_test, predictions)
    mse = mean_squared_error(labels_test, predictions)
    mae = mean_absolute_error(labels_test, predictions)
    evs = explained_variance_score(labels_test, predictions)
    num_test = len(predictions)
    num_train = len(labels_train)

    mse_quadratico = np.sqrt(mse)

    num_errors = sum(predictions != labels_test)
    total_items_tested = len(labels_test)

    print(f"R² no conjunto de teste: {r2:.4f}")
    print(f"Raiz do Erro médio quadrático no conjunto de teste: {mse_quadratico:.4f}")
    print(f"Erro médio absoluto no conjunto de teste: {mae:.4f}")
    print(f"Explained Variance Score: {evs:.4f}")
    print(f"Número de erros no conjunto de teste: {num_errors}")
    print(f"Quantidade total de itens testados: {total_items_tested}")

    # Gráficos
    df = pd.DataFrame({'Real': labels_test, 'Predicted': predictions})

    # Calcular a diferença entre os valores reais e previstos
    difference = df['Real'] - df['Predicted']
    average_difference = np.mean(difference)

    # Calcular o erro padrão da diferença
    std_difference = np.std(difference)

    # Gráfico de dispersão valor real x previsão com variação de erro
    plt.figure(figsize=(8, 6))
    plt.scatter(df['Real'], df['Predicted'], label='Valor Real vs. Previsão', alpha=0.5, color='b')
    plt.plot(df['Real'], df['Real'], linestyle='--', color='r', label='Linha de Igualdade')
    plt.title('Gráfico Valor Real vs. Previsão: ' + Title)
    plt.xlabel('Valor Real')
    plt.ylabel('Previsão')

    # Desenhar linhas de variação de erro
    plt.fill_between(df['Real'], df['Real'] - std_difference, df['Real'] + std_difference, alpha=0.2, color='gray', label='Variação de Erro')

    # Exibir média da diferença no gráfico
    textstr = f'Média da Diferença: {average_difference:.4f}\nR²: {r2:.4f}\nMSE: {mse:.4f}\nMAE: {mae:.4f} \nTrain: {num_train} \nTests: {num_test}'
    plt.text(0.05, 0.85, textstr, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black', alpha=0.7))

    # Adicionar legenda
    plt.legend(loc='upper left')

    plt.show()

    # Criar um DataFrame com os valores reais e previstos
    results_df = pd.DataFrame({'Real': labels_test, 'Predicted': predictions, features_test.columns[0]: features_test.iloc[:, 0]})

    return results_df

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
    plt.grid()

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
    plt.plot(train_sizes, test_scores_mean_r2, 'o-', color='g', label='Avaliação')

    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()
