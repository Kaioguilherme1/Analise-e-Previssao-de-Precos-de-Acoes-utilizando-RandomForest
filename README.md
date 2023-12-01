# Análise e Previsão de Preços de Ações

Este repositório contém uma análise de preços de ações utilizando o modelo RandomForest. Otimizado com algoritmos genéticos, o modelo é avaliado para previsões de saídas únicas e múltiplas em um horizonte de 12 meses.

## Conteúdo

- `Data/`: Conjunto de dados utilizado.
- `Dataset_b3/`: Conjunto de dados da Bolsa de Valores B3.
- `Dataset_scripts/`: Scripts para coleta e preparação de dados.
- `LICENSE`: Licença do projeto.
- `Modelo/`: Código-fonte e scripts do modelo.
- `README.md`: Documentação e instruções de uso.
- `Resultados/`: Resultados das previsões.

## Como Usar

1. **Clone o repositório.**

```bash
git clone https://github.com/Kaioguilherme1/Analise-e-Previssao-de-Precos-de-Acoes-utilizando-RandomForest.git
```

2. **Navegue até o diretório do modelo.**

```bash
cd Analise-e-Previsao-de-Precos-de-Acoes-utilizando-RandomForest/Modelo
```

3. **Execute o algoritmo genético para otimização dos hiperparâmetros.**

```bash
python3 hyperparametro.py
```

4. **Execute o RandomForest para previsões de saída única.**

```bash
python3 RandomForest.py
```

5. **Execute o historiograma para previsões de saídas múltiplas.**

```bash
python3 historiograma.py
```

6. **Explore os resultados na pasta `Resultados/` para análise e interpretação.**

## Requisitos

- Python 3.x
- Bibliotecas Python: numpy, pandas, seaborn, matplotlib, scikit-learn
```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```

## [Resultados]()


## ✒️ Autores

* **developer** - *Initial Work* - [Kaio Guilherme](https://github.com/Kaioguilherme1)

## 📑 Licença

Esse projeto esta sob a licença(MIT) - veja o arquivo [Licenca.md](https://github.com/Kaioguilherme1/Analise-e-Previssao-de-Precos-de-Acoes-utilizando-RandomForest/blob/main/LICENSE) para mais detalhes.
