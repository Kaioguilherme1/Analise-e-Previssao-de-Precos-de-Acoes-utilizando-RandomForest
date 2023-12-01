from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import r2_score
from Genetics import Genetic
from Data import features_train, labels_train, features_test, labels_test

def fiteness(chromosome: list):

    model = RandomForestRegressor(n_estimators=chromosome[0] + 1,
                                    max_depth=chromosome[1] + 1,
                                    min_samples_split=chromosome[2] + 2,
                                    min_samples_leaf=chromosome[3] + 1,
                                    max_features=chromosome[4] + 1,
                                    random_state=42)
    model.fit(features_train, labels_train)
    predictions = model.predict(features_test)
    return r2_score(labels_test, predictions)

population = Genetic(chromosome_size=5,
                     population_size=100,
                     genes_number=30,
                     generations=50,
                     fitness_function=fiteness,
                     mutation_prob=0.025,
                     best=0.5,
                     num_elites=4,
                     selection_prob=0.001,)

results, performance = population.run()  # executa o algoritmo genetico
population.print_performace(performance)  # imprime o desempenho do algoritmo

# imprime o melhor resultado
print("Melhor Resultado: ")
print("Número de gerações: ", len(results))
print("Chromosome: ", results[-1][0])
print("Fitness: ", results[-1][1])

# imprime o gráfico
population.plot_graphic(results, "Randon forest Regressor")


