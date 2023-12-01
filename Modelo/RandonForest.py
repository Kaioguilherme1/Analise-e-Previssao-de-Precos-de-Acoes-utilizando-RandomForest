from sklearn.ensemble import RandomForestRegressor

from Data import features_train, labels_train, features_test
from avaliar import evaluate, plot_learning_curve

hyperparameter = [3, 4, 29, 14, 21]
# Definir o modelo
# model = RandomForestRegressor(n_estimators=hyperparameter[0] + 1,
#                               max_depth=hyperparameter[1] + 1,
#                               min_samples_split=hyperparameter[2] + 2,
#                               min_samples_leaf=hyperparameter[3] + 1,
#                               max_features=hyperparameter[4] + 1,
#                               random_state=42)

model = RandomForestRegressor()
model.fit(features_train, labels_train)
predictions = model.predict(features_test)

print(evaluate(predictions, 'Random Forest 2Y - saida unica | sem hyperparametro').head())
plot_learning_curve(model, 'Random Forest 2Y - saida unica | sem hyperparametro')
