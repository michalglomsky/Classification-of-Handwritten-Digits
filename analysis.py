import tensorflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, f1_score
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV
import keras

# Load the data
x_train, y_train = keras.datasets.mnist.load_data(path='mnist.npz')[0]

# Reshape the datasets
x_train = np.reshape(x_train, shape=(60000, 784))

# Use only the first 6000 rows
x_train = x_train[:6000]
y_train = y_train[:6000]

# Train, test, split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=40)

# Normalize the data
normalizer = Normalizer()
x_train_norm = normalizer.fit_transform(x_train)
x_test_norm = normalizer.fit_transform(x_test)


# Function for fitting, predicting and evaluating a model
def fit_predict_eval(model, features_train, features_test, target_train, target_test):
    # here you fit the model
    model.fit(features_train, target_train)

    # Make a prediction
    target_pred = model.predict(features_test)

    # Calculate accuracy and save it to score
    score = precision_score(target_test, target_pred, average='macro', zero_division=0)

    # Store the score in the dictionary
    print(f'best estimator: {model}')
    print(f'accuracy: {score}\n')

def main():

    # Hyperparameters for each model
    KNN_hyperparameters = {'n_neighbors': [3, 4], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'brute']}
    RForest_hyperparameters = {'n_estimators': [300, 500], 'max_features': ['sqrt', 'log2'], 'class_weight': ['balanced', 'balanced_subsample']}

    # Grid search for each model
    grid_search_KNN = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=KNN_hyperparameters, scoring='accuracy', n_jobs=-1)
    grid_search_RForest = GridSearchCV(estimator=RandomForestClassifier(random_state=40), param_grid=RForest_hyperparameters, scoring='accuracy', n_jobs=-1)

    # Dictionary of models for cross-validation
    models = {'K-nearest neighbours algorithm': grid_search_KNN,
        'Random forest algorithm': grid_search_RForest}
    
    # Running through models on normalized data
    for model in models.items():
        print(f'{model[0]}')
        model[1].fit(x_train_norm, y_train)
        best_model = model[1].best_estimator_
        fit_predict_eval(best_model, x_train_norm, x_test_norm, y_train, y_test)

if __name__ == "__main__":
    main()
