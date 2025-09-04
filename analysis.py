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
def fit_predict_eval(model, features_train, features_test, target_train, target_test, scores_dict):
    # here you fit the model
    model.fit(features_train, target_train)

    # make a prediction
    target_pred = model.predict(features_test)
    # The next line is not used, so it's commented out to clean the code.
    # target_proba = model.predict_proba(features_test)

    # calculate accuracy and save it to score
    score = precision_score(target_test, target_pred, average='macro', zero_division=0)

    # Print the message
    print(f'Model: {model}\nAccuracy: {round(score, 4)}\n')

    # Store the score in the dictionary
    scores_dict[str(model.__class__.__name__)] = score


def main():
    # Dictionary of models
    models = [
        KNeighborsClassifier(),
        DecisionTreeClassifier(random_state=40),
        LogisticRegression(random_state=40, max_iter=1000),
        RandomForestClassifier(random_state=40)
    ]

    # Evaluate each model with normalized data and store scores
    # Dictionaries for accuracy scores of non-normalized and normalized data
    scores, scores_norm = {}, {}

    print("Non-normalized data scores:\n")
    for model in models:
        fit_predict_eval(model, x_train, x_test, y_train, y_test, scores)

    print("Normalized data scores:\n")
    for model in models:
        fit_predict_eval(model, x_train_norm, x_test_norm, y_train, y_test, scores_norm)

    # Check whether normalization have a net positive impact on the accuracy scores
    comparison = 0
    for score, score_norm in scores, scores_norm:
        comparison += score_norm-score
        
    # Final message - Q1: Does the normalization have a positive impact in general? (yes/no)
    if comparison > 0:
        print(f"\nThe answer to the 1st question: yes")
    else:
        print(f"\nThe answer to the 1st question: no")

    # Q2: Which two models show the best scores? Round the result to the third decimal
    #     place and print the accuracy of models in descending order.

    # Sort the normalized scores dictionary in descending order
    sorted_scores_norm = sorted(scores_norm.items(), key=lambda item: item[1], reverse=True)
    print(f"The answer to the 2nd question: {sorted_scores_norm[0][0]}-{round(sorted_scores_norm[0][1], 3)}, {sorted_scores_norm[1][0]}-{round(sorted_scores_norm[1][1], 3)}")


if __name__ == "__main__":
    main()
