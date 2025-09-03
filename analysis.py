import tensorflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, f1_score
import keras


# Load the data
x_train, y_train = keras.datasets.mnist.load_data(path='mnist.npz')[0]

# Reshape the datasets
x_train = np.reshape(x_train,shape=(60000, 784))

# Use only the first 6000 rows
x_train = x_train[:6000]
y_train = y_train[:6000]

# Train, test, split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=40)


# Function for fitting, predicting and evaluating a model
def fit_predict_eval(model, features_train, features_test, target_train, target_test, scores_dict):

    # here you fit the model
    model.fit(features_train, target_train)

    # make a prediction
    target_pred = model.predict(features_test)
    target_proba = model.predict_proba(features_test)

    # calculate accuracy and save it to score
    score = precision_score(target_test, target_pred, average='macro')

    # Print the message
    print(f'Model: {model}\nAccuracy: {round(score, 4)}\n')

    # Store the score in the dictionary
    scores_dict[str(model.__class__.__name__)] = score


def main():
    """# Print overview of the data
    print(f"x_train shape: {x_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print("Proportion of samples per class in train set:")
    print(pd.Series(y_train).value_counts(normalize=True))"""

    # Create a dictionary to hold the scores for each model
    scores = {}

    # K-nearest Neighbors
    fit_predict_eval(
        model=KNeighborsClassifier(),
        features_train=x_train,
        features_test=x_test,
        target_train=y_train,
        target_test=y_test,
        scores_dict=scores
    )

    # Decision Tree
    fit_predict_eval(
        model=DecisionTreeClassifier(random_state=40),
        features_train=x_train,
        features_test=x_test,
        target_train=y_train,
        target_test=y_test,
        scores_dict=scores
    )

    # Logistic Regression
    fit_predict_eval(
        model=LogisticRegression(),
        features_train=x_train,
        features_test=x_test,
        target_train=y_train,
        target_test=y_test,
        scores_dict=scores
    )

    # Random Forest
    fit_predict_eval(
        model=RandomForestClassifier(random_state=40),
        features_train=x_train,
        features_test=x_test,
        target_train=y_train,
        target_test=y_test,
        scores_dict=scores
    )

    # Find the maximum score
    max_score = max(scores.values())

    # Find the model with the maximum score
    best_model = max(scores, key=scores.get)

    # Final message
    print(f"The answer to the question: {best_model} - {round(max_score, 3)}")


if __name__=="__main__":
    main()
