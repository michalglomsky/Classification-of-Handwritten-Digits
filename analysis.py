import tensorflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import keras

# Load the data
x_train, y_train = keras.datasets.mnist.load_data(path='mnist.npz')[0]

# Reshape the datasets
x_train = np.reshape(x_train,newshape=(60000, 784))

# Use only the first 6000 rows
x_train = x_train[:6000]
y_train = y_train[:6000]

# Train, test, split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=42)

def main():
    # Print overview of the data
    print(f"x_train shape: {x_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print("Proportion of samples per class in train set:")
    print(pd.Series(y_train).value_counts(normalize=True))

if __name__=="__main__":
    main()
