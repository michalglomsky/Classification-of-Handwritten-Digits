import tensorflow
import numpy as np
import keras

# Load the data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Change shape of the features array to 2D-array of rows for images and columns for pixels
x_train = np.reshape(x_train,newshape=(60000, 784))

def main():
    # Print overview of the data
    print(f"Classes: {np.unique(y_train)}")
    print(f"Features' shape: {x_train.shape}")
    print(f"Target's shape: {y_train.shape}")
    print(f"min: {np.min(x_train)}, max: {np.max(x_train)}")

if __name__=="__main__":
    main()