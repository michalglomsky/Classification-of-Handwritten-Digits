# ✍️ Handwritten Digit Classification with Hyperparameter Tuning

This project demonstrates an end-to-end machine learning workflow for classifying handwritten digits from the famous MNIST dataset. It focuses on optimizing **K-Nearest Neighbors** and **Random Forest** classifiers by using `GridSearchCV` to automatically find the best hyperparameters, resulting in highly accurate and robust models.

The script handles everything from data loading and preprocessing to model training, tuning, and final evaluation.

## ✨ Features

- **Data Handling**: Loads the MNIST dataset directly via Keras and preprocesses it by reshaping and normalizing the image data for optimal model performance.
- **Automated Hyperparameter Tuning**: Employs `GridSearchCV` to systematically search for the optimal hyperparameters for both KNN and Random Forest models, saving manual effort and improving accuracy.
- **Model Optimization**: Identifies the best-performing configuration for each algorithm based on cross-validated accuracy scores.
- **Performance Evaluation**: Fits the fine-tuned models and evaluates their final performance on an unseen test set, printing the results.

## 🛠️ Technologies Used

*   **Core**: Python
*   **Machine Learning**: Scikit-learn
*   **Dataset/Deep Learning Library**: Keras / TensorFlow
*   **Numerical Computing**: NumPy

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

- Python 3.8+

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/michalglomsky/Classification-of-Handwritten-Digits.git
    cd Classification-of-Handwritten-Digits
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```
    *(Note: The code from your project should be saved as a Python file, e.g., `main.py`)*

## 🏃‍♀️ Usage

Once the setup is complete, you can run the script to start the classification and tuning process.

1.  **Run the script:**
    ```sh
    python main.py
    ```

2.  **Observe the output:**

    The script will perform the following actions automatically:
    - Load and preprocess the MNIST dataset.
    - Define hyperparameter grids for the K-Nearest Neighbors and Random Forest models.
    - Run `GridSearchCV` to find the best combination of parameters for each model. This may take a few moments.
    - Evaluate the fine-tuned models on the test data and print their final accuracy.

    The output will show the best estimator found and its performance:
    ```
    K-nearest neighbours algorithm
    best estimator: KNeighborsClassifier(n_neighbors=3, weights='distance')
    accuracy: 0.95...

    Random forest algorithm
    best estimator: RandomForestClassifier(class_weight='balanced', max_features='sqrt', n_estimators=500, random_state=40)
    accuracy: 0.94...
    ```

## 📄 License

This project is licensed under the MIT License.

---

Created by Michał Głomski

