import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv(r'E:\Student Performance Prediction\final_df.csv')

# Data analysis
df = df[df["sum_click"] <= 10]
df = df[df["num_of_prev_attempts"] <= 4]


# Split the data into features (X) and target (Y)
X = df.drop("final_result", axis=1)  # Features
Y = df["final_result"]  # Target variable

# Split the data into training and testing sets (70% train, 30% test)
X1_train, X1_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

scaler = MinMaxScaler()

X1_train = scaler.fit_transform(X1_train)
X1_test = scaler.transform(X1_test)

def map_label(label):
    if label == 'Fail':
        return 0
    else:
        return 1

map_label = np.vectorize(map_label)

# Convert y_train and y_test to numpy arrays and reshape based on their lengths
Y_train = np.array(Y_train).reshape(1, -1)
Y_test = np.array(Y_test).reshape(1, -1)

# Transpose X_train and X_test
X_train = X1_train.T
X_test = X1_test.T

print("After reshaping:")
print("X_train shape:", X_train.shape)
print("y_train shape:", Y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", Y_test.shape)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def model(X, Y, learning_rate, iterations):
    # Ensure X is of shape (features, samples)
    m = X.shape[1]  # Number of samples
    n = X.shape[0]  # Number of features

    # Initialize weights and bias
    W = np.zeros((n, 1))
    B = 0

    cost_list = []

    for i in range(iterations):
        # Linear model
        Z = np.dot(W.T, X) + B  # Z should be of shape (1, m)
        A = sigmoid(Z)  # A should also be (1, m)

        # Cost function
        cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

        # Gradient calculation
        dW = (1 / m) * np.dot(X, (A - Y).T)  # dW shape should be (n, 1)
        dB = (1 / m) * np.sum(A - Y)

        # Parameter updates
        W = W - learning_rate * dW
        B = B - learning_rate * dB

        # Store cost every 10% of iterations
        cost_list.append(cost)

        if i % (iterations / 10) == 0:
            print("Cost after", i, "iterations:", cost)

    return W, B, cost_list

# Apply the map_label function to y_train and y_test to convert them to numeric values
Y_train = map_label(Y_train)
Y_test = map_label(Y_test)

# Convert to numpy arrays and ensure float dtype
y_train = np.array(Y_train, dtype=float)
y_test = np.array(Y_test, dtype=float)

# Run the model with the updated y_train and y_test
iterations = 100000
learning_rate = 0.0015
W, B, cost_list = model(X_train, y_train, learning_rate=learning_rate, iterations=iterations)

plt.plot(np.arange(iterations), cost_list)
plt.show()


def accuracy(X, Y, W, B):
    Z = np.dot(W.T, X) + B
    A = sigmoid(Z)

    A = A > 0.5

    A = np.array(A, dtype='int64')

    acc = (1 - np.sum(np.absolute(A - Y)) / Y.shape[1]) * 100

    print("Accuracy of the model is : ", round(acc, 2), "%")


def plot_predictions(X, Y, W, B):
    # Get predictions using the model
    Z = np.dot(W.T, X) + B
    A = sigmoid(Z)

    # Convert predictions to binary classes (0 or 1)
    predictions = (A > 0.5).astype(int)

    # Plot the actual vs. predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(range(Y.shape[1]), Y.flatten(), color="blue", alpha=0.6, label="Actual", marker='o')
    plt.scatter(range(Y.shape[1]), predictions.flatten(), color="red", alpha=0.6, label="Predicted", marker='x')

    # Add labels and legend
    plt.xlabel("Sample Index")
    plt.ylabel("Label")
    plt.title("Actual vs Predicted Labels")
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()

# Plot the predictions against actual values
plot_predictions(X_test, y_test, W, B)