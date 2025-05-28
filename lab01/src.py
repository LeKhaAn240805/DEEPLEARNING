import numpy as np
from tensorflow.keras.datasets import mnist
from statistics import mean, stdev
from tabulate import tabulate

# ==== 1. LOAD & PREPROCESS DATA ====
def load_mnist_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Flatten images: (28,28) -> (784,)
    X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 784).astype('float32') / 255.0

    # One-hot encode labels
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_mnist_data()

# ==== 2. ACTIVATION FUNCTIONS & DERIVATIVES ====
def relu(Z): return np.maximum(0, Z)
def relu_deriv(Z): return (Z > 0).astype(float)

def sigmoid(Z): return 1 / (1 + np.exp(-Z))
def sigmoid_deriv(Z): sig = sigmoid(Z); return sig * (1 - sig)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

# ==== 3. MODEL COMPONENTS ====
def initialize_parameters(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

def forward_propagation(X, W1, b1, W2, b2, activation_func):
    Z1 = X.dot(W1) + b1
    A1 = activation_func(Z1)
    Z2 = A1.dot(W2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def compute_loss(A2, Y):
    m = Y.shape[0]
    log_probs = -np.log(A2[range(m), Y.argmax(axis=1)])
    return np.sum(log_probs) / m

def backward_propagation(X, Y, Z1, A1, Z2, A2, W1, W2, activation_deriv):
    m = X.shape[0]
    dZ2 = A2 - Y
    dW2 = A1.T.dot(dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dA1 = dZ2.dot(W2.T)
    dZ1 = dA1 * activation_deriv(Z1)
    dW1 = X.T.dot(dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    return dW1, db1, dW2, db2

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

# ==== 4. TRAINING ====
def train(X, Y, W1, b1, W2, b2, batch_size, learning_rate, epochs, activation_func, activation_deriv):
    m = X.shape[0]
    for epoch in range(epochs):
        for i in range(0, m, batch_size):
            X_batch = X[i:i+batch_size]
            Y_batch = Y[i:i+batch_size]
            Z1, A1, Z2, A2 = forward_propagation(X_batch, W1, b1, W2, b2, activation_func)
            dW1, db1, dW2, db2 = backward_propagation(X_batch, Y_batch, Z1, A1, Z2, A2, W1, W2, activation_deriv)
            W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
    return W1, b1, W2, b2

# ==== 5. PREDICT ====
def predict(X, W1, b1, W2, b2, activation_func):
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2, activation_func)
    return np.argmax(A2, axis=1)

# ==== 6. HYPERPARAMETER TESTING ====
hyperparams = [
    (32, 0.1, 16, relu, relu_deriv),
    (16, 0.2, 64, sigmoid, sigmoid_deriv),
    (64, 0.3, 32, relu, relu_deriv),
    (32, 0.4, 128, sigmoid, sigmoid_deriv),
    (16, 0.5, 32, relu, relu_deriv),
]

results = []

for batch_size, lr, hidden_size, act_func, act_deriv in hyperparams:
    accuracies = []
    for _ in range(5):
        W1, b1, W2, b2 = initialize_parameters(784, hidden_size, 10)
        W1, b1, W2, b2 = train(X_train, y_train, W1, b1, W2, b2, batch_size, lr, 10, act_func, act_deriv)
        y_pred = predict(X_test, W1, b1, W2, b2, act_func)
        acc = np.mean(y_pred == y_test.argmax(axis=1))
        accuracies.append(acc)
    mean_acc = mean(accuracies)
    std_acc = stdev(accuracies)
    results.append((batch_size, lr, hidden_size, act_func.__name__, mean_acc, std_acc))

# ==== 7. OUTPUT ====
headers = ["Batch Size", "Learning Rate", "Hidden Size", "Activation", "Mean Acc (%)", "Std Acc (%)"]
table = []

for r in results:
    table.append([r[0], r[1], r[2], r[3], round(r[4]*100, 2), round(r[5]*100, 2)])

print("\nKết quả huấn luyện với các cấu hình:")
print(tabulate(table, headers=headers, tablefmt="grid"))