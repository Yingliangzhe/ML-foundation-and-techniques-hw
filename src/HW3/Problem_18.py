import numpy as np


train_file_path = 'HW3/dataset/data_train.txt'
test_file_path = 'HW3/dataset/data_test.txt'


def preprocess(x):
    n, d = x.shape
    return np.c_[np.ones(n), x]


def sigmoid(s):
    return 1 / (1 + np.exp(-s))


def gradient(x, y, w):
    s = x.dot(w).reshape(-1, 1) * -y
    theta = sigmoid(s)
    # x * y multiplicator: y[i] is applied to every x_n[i] element
    temp = -x * y
    grad = np.mean(temp * theta, axis=0).reshape(-1, 1)
    return grad


data_train = np.genfromtxt(train_file_path)
data_test = np.genfromtxt(test_file_path)

x_train = data_train[:, :-1]
y_train = data_train[:, -1].reshape(-1, 1)
# add the x0 item to all x_train
x_train = preprocess(x_train)

x_test = data_test[:, :-1]
y_test = data_test[:, -1].reshape(-1, 1)
# add the x0 item to all x_test
x_test = preprocess(x_test)


n, d = x_train.shape

# Problem 18
w = np.ones(d).reshape(-1, 1)

k = 0.001

for i in range(2000):
    grad = gradient(x_train, y_train, w)
    w -= k * grad

y_test_pred = x_test.dot(w)
y_test_pred[y_test_pred > 0] = 1
y_test_pred[y_test_pred < 0] = -1
E_out = np.mean(y_test_pred != y_test)
print(E_out)
print(w)


