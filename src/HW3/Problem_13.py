import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from sklearn.preprocessing import PolynomialFeatures


# show this generated data in plot
n = 1000
p = 0.1

# times of experiments
m = 1000


def generate_data(n, p=0.1):
    x = np.random.uniform(-1, 1, size=(n, 2))
    # axis = 1, means that columns will be summed up
    y = np.sign(np.sum(x ** 2, axis=1) - 0.6)
    # than we flip the y in 0.1 probability, to introduce the noise
    P = np.random.uniform(0, 1, n)
    # using this form we will get the
    y[P < p] *= -1

    return x, y

"""
x, y = generate_data(n, p)
plt.scatter(x[y > 0][:, 0], x[y > 0][:, 1], s=1)
plt.scatter(x[y < 0][:, 0], x[y < 0][:, 1], s=1)
plt.show()

"""


def linear_regression(n, p):

    E_in = np.array([])

    # for linear regression we can build a close form solution
    for i in range(m):
        x, y = generate_data(n, p)
        x = np.c_[np.ones(n), x]

        w = inv(x.T.dot(x)).dot(x.T).dot(y)
        # * is the corresponding element multiplicate, dot ist matrix multiplicate
        e_in = np.mean(np.sign(x.dot(w)*y) < 0)
        E_in = np.append(E_in, e_in)
    return E_in

"""
E_in = linear_regression(n, p)
print(np.average(E_in))
plt.hist(E_in)
plt.title('E_in without feature transform')
plt.show()
"""


"""
Linear Regression with feature transform
"""
# to transform the data to 2 polynomial
poly = PolynomialFeatures(2)
w = []
E_in = np.array([])
E_out = np.array([])

for i in range(m):

    x, y = generate_data(n)
    x_poly = poly.fit_transform(x)
    w_poly = inv(x_poly.T.dot(x_poly)).dot(x_poly.T).dot(y)
    # here, we need to compute, how is the mean value of error in each experiment
    e_in = np.mean(np.sign(x_poly.dot(w_poly)) * y < 0)
    E_in = np.append(E_in, e_in)

    # test dataset
    x_test, y_test = generate_data(n)
    x_test_poly = poly.fit_transform(x_test)
    e_out = np.mean(np.sign(x_test_poly.dot(w_poly)) * y_test < 0)
    E_out = np.append(E_out, e_out)

    w.append(w_poly)

w = np.array(w)
w3 = w[:, 4]

plt.hist(w3)
plt.title('w3')
plt.show()
print(f'w3 mean value is {w3}')
print(f'w mean value is {str(np.mean(w, axis=0))}')
print(f'E_out error is {str(np.mean(E_out))}')

pass



