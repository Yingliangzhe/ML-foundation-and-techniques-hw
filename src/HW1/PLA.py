import numpy as np
import matplotlib
from matplotlib import pyplot as plt


def preprocess(data_path):

    data = np.genfromtxt(data_path)

    n, d = data.shape

    x = data[:, :-1]
    # add a x0 to the original X
    x = np.c_[np.ones(n), x]
    y = data[:, -1]
    return x, y


def judge(x, y, w):
    n, d = x.shape

    num = np.sum(x.dot(w)*y > 0)
    return num == n


def PLA(x, y, eta=1, max_step=np.inf):

    n, d = x.shape
    w = np.zeros(d)
    t = 0 # to record iteration number
    i = 0 # element index
    last = 0 # last one that failed

    while not judge(x, y, w) and t < max_step:
        if np.sign(x[i, :].dot(w)*y[i]) <= 0:
            t += 1
            w += eta * y[i] * x[i, :]
            last = i
        i += 1
        if i == n:
            i = 0
    return t, last, w


def f1(g, x, y, n: int, eta=1, max_step=np.inf):
    """
    run this algorithm n times, to see the average iteration number finding the best line
    :param g:
    :param x:
    :param y:
    :param n:
    :param eta:
    :param max_step:
    :return:
    """
    result = []
    data = np.c_[x, y]
    for i in range(n):
        np.random.shuffle(data)
        x = data[:, :-1]
        y = data[:, -1]
        result.append(g(x, y, eta=eta, max_step=max_step)[0])

    plt.hist(result)
    plt.xlabel('iteration times')
    counter = np.mean(result)
    plt.title(f'average iteration times is {counter}')
    plt.show()


#f1(HW1, X, y, 2000, 0.5)


def count(x, y, w):
    # record the error number
    num = np.sum(x.dot(w)*y <= 0)
    return num


def Pocket_PLA(x, y, eta=1, max_step=np.inf):
    n, d = x.shape

    w = np.zeros(d)
    w0 = np.zeros(d)

    t = 0
    error = count(x, y, w0)
    i = 0

    while error != 0 and t < max_step:
        if np.sign(x[i,:].dot(w)*y[i])<=0:
            w += eta*y[i]*x[i,:]
            t += 1
            error_now = count(x, y, w)
            if error_now < error:
                error = error_now
                w0 = np.copy(w)

        i += 1

        if i == n:
            i = 0
    return error, w0

def f2(g, x1, y1, x2, y2, n, eta=1, max_step=np.inf):
    """
    train n times, x1, y1 train, x2, y2 test
    :param g:
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :param n:
    :param eta:
    :param max_step:
    :return:
    """
    result = []
    data = np.c_[x1, y1]
    m = x2.shape[0]
    for _ in range(n):
        np.random.shuffle(data)
        x = data[:, :-1]
        y = data[:, -1]
        w = g(x, y, eta=eta, max_step=max_step)[-1]
        result.append(count(x2, y2, w)/m) # 116 / 400
    plt.hist(result)
    plt.xlabel('error rate')
    plt.title('average error rate ' + str(np.mean(result)))
    plt.show()
