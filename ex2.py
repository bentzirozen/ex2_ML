from numpy.random.mtrand import shuffle
import numpy as np

def perceptron(X,Y):
        np.linalg.norm(X,ord=2)
        orig = X
        eta = 0.1
        iters = 20
        weights = np.zeros((3, X.shape[1]))
        set = np.c_[X, Y]
        for _ in range(iters):
            shuffle(set)
            for x, y in zip(X, Y):
                y = float(y)
                y_hat = np.argmax(np.dot(weights, x))
                # update
                if y != y_hat:
                    y = int(y)
                    y_hat = int(y_hat)
                    weights[y, :] += eta * x
                    weights[y_hat, :] += -eta * x


def passive_agressive(X,Y):
    iters = 50
    weights_arr = np.zeros(3, X.shape[1])
    for i in range(iters):
        set = np.c_[X, Y]
        shuffle(set)
        for x, y in zip(X, Y):
            # predict.
            y_hat = np.argmax(np.dot(weights_arr, x))
            # update
            if y != y_hat:
                loss = max(0.0, 1 - np.dot(weights_arr[y], x) + np.dot(weights_arr[y_hat], x))
                loss /= ((np.power(np.linalg.norm(x, ord=2), 2)) * 2)
                weights_arr[y, :] = weights_arr[y, :] + loss * x
                weights_arr[y_hat, :] = weights_arr[y_hat, :] - loss * x

def min_max_norm(examples):

    features = len(examples[0])
    new_max = 10
    new_min = 4

    for i in range(features):
        old_min = np.amin(examples[:, i])              # minimum feature value .
        denom = np.amax(examples[:, i]) - old_min      # old_max - old_min .
        if denom!=0:
            value = (examples[:, i] - old_min) / denom
            examples[:, i] = value * (new_max - new_min) + new_min


def read_data(data):
    X = []
    sex_dict = {'M': 0, 'F': 1, 'I': 2}
    with open(data, 'r') as values:
        for idx in values:
            idx = idx.strip().split(',')
            idx[0] = sex_dict[idx[0]]
            X.append(np.array(idx, dtype=np.float64))
    return np.asarray(X)
def main(args):
    X = read_data(args[1])
    Y = np.genfromtxt(args[2],delimiter=',')
    perceptron(X,Y)
if __name__ == '__main__':
    main(['ex2.py', 'train_x.txt', 'train_y.txt', 'test_x.txt'])