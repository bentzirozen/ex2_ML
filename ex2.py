from numpy.random import shuffle
import numpy as np

def perceptron(X,Y):
        z_score_norm(X)
        eta = 1
        iters = 18
        weights = np.zeros((3, X.shape[1]))
        set = np.c_[X, Y]
        for _ in range(iters):
           # shuffle(X)
            #shuffle(Y)
            for x, y in zip(X,Y):
                y = float(y)
                y_hat = np.argmax(np.dot(weights, x))
                # update
                if y != y_hat:
                    y = int(y)
                    y_hat = int(y_hat)
                    weights[y, :] += eta * x
                    weights[y_hat, :] += -eta * x
        test_x = read_data("check_x.txt")
        test_y = np.genfromtxt("check_y.txt",delimiter=',')
        test(test_x,test_y,weights)

def passive_agressive(X,Y):
    z_score_norm(X)
    iters = 15
    weights = np.zeros((3, X.shape[1]))
    for i in range(iters):
        for x, y in zip(X, Y):
            y_hat = np.argmax(np.dot(weights, x))
            # update
            if y != y_hat:
                y = int(y)
                y_hat = int(y_hat)
                loss = max(0.0, 1 - np.dot(weights[y], x) + np.dot(weights[y_hat], x))
                loss /= ((np.power(np.linalg.norm(x, ord=2), 2)) * 2)
                weights[y, :] = weights[y, :] + loss * x
                weights[y_hat, :] = weights[y_hat, :] - loss * x
    test_x = read_data("check_x.txt")
    test_y = np.genfromtxt("check_y.txt", delimiter=',')
    test(test_x, test_y, weights)

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

def z_score_norm(examples):
    features = examples.shape[1]
    for i in range(features):
        mean = np.mean(examples[:, i])
        std_dev = np.std(examples[:, i])
        if std_dev !=0:
            examples[:, i] = (examples[:, i] - mean) / std_dev


def read_data(data):
    X = []
    sex_dict = {'M': 0, 'F': 1, 'I': 2}
    with open(data, 'r') as values:
        for idx in values:
            idx = idx.strip().split(',')
            idx[0] = sex_dict[idx[0]]
            X.append(np.array(idx, dtype=np.float64))
    return np.asarray(X)

def test(test_examples, test_labels,weights):
        errors = 0
        m = test_examples.shape[0]  # number of test examples
        z_score_norm(test_examples)
        for i in range(m):
            # y_hat = np.argmax(np.dot(self.weights, test_examples[i]))
            y_hat = np.argmax(np.dot(weights, test_examples[i]))

            if test_labels[i] != y_hat:
                errors = errors + 1

        loss = float(errors) / m

        print("number of errors: " + str(errors))
        print("number of tests: " + str(m))
        print("the Loss is: " + str(loss))
        print("which means: " + str((1 - loss) * 100) + " of correctness")

def main(args):
    X = read_data(args[1])
    Y = np.genfromtxt(args[2],delimiter=',')
    passive_agressive(X,Y)
    perceptron(X,Y)
if __name__ == '__main__':
    main(['ex2.py', 'train_x.txt', 'train_y.txt', 'test_x.txt'])