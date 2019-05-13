from numpy.random.mtrand import shuffle
import numpy as np


class Perceptron:

    def __init__(self, examples, labels, epochs=100, num_classes=3):

        self.before = examples
        self.examples = examples            # shape = (#examples=3286, #features=8).
        self.labels = labels                # shape = (#labels=3286).
        self.epochs = epochs

        self.min_max_norm()                 # normalize the data .
        self.set = np.c_[examples, labels]  # save as set for shuffle.

        # weights shape = (#classes=3, #features=8).
        self.weights = np.zeros((num_classes, examples.shape[1]))

        self.eta = 1

    def train(self):
        for _ in range(self.epochs):
            shuffle(self.set)
            for x, y in zip(self.examples, self.labels):
                y = float(y)
                y_hat = np.argmax(np.dot(self.weights, x))
                # update
                if y != y_hat:
                    y = int(y)
                    y_hat = int(y_hat)
                    self.weights[y, :] += self.eta * x
                    self.weights[y_hat, :] += -self.eta * x
        test_x = self.read_data("check_x.txt")
        test_y = np.genfromtxt("check_y.txt", delimiter=',')
        self.test(test_x, test_y)

    def test(self, test_examples, test_labels):

        errors = 0
        m = test_examples.shape[0]      # number of test examples

        t = self.min_max_test(test_examples)

        for i in range(m):
            # y_hat = np.argmax(np.dot(self.weights, test_examples[i]))
            y_hat = np.argmax(np.dot(self.weights, t[i]))

            if test_labels[i] != y_hat:
                errors = errors + 1

        loss = float(errors) / m

        print("number of errors: " + str(errors))
        print("number of tests: " + str(m))
        print("the Loss is: " + str(loss))
        print("which means: " + str((1 - loss) * 100) + " of correctness")

    def z_score_norm(self):
        features = self.examples.shape[1]

        for i in range(features):

            mean = np.mean(self.examples[:, i])
            std_dev = np.std(self.examples[:, i])

            self.examples[:, i] = (self.examples[:, i] - mean) / std_dev

    def min_max_norm(self):

        features = len(self.examples)
        new_max = 10
        new_min = 4

        for i in range(features):
            old_min = np.amin(self.examples[:, i])              # minimum feature value .
            denom = np.amax(self.examples[:, i]) - old_min      # old_max - old_min .
            if denom!=0:
                value = (self.examples[:, i] - old_min) / denom

            self.examples[:, i] = value * (new_max - new_min) + new_min

    def min_max_test(self, _examples):

        features = self.examples.shape[1]
        new_max = 10
        new_min = 4

        for i in range(features):
            old_min = np.amin(self.before[:, i])              # minimum feature value .
            denom = np.amax(self.before[:, i]) - old_min      # old_max - old_min .

            value = (_examples[:, i] - old_min) / denom

            _examples[:, i] = value * (new_max - new_min) + new_min

        return _examples

def read_data(data):
    X = []
    sex_dict = {'M': 0, 'F': 1, 'I': 2}
    with open(data, 'r') as values:
        for idx in values:
            idx = idx.strip().split(',')
            idx[0] = sex_dict[idx[0]]
            X.append(np.array(idx, dtype=np.float64))
    return X

X = read_data("train_x.txt")
Y = np.genfromtxt("train_y.txt", delimiter=',')
p = Perceptron(X, Y)
p.train_perc(X, Y)