import numpy as np
import sys
def perceptron(X,Y):
        z_score_norm(X)
        eta = 0.1
        iters = 15
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
        return weights

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
                norm = ((np.power(np.linalg.norm(x, ord=2), 2)) * 2)
                if(norm!=0):
                    loss/=norm
                    weights[y, :] = weights[y, :] + loss * x
                    weights[y_hat, :] = weights[y_hat, :] - loss * x
    return weights

def svm(X,Y):
    z_score_norm(X)
    iters = 50
    eta =0.01
    alpha = 0.1
    weights = np.zeros((3, X.shape[1]))
    for i in range(iters):
        for x, y in zip(X, Y):
            y_hat = np.argmax(np.dot(weights, x))
            # update
            if y != y_hat:
                y = int(y)
                y_hat = int(y_hat)
                weights[y, :] = (1 - eta * alpha) * weights[y, :] + eta * x
                weights[y_hat, :] = (1 - eta * alpha) * weights[y_hat, :] - eta * x
            for i in range(weights.shape[0]):
                if i != y and i != y_hat:
                    weights[i, :] = (1 - eta * alpha) * weights[i, :]
    return weights

def z_score_norm(examples):
    features = examples.shape[1]
    for i in range(features):
        mean = np.mean(examples[:, i])
        denom = np.std(examples[:, i])
        if denom !=0:
            examples[:, i] = (examples[:, i] - mean) / denom


def read_data(data):
    X = []
    sex_dict = {'M': 0, 'F': 1, 'I': 2}
    with open(data, 'r') as values:
        for idx in values:
            idx = idx.strip().split(',')
            idx[0] = sex_dict[idx[0]]
            X.append(np.array(idx, dtype=np.float64))
    return np.asarray(X)

def test(test_examples, test_labels,pa_weights,per_weights,svm_weights):
        m = test_examples.shape[0]  # number of test examples
        z_score_norm(test_examples)
        for i in range(m):
            # y_hat = np.argmax(np.dot(self.weights, test_examples[i]))
            y_hat = np.argmax(np.dot(per_weights, test_examples[i]))
            per_result = str(y_hat)
            per_result+=","
            y_hat = np.argmax(np.dot(pa_weights, test_examples[i]))
            pa_result = str(y_hat)
            y_hat = np.argmax(np.dot(svm_weights, test_examples[i]))
            svm_result = str(y_hat)
            svm_result+=","
            print("perceptron: ",per_result,"svm: ",svm_result,"pa: ",pa_result)


if __name__ == '__main__':
    args = sys.argv
    X = read_data(args[1])
    Y = np.genfromtxt(args[2], delimiter=',')
    pa_weights = passive_agressive(X, Y)
    perceptron_weights = perceptron(X, Y)
    svm_weights = svm(X, Y)
    test_x = read_data(args[3])
    test(test_x, Y, pa_weights, perceptron_weights, svm_weights)