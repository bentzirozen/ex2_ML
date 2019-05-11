import numpy as np
from random import shuffle

def read_date(data):
    X = []
    sex_dict = {'M': 0, 'F': 1, 'I': 2}
    with open(data, 'r') as values:
        for idx in values:
            idx = idx.strip().split(',')
            idx[0] = sex_dict[idx[0]]
            X.append(np.array(idx,dtype=np.float64))
    return np.asanyarray(X)
def _init_wights():
    rows = 3
    columns = 8
    wights = [[0 for column in range(columns)] for row in range(rows)]
    wights = np.asarray(wights)
    wights = wights.astype(np.float)
    return wights

def main(args):
    X = read_date(args[1])
    Y = np.genfromtxt(args[2],delimiter=',')
    perceptron(X,Y)
def perceptron(X,Y):
    eta = 0.01
    iters = 20
    d = len(X[0])
    #3 - 3 types of weights
    w = np.zeros(shape=(3,d))
   # w = _init_wights()
    for _ in range(iters):
        for x,y in zip(X,Y):
            y = float(y)
            y_hat = np.argmax(np.dot(w,x))
            #update
            if y!=y_hat:
                y = int(y)
                y_hat = int(y_hat)
                w[y, :]+=eta * x
                w[y_hat,:]+= -eta*x
    print("finished without errors")








if __name__ == '__main__':
    main(['ex2.py', 'train_x.txt', 'train_y.txt', 'test_x.txt'])


