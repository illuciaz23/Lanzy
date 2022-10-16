import numpy as np

import matplotlib.pyplot as plt
# from tqdm import tqdm
import matplotlib.pyplot as plt
import torch


class Knn(object):

    def __init__(self, k=2):
        self.k = k

    def fit(self, X, y):
        self.X = torch.tensor(X).cuda()

        self.y = torch.tensor(y).cuda()

    def predict(self, X):
        # TODO Predict the label of X by
        # the k nearest neighbors.
        X = torch.tensor(X).cuda()

        # Input:
        # X: np.array, shape (n_samples, n_features)

        # Output:
        # y: np.array, shape (n_samples,)

        # Hint:
        # 1. Use self.X and self.y to get the training data.
        # 2. Use self.k to get the number of neighbors.
        # 3. Use np.argsort to find the nearest neighbors.

        # YOUR CODE HERE
        # raise NotImplementedError
        y = torch.zeros((int(1e4),))
        for i in range(0, int(1e4)):
            #print(i)

            #s = torch.sqrt(torch.sum(torch.square(self.X.float() - X[i].float()), dim=(1, 2))) #欧式距离
            s = torch.sum(torch.abs(self.X - X[i]), dim=(1, 2))
            l = torch.argsort(s)
            neibors = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
            for j in range(0, self.k):
                neibors[self.y[l[j]]] += 1/s[l[j]]

            y[i] = neibors.index(max(neibors))
        return y.numpy()
        # End of todo


def load_mnist(root='../data/mnist'):
    # TODO Load the MNIST dataset
    # 1. Download the MNIST dataset from
    #    http://yann.lecun.com/exdb/mnist/
    # 2. Unzip the MNIST dataset into the
    #    mnist directory.
    # 3. Load the MNIST dataset into the
    #    X_train, y_train, X_test, y_test
    #    variables.

    # Input:
    # root: str, the directory of mnist

    # Output:
    # X_train: np.array, shape (6e4, 28, 28)
    # y_train: np.array, shape (6e4,)
    # X_test: np.array, shape (1e4, 28, 28)
    # y_test: np.array, shape (1e4,)

    # Hint:
    # 1. Use np.fromfile to load the MNIST dataset(notice offset).
    # 2. Use np.reshape to reshape the MNIST dataset.

    # YOUR CODE HERE
    # raise NotImplementedError
    x_train = np.fromfile(root + '/train-images-idx3-ubyte', dtype=np.int8, offset=16).reshape((int(6e4), 28, 28))
    y_train = np.fromfile(root + '/train-labels-idx1-ubyte', dtype=np.int8, offset=8)
    x_test = np.fromfile(root + '/t10k-images-idx3-ubyte', dtype=np.int8, offset=16).reshape((int(1e4), 28, 28))
    y_test = np.fromfile(root + '/t10k-labels-idx1-ubyte', dtype=np.int8, offset=8)

    return x_train, y_train, x_test, y_test

    # End of todo


def main():
    # We move our tensor to the GPU if available

    X_train, y_train, X_test, y_test = load_mnist()

    for k in range(5,30,2):
        knn = Knn(k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        # correct = sum((y_test - y_pred) == 0)
        acc = sum((y_test - y_pred) == 0) / len(X_test)
        print(k, ':', acc)
    '''
    print('==> correct:', correct)
    print('==> total:', len(X_test))
    print('==> acc:', correct / len(X_test))

    # plot pred samples
    fig, ax = plt.subplots(nrows=4, ncols=5, sharex='all', sharey='all')
    fig.suptitle('Plot predicted samples')
    ax = ax.flatten()
    for i in range(20):
        img = X_test[i]
        ax[i].set_title(y_pred[i])
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()
    '''


if __name__ == '__main__':
    main()
