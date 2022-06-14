import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:

    # This function intializes the alpha value and iteration
    def __init__(self, alpha=0.01, n_iteration=100):
        self.alpha = alpha  # value in the object
        self.n_iter = n_iteration
        self.theta = []
        self.cost = []

    # This function is resonsible for calculating the sigmoid value with given parameter
    def _sigmoid_function(self, x):
        value = 1 / (1 + np.exp(-x))
        return value

    def _cost_function(self, h, y):  # The fuctions calculates the cost value
        m = len(y)
        cost = (1 / m) * (np.sum(-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))))
        return cost

    # This function calculates the theta value by gradient descent
    def _gradient_descent(self, X, h, theta, y, m):
        gradient_value = np.dot(X.T, (h - y)) / m
        theta -= self.alpha * gradient_value
        return theta

    def fit(self, X, y):  # This function primarily calculates the optimal theta value using which we predict the future data
        print("Fitting the given dataset..")
        X = np.insert(X, 0, 1, axis=1)
        m = len(y)
        for i in np.unique(y):
            print('Descending the gradient for label type ' + str(i) + 'vs Rest')
            y_onevsall = np.where(y == i, 1, 0)
            theta = np.zeros(X.shape[1])
            cost = []
            for _ in range(self.n_iter):
                z = X.dot(theta)
                h = self._sigmoid_function(z)
                theta = self._gradient_descent(X, h, theta, y_onevsall, m)
                cost.append(self._cost_function(h, y_onevsall))
            self.theta.append((theta, i))
            self.cost.append((cost, i))
        return self

    def predict(self, X):  # this function calls the max predict function to classify the individul feauter
        X = np.insert(X, 0, 1, axis=1)
        X_predicted = [max((self._sigmoid_function(i.dot(theta)), c)
                           for theta, c in self.theta)[1] for i in X]

        return X_predicted

    def accuracy(self, X, y):  # This function compares the predicted label with the actual label to find the model performance
        accuracy = sum(self.predict(X) == y) / len(y)
        return accuracy

    def _plot_cost(self, costh):  # This function plot the Cost function value
        for cost, c in costh:
            plt.plot(range(len(cost)), cost, 'r')
            plt.title("Convergence Graph of Cost Function of type-" +
                      str(c) + " vs All")
            plt.xlabel("Number of Iterations")
            plt.ylabel("Cost")
            plt.show()
