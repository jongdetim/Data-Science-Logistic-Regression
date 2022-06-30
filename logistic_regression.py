from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:

    # This function intializes the alpha value and iteration
    def __init__(self, alpha=0.01, n_iteration=100):
        self.alpha = alpha  # value in the object
        self.n_iter = n_iteration
        self.theta = []
        self.cost = []
        self.weights = []

    # This function is resonsible for calculating the sigmoid value with given parameter
    def _sigmoid_function(self, x):
        value = 1 / (1 + np.exp(-x))
        return value

    def _cost_function(self, h, y):  # The fuctions calculates the cost value
        m = len(y)
        cost = (1 / m) * (np.sum(-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))))
        return cost

    # This function calculates the theta value by gradient descent
    def _gradient_descent(self, x, h, theta, y, m):
        gradient_value = np.dot(x.T, (h - y)) / m
        theta -= self.alpha * gradient_value
        return theta

    def fit(self, x, y):  # This function primarily calculates the optimal theta value using which we predict the future data
        print("Fitting the given dataset..")
        self.theta = []
        self.cost = []
        x = np.insert(x, 0, 1, axis=1)
        m = len(y)
        for i in np.unique(y):
            print('Descending the gradient for label type ' + str(i) + 'vs Rest')
            y_onevsall = np.where(y == i, 1, 0)
            theta = np.zeros(x.shape[1])
            cost = []
            thetas = []
            for _ in range(self.n_iter):
                z = x.dot(theta)
                h = self._sigmoid_function(z)
                theta = self._gradient_descent(x, h, theta, y_onevsall, m)
                print(theta)
                cost.append(self._cost_function(h, y_onevsall))
                thetas.append(deepcopy(theta))
            self.theta.append((theta, i))
            self.weights.append((thetas, i))
            self.cost.append((cost, i))
        return self

    def predict(self, x):  # this function calls the max predict function to classify the individual features
        x = np.insert(x, 0, 1, axis=1)
        x_predicted = [max((self._sigmoid_function(i.dot(theta)), c)
                           for theta, c in self.theta)[1] for i in x]
        return x_predicted

    def predict2(self, X, y):
        print(self.theta)
        print(np.array(self.theta, dtype=object).dot(X.T))
        X = np.insert(X, 0, 1, axis=1)
        predictions = self._sigmoid_function(np.array(self.theta).dot(X.T)).T
        return [np.unique(y)[x] for x in predictions.argmax(1)]

    def accuracy(self, x, y):  # This function compares the predicted label with the actual label to find the model performance
        accuracy = sum(self.predict(x) == y) / len(y)
        return accuracy

    def _plot_cost(self, costh):  # This function plot the cost function value
        for cost, c in costh:
            plt.plot(range(len(cost)), cost, 'r')
            plt.title("Convergence Graph of Cost Function of type-" +
                      str(c) + " vs All")
            plt.xlabel("Number of Iterations")
            plt.ylabel("Cost")
            plt.show()
    
    def _plot_weights(self):
        weights, c = self.weights[0]
        for weight in weights:
            print(weight)
            plt.plot(range(len(weight)), weight, 'r')
            plt.title("weights over time of type: " +
                        str(c) + " vs All")
            plt.xlabel("features")
            plt.ylabel("Weight")
            plt.show()

    def save_model(self, filename='./datasets/weights.csv'):
        f = open(filename, 'w+')
        # np.savetxt(f, self.theta)
        f.write(str(self.theta))