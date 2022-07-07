from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    '''
    Logistic Regression model class

    Arguments
    ----------
    alpha : float
        learning rate for gradient descent algorithm. default is 0.01
    n_terations : int
        number of times to perform the gradient descent step. default is 100
    random_weights : bool
        set True to use random starting weights. False to initialize weights to 0. default is False
    '''
    def __init__(self, alpha:float=0.01, n_iteration:int=100, random_weights:bool=False):
        self.alpha = alpha
        self.n_iter = n_iteration
        self.theta = []
        self.cost = []
        self.weights = []
        self.random_weights = random_weights

    def _sigmoid_function(self, x):
        '''Sigmoid function to introduce nonlinearity'''
        value = 1 / (1 + np.exp(-x))
        return value

    def _cost_function(self, h, y):
        '''Cost function that the model tries to find the minimum of by using gradient descent'''
        m = len(y)
        cost = (1 / m) * (np.sum(-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))))
        return cost

    def _gradient_descent(self, x, h, theta, y, m):
        '''Calculates the gradient values for each feature (partial derivatives) & adjusts thetas accordingly'''
        gradient_value = np.dot(x.T, (h - y)) / m
        theta -= self.alpha * gradient_value
        return theta

    def fit(self, x, y):
        '''Minimizes the cost function by adjusting the theta values with which we predict the data by using gradient descent'''
        print("Feeding data to model..")
        self.theta = []
        self.cost = []
        x = np.insert(x, 0, 1, axis=1)
        m = len(y)
        for category in np.unique(y):
            print('Descending the gradient for label type ' + str(category) + ' vs Rest')
            y_onevsall = np.where(y == category, 1, 0)
            if (self.random_weights):
                theta = np.random.rand(x.shape[1]) - 0.5
            else:
                theta = np.zeros(x.shape[1])
            cost = []
            thetas = []
            for _ in range(self.n_iter):
                z = x.dot(theta)
                h = self._sigmoid_function(z)
                theta = self._gradient_descent(x, h, theta, y_onevsall, m)
                cost.append(self._cost_function(h, y_onevsall))
                thetas.append(deepcopy(theta))
            self.theta.append((theta, category))
            self.weights.append((thetas, category))
            self.cost.append((cost, category))
        return self

    def predict(self, x):
        '''Takes the maximum probability of the models to predict the category'''
        x = np.insert(x, 0, 1, axis=1)
        x_predicted = [max((self._sigmoid_function(i.dot(theta)), c)
                           for theta, c in self.theta)[1] for i in x]
        return x_predicted

    def accuracy(self, x, y):
        '''Compares the predicted label with the actual label to find the model accuracy'''
        accuracy = sum(self.predict(x) == y) / len(y)
        return accuracy

    def plot_cost(self):
        '''Plot the cost function value for each category'''
        for cost, c in self.cost:
            plt.plot(range(len(cost)), cost, 'r')
            plt.title("Convergence Graph of Cost Function of type-" +
                      str(c) + " vs All")
            plt.xlabel("Number of Iterations")
            plt.ylabel("Cost")
            plt.show()
    
    def _plot_weights(self):
        '''Plot the weights after each iteration'''
        weights, c = self.weights[0]
        for weight in weights:
            plt.plot(range(len(weight)), weight, 'r')
            plt.title("weights over time of type: " +
                        str(c) + " vs All")
            plt.xlabel("features")
            plt.ylabel("Weight")
            plt.show()

    def save_model(self, filename='./datasets/weights.txt'):
        '''Write the weights to a file as a string representation of a list of tuples of numpy arrays and corresponding category as string'''
        with open(filename, 'w+', encoding='utf8') as f:
            f.write(str(self.theta))
