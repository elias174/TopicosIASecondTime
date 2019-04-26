# porcentaje de ejemplos clasificados correctamente
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LogisticRegression(object):
    def __init__(self, train_set, test_set, iterations):
        self.train_set = train_set
        self.test_set = test_set

        self.X_train = self.train_set.iloc[:,self.train_set.columns != 'Clase']
        self.y_train = self.train_set.iloc[:,-1]

        self.X_test = self.test_set.iloc[:,self.test_set.columns != 'Clase']
        self.y_test = self.test_set.iloc[:,-1]

        self.iterations = iterations
        self.cost_history = np.zeros(iterations)
        self.theta = None

    def normalize(self):
        X_media = np.concatenate((self.X_train, self.X_test), axis=0).mean(axis=0)
        X_std = np.concatenate((self.X_train, self.X_test), axis=0).std(axis=0)
        y_media = np.concatenate((self.y_train, self.y_test), axis=0).mean()
        y_std = np.concatenate((self.y_train, self.y_test), axis=0).std()
        #
        self.X_train = (self.X_train - X_media) / X_std
        # self.y_train = (self.y_train - y_media) / y_std
        self.X_test = (self.X_test - X_media) / X_std
        # self.y_test = (self.y_test - y_media) / y_std
        #
        self.X_train = np.concatenate((np.ones([len(self.y_train), 1]), self.X_train), axis=1)
        self.X_test = np.concatenate((np.ones([len(self.y_test), 1]), self.X_test), axis=1)

        self.n_features = self.X_train.shape[1]
        self.theta = np.zeros([self.X_train.shape[1], ])

    def calculate_cost(self, theta, X, y):
        y_pred_test = self.hyphotesis(theta, X)
        first_to_sum = y * np.log(y_pred_test)
        second_to_sum = (1-y) * np.log(1-y_pred_test)
        to_sum = first_to_sum + second_to_sum
        return -(1/(X.shape[0])) * np.sum(to_sum)

    def train(self, alpha):
        for it in range(self.iterations):
            H_theta = self.hyphotesis(self.theta, self.X_train)
            self.theta = self.theta - alpha * (
                self.X_train.T.dot((H_theta - self.y_train)))
            self.cost_history[it] = self.calculate_cost(self.theta, self.X_train, self.y_train)

    def plot_cost_history(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_ylabel('J(Theta)')
        ax.set_xlabel('Iterations')
        _ = ax.plot(range(self.iterations), self.cost_history, 'g.')
        plt.show()

    def get_error(self):
        return self.calculate_cost(
            self.theta,
            self.X_test,
            self.y_test
        )

    def accuracy(self):
        predictions = self.hyphotesis(self.theta, self.X_test)
        asserts = 0
        for prediction, y_value in zip(predictions, self.y_test):
            value_prediction = 1 if prediction > 0.5 else 0
            if y_value == value_prediction:
                asserts += 1

        print(asserts/self.y_test.shape[0])

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, theta, X):
        return np.dot(X, theta)

    def hyphotesis(self, theta, X):
        return self.sigmoid(self.predict(theta, X))

    def plot_elements(self, X_, y_):
        one_type = X_[y_ == 0]
        second_type = X_[y_ == 1]
        plt.scatter(one_type[:, 1], one_type[:, 2], s=10)
        plt.scatter(second_type[:, 1], second_type[:, 2], s=10)

        x_values = [np.min(X_[:, 1] - 2), np.max(X_[:, 2] + 2)]
        y_values = - (self.theta[0] + np.dot(self.theta[1], x_values)) / self.theta[2]
        plt.plot(x_values, y_values, label='Decision Boundary')
        #
        plt.show()

    def plot_train_elements(self):
        self.plot_elements(self.X_train, self.y_train)

    def plot_test_elements(self):
        self.plot_elements(self.X_test, self.y_test)

def execute():
    train = pd.read_csv('csv_regression/Train_ex2data1.csv')
    test = pd.read_csv('csv_regression/Test_ex2data1.csv')

    logistic_regression = LogisticRegression(train, test, 1000)
    logistic_regression.normalize()
    logistic_regression.train(0.01)
    logistic_regression.accuracy()
    #logistic_regression.plot_test_elements()
    # print(logistic_regression.get_error())
    # logistic_regression.plot_cost_history()


execute()