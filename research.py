# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 14:55:57 2018

@author: Simon
"""
# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# charge data
data = pd.read_csv("data_ft_linear_regression.csv")

# plot data
plt.figure()
plt.scatter(data.km, data.price)
plt.show()


# linear regression
class fit_lin_reg():
    '''
    This class is used to fit a linear regression using a gradient descent
    algorithm given a set of data
    '''

    def __init__(self, data, learning_rate, max_iter, precision):
        '''
        fit_lin_reg class constructor
        args:
            - data (pd.dataframe)
            - learning_rate (float)
            - max_iter (int) maximum number of iteration in the gradient descent
            - precision (float) minimum value of the loss in the gradient descent
        '''
        self.raw_data = data.copy()
        self.data = data
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.precision = precision

        self.theta0 = 0
        self.theta1 = 0

        self.nb_iter = 0
        self.loss = 1000
        self.theta0_ev = []
        self.theta1_ev = []
        self.loss_ev = []
        

    def estimate_price(self, mileage):
        '''
        Given the mileage of a car, return a prediction of price according to
        theta0 and theta1
        args:
            - mileage(float)
        '''
        _ = self.theta0 + mileage * self.theta1
        return _

    def stopping_criterion(self):
        '''
        Create a double stopping criterion, a maximum number of iteration and a
        precision to reach for the loss.
        '''
        boolean = (self.nb_iter < self.max_iter) & (self.loss > self.precision)
        return boolean
    
    def preprocess_data(self):
        self.data.km = (self.data.km - np.mean(self.data.km))/np.std(self.data.km)
        

    def train_model(self):
        """
        Train a linear regression model on the given data
        """
        self.preprocess_data()

        self.theta0_ev.append(self.theta0)
        self.theta1_ev.append(self.theta1)
        y_current = self.data.km.apply(self.EstimatePrice(), 1)
        self.loss = np.sum([elem**2 for elem in (self.data.price - y_current)])/ len(self.data)
        self.loss_ev.append(self.loss)

        while self.stopping_criterion():
            y_current = self.data.km.apply(self.EstimatePrice, 1)
            gradient_theta0 = self.learning_rate * (1/ len(self.data)) * np.sum(y_current - self.data.price)
            gradient_theta1 = self.learning_rate * (1/ len(self.data)) * np.sum(self.data.km * (y_current - self.data.price))
            self.theta0 = self.theta0 - gradient_theta0
            self.theta1 = self.theta1 - gradient_theta1

            self.nb_iter = self.nb_iter + 1
            self.loss = np.sum([elem**2 for elem in (self.data.price - y_current)])/ len(self.data)

            self.loss_ev.append(self.loss)
            self.theta0_ev.append(self.theta0)
            self.theta1_ev.append(self.theta1)
    
    def plot_train_info(self):
        _f = plt.figure()
        plt.suptitle("Evolution during iterations")
        plt.subplot(1, 3, 1)
        plt.plot(self.theta0_ev, label = "theta0")
        plt.legend()
        plt.subplot(1, 3, 2)
        plt.plot(self.theta1_ev, label = "theta1")
        plt.legend()
        plt.subplot(1, 3, 3)
        plt.plot(self.loss_ev, label = "loss")
        plt.legend()
        return _f
    
    def plot_result(self):
        _f = plt.figure()
        plt.scatter(self.raw_data.km, self.raw_data.price, label = "data")
        y_current = self.raw_data.km.apply(self.EstimatePrice, 1)
        plt.plot(self.raw_data.km, y_current, label = "predictions")
        plt.legend()
        return _f
        

