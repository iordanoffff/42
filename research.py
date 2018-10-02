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
        self.data = data
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.precision = precision

        self.theta0 = 0
        self.theta1 = 0

        self.nb_iter = 0
        self.loss = 1000

    def EstimatePrice(self, mileage):
        '''
        Given the mileage of a car, return a prediction of price according to
        theta0 and theta1
        args:
            - mileage(float)
        '''
        _ = self.theta0 + mileage * self.theta1
        return _

    def StopingCriterion(self):
        '''
        Create a double stopping criterion, a maximum number of iteration and a
        precision to reach for the loss.
        '''
        boolean = (self.nb_iter < self.max_iter) & (self.loss > self.precision)
        return boolean

    def TrainModel(self):
        """
        Train a linear regression model on the given data
        """
        while self.StopingCriterion():
            err = self.data.km.apply(self.EstimatePrice, 1) - self.data.price
            self.theta0 = self.theta0 - (self.learning_rate/len(self.data)) * np.sum(err)
            self.theta1 = self.theta1 - (self.learning_rate/len(self.data)) * np.sum(err * self.data.km)

            self.nb_iter = self.nb_iter + 1
            self.loss = np.mean((err)**2)
            print("theta0: {}".format(self.theta0))
            print("theta1: {}".format(self.theta1))
            print("loss: {}".format(self.loss))
