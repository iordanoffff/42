# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 14:55:57 2018

@author: Simon
"""
import numpy as np
import pandas as pd

class fit_lin_reg():
    '''
    This class is used to fit a linear regression using a gradient descent
    algorithm given a set of data
    '''
   
    def __init__(self, data, learning_rate, x, y):
        '''
        fit_lin_reg class constructor  
        args: 
            - data (pd.dataframe)
            - learning_rate (float)
            - x (str) name of the x column
            - y (str) name of the y column
        '''
        self.data = data
        self.theta0 = 0
        self.theta1 = 0
        self.x = x
        self.y = y
        self.learning_rate = learning_rate


    def EstimatePrice(self, mileage):
        '''
        Given the mileage of a car, return a prediction of price according to
        theta0 and theta1
        args:
            - mileage(float)
        '''
        _ = self.theta0 + mileage * self.theta1
        return _
    
    def TrainModel(self):
        
        err_abs = getattr(self.data, self.x).apply(self.EstimatePrice, 1) - getattr(self.data, self.y)
        print(err_abs)
        self.theta0 = (self.learning_rate/len(self.data)) * np.sum( err_abs)
        self.theta1 = (self.learning_rate/len(self.data)) * np.sum(err_abs * getattr(self.data, self.y))
        
        
        
        
        
        
        
        
        
        

