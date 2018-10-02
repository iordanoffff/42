# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 14:55:57 2018

@author: Simon
"""

class fit_lin_reg():
    '''
    This class is used to fit a linear regression using a gradient descent
    algorithm given a set of data
    '''
   
    def __init__(self, data, learning_rate):
        '''
        fit_lin_reg class constructor  
        args: 
            - data (pd.dataframe)
        '''
        self.data = data
        self.theta0 = 0
        self.theta1 = 0
        self.learning_rate = learning_rate


    def EstimatePrice(self):
        mileage = input("what is the mileage of your car? ")
        _ = self.theta0 + int(mileage) * self.theta1
        return _
    
    def TrainModel(self):
        

