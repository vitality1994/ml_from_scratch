#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# My Codes
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings(action='ignore')
import numpy as np

def my_cross_val(model, loss_func, X, y, k):
    
    score_list = []
    
    for i in range(k):
        
        # Seperate data for training and validation

        target_hat = []

        if i < k-1:
        
            X_valid = X[i*int(X.shape[0]/k):(i+1)*int(X.shape[0]/k),:]
            X_train = np.delete(X, slice(i*int(X.shape[0]/k),(i+1)*int(X.shape[0]/k)), axis=0)       
            y_valid = y[i*int(X.shape[0]/k):(i+1)*int(X.shape[0]/k)]
            y_train = np.delete(y, slice(i*int(X.shape[0]/k),(i+1)*int(X.shape[0]/k)), axis=0)       
        
        else:
            X_valid = X[i*int(X.shape[0]/k):,:]
            X_train = np.delete(X, slice(i*int(X.shape[0]/k),), axis=0)       
            y_valid = y[i*int(X.shape[0]/k):]
            y_train = np.delete(y, slice(i*int(X.shape[0]/k),), axis=0)       
        
        
        # Training and validation using the model selected
        
        model_fitted = model
        model.fit(X_train, y_train)
        target_hat = model_fitted.predict(X_valid)
        
        

        # Calculating loss using either mse or error rate
        
        if loss_func == "mse":
            sum_error = 0

            for j in range(len(y_valid)):
                error = y_valid[j] - target_hat[j]
                sum_error += error**2
                mse = (1/len(y_valid))*sum_error
            
            result_loss = mse
        
        
        elif loss_func == "err_rate":
            count = 0
            for j in range(len(y_valid)):
                
                if y_valid[j] != target_hat[j]:
                    count+=1
                    err_rate = (1/len(y_valid))*count
            
            result_loss = err_rate
        
        # put values of loss function (k different values) in the list
        score_list.append(round(result_loss, 5))
    
    
    
    return(score_list)

