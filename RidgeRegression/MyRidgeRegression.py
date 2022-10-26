import numpy as np

class MyRidgeRegression:
    
    def __init__(self, lamda_val):
        
        self.lamda = lamda_val
    
    def fit(self, X, y):
        self.w_sol = np.linalg.inv((np.transpose(X)@X+self.lamda*np.identity(1)))@np.transpose(X)@y
                 
    def predict(self, X):
        return(X@self.w_sol)

