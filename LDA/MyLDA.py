#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

class MyLDA:
    
    def __init__(self, lamda_val):
        self.lamda = lamda_val
    
    def fit(self, X, y):
        
        y_fixed = []
        for i in range(len(y)):
            y_fixed.append(int(y[i]))
            
        data = pd.concat([pd.DataFrame(X), pd.DataFrame(y_fixed)], axis=1)
        data.columns = ['feature1', 'feature2', 'class']
        mean_list=[]
        mean_list.append(data[data['class']==0]['feature1'].mean())
        mean_list.append(data[data['class']==0]['feature2'].mean())
        mean_list.append(data[data['class']==1]['feature1'].mean())
        mean_list.append(data[data['class']==1]['feature1'].mean())
     
        
        m_zero = np.array([mean_list[0], mean_list[1]])
        m_one = np.array([mean_list[2], mean_list[3]])
        
        temp_1 = 0
        for i in range(len(data[data['class']==0])):
            #print((data[data['class']==0].iloc[0]))
            #print((np.array(data[data['class']==0])))
            #print(data[data['class']==0].loc[i])
            
            
            temp_1+=(np.array(data[data['class']==0].iloc[i][0], data[data['class']==0].iloc[i][1])-m_zero)@np.transpose(np.array(data[data['class']==0].iloc[i][0], data[data['class']==0].iloc[i][1])-m_zero)

            
        temp_2 = 0
       
        for i in range(len(data[data['class']==1])):
            
            temp_2+=(np.array(data[data['class']==1].iloc[i][0], data[data['class']==1].iloc[i][1])-m_zero)@np.transpose(np.array(data[data['class']==1].iloc[i][0], data[data['class']==1].iloc[i][1])-m_zero)

            #temp_2+=(np.array(data[data['class']==1].loc[i+len(data[data['class']==0])][0], data[data['class']==1].loc[i+len(data[data['class']==0])][1]])-m_one)@np.transpose(np.array([data[data['class']==1].loc[i+999][0], data[data['class']==1].loc[i+999][1]])-m_one)  
    
        S_w = temp_1 + temp_2
        
        
        self.w_sol = (1/S_w)*(m_one - m_zero)

        
        
    def predict(self, X):
        projected_val = []
        for i in range(len(X)):
            
            projected_val.append(np.transpose(self.w_sol)@X[i])
            
            
        list_final_2 = []
        for i in range(len(projected_val)):
            if projected_val[i]>self.lamda:
                list_final_2.append(1)
            else:
                list_final_2.append(0)
        
        #plt.figure(figsize=(5, 2.5))
        #plt.hist(projected_val)
        #plt.show()
        
        return(list_final_2)

