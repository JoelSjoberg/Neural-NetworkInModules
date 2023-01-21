#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cupy as cp
from activations import get_derivative


# In[5]:


class MSE:
    def __init__(self):
        self.y_t = None
        self.batch_y_t = None
    
    def set_yt(self, y_t):
        self.y_t = y_t
    
    def J(self, y_p):

        error = cp.square(self.y_t - y_p) * 1/(2 * self.batch_y_p.size)
        return cp.nan_to_num(error)
        
    def compute(self, y_p, y_t):
        self.y_t = y_t
            
        return self.J(y_p)
    
    def derivative(self, x):
        x = x.ravel()
        return get_derivative(self.J, x)
            
    def set_latent_points(self, points):
        pass


# In[ ]:




