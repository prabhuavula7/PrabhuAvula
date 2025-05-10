#!/usr/bin/env python
# coding: utf-8

# # Anomaly Detection


import numpy as np
import matplotlib.pyplot as plt
from utils import *

get_ipython().run_line_magic('matplotlib', 'inline')

X_train, X_val, y_val = load_data()


# #### View the variables
print("The first 5 elements of X_train are:\n", X_train[:5])  


# In[4]:

# Display the first five elements of X_val
print("The first 5 elements of X_val are\n", X_val[:5])  


# In[5]:


# Display the first five elements of y_val
print("The first 5 elements of y_val are\n", y_val[:5])  


# #### Check the dimensions of your variables

# In[6]:


print ('The shape of X_train is:', X_train.shape)
print ('The shape of X_val is:', X_val.shape)
print ('The shape of y_val is: ', y_val.shape)


# #### Visualize your data

# In[7]:


# Create a scatter plot of the data.
plt.scatter(X_train[:, 0], X_train[:, 1], marker='x', c='b') 

# Set the title
plt.title("The first dataset")
# Set the y-axis label
plt.ylabel('Throughput (mb/s)')
# Set the x-axis label
plt.xlabel('Latency (ms)')
# Set axis range
plt.axis([0, 30, 0, 30])
plt.show()

#FUNCTION: estimate_gaussian

def estimate_gaussian(X): 
    """
    Calculates mean and variance of all features 
    in the dataset
    
    Args:
        X (ndarray): (m, n) Data matrix
    
    Returns:
        mu (ndarray): (n,) Mean of all features
        var (ndarray): (n,) Variance of all features
    """

    m, n = X.shape 
    mu = np.mean(X, axis=0)  
    var = np.var(X, axis=0)
        
    return mu, var

#check if your implementation is correct by running the following test code:

# In[9]:

# Estimate mean and variance of each feature
mu, var = estimate_gaussian(X_train)              

print("Mean of each feature:", mu)
print("Variance of each feature:", var)
    
# UNIT TEST
from public_tests import *
estimate_gaussian_test(estimate_gaussian)

# In[10]:


#Returns the density of the multivariate normal at each data point (row) of X_train
p = multivariate_gaussian(X_train, mu, var)

#Plotting code 
visualize_fit(X_train, mu, var)

#FUNCTION: select_threshold

def select_threshold(y_val, p_val): 
    """
    Finds the best threshold to use for selecting outliers 
    based on the results from a validation set (p_val) 
    and the ground truth (y_val)
    
    Args:
        y_val (ndarray): Ground truth on validation set
        p_val (ndarray): Results on validation set
        
    Returns:
        epsilon (float): Threshold chosen 
        F1 (float):      F1 score by choosing epsilon as threshold
    """ 

    best_epsilon = 0
    best_F1 = 0
    F1 = 0
    
    step_size = (max(p_val) - min(p_val)) / 1000
    
    for epsilon in np.arange(min(p_val), max(p_val), step_size):
    
        predictions = (p_val < epsilon)  # Classify anomalies (True for anomalies, False otherwise)
        
        tp = np.sum((predictions == 1) & (y_val == 1))  # True Positives
        fp = np.sum((predictions == 1) & (y_val == 0))  # False Positives
        fn = np.sum((predictions == 0) & (y_val == 1))  # False Negatives
        
        if tp + fp == 0 or tp + fn == 0:
            continue  # Avoid division by zero

        precision = tp / (tp + fp)  # Precision calculation
        recall = tp / (tp + fn)  # Recall calculation

        F1 = (2 * precision * recall) / (precision + recall)
        
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
        
    return best_epsilon, best_F1

# Check your implementation using the code below

# In[14]:


p_val = multivariate_gaussian(X_val, mu, var)
epsilon, F1 = select_threshold(y_val, p_val)

print('Best epsilon found using cross-validation: %e' % epsilon)
print('Best F1 on Cross Validation Set: %f' % F1)
    
# UNIT TEST
select_threshold_test(select_threshold)

# In[15]:


# Find the outliers in the training set 
outliers = p < epsilon

# Visualize the fit
visualize_fit(X_train, mu, var)

# Draw a red circle around those outliers
plt.plot(X_train[outliers, 0], X_train[outliers, 1], 'ro',
         markersize= 10,markerfacecolor='none', markeredgewidth=2)

# High-dimensional dataset
# Now,  run the anomaly detection algorithm that you implemented on a more realistic and much more challenging dataset. 
# In this dataset, each example is described by 11 features, capturing many more properties of your compute servers.
# In[16]:


# load the dataset
X_train_high, X_val_high, y_val_high = load_data_multi()


#Check the dimensions of your variables
# In[17]:


print ('The shape of X_train_high is:', X_train_high.shape)
print ('The shape of X_val_high is:', X_val_high.shape)
print ('The shape of y_val_high is: ', y_val_high.shape)


#Anomaly detection 
# Now, let's run the anomaly detection algorithm on this new dataset.

# In[18]:


# Apply the same steps to the larger dataset

# Estimate the Gaussian parameters
mu_high, var_high = estimate_gaussian(X_train_high)

# Evaluate the probabilites for the training set
p_high = multivariate_gaussian(X_train_high, mu_high, var_high)

# Evaluate the probabilites for the cross validation set
p_val_high = multivariate_gaussian(X_val_high, mu_high, var_high)

# Find the best threshold
epsilon_high, F1_high = select_threshold(y_val_high, p_val_high)

print('Best epsilon found using cross-validation: %e'% epsilon_high)
print('Best F1 on Cross Validation Set:  %f'% F1_high)
print('# Anomalies found: %d'% sum(p_high < epsilon_high))
