#!/usr/bin/env python
# coding: utf-8

# K-means Clustering 
# Implement the K-means algorithm and use it for image compression. 
# Using the K-means algorithm for image compression by reducing the number of colors in an image to only those most common.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from utils import *

get_ipython().run_line_magic('matplotlib', 'inline')

# This function takes the data matrix `X` and the locations of all centroids inside `centroids` 
# t should output a one-dimensional array `idx` (which has the same number of elements as `X`) that holds the index  of the closest centroid (a value in $\{0,...,K-1\}$, where $K$ is total number of centroids) to every training example . *(Note: The index range 0 to K-1 varies slightly from what is shown in the lectures (i.e. 1 to K) because Python list indices start at 0 instead of 1)*
# find_closest_centroids

def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example
    
    Args:
        X (ndarray): (m, n) Input values      
        centroids (ndarray): (K, n) centroids
    
    Returns:
        idx (array_like): (m,) closest centroids
    
    """

    # Set K
    K = centroids.shape[0]

    # You need to return the following variables correctly
    idx = np.zeros(X.shape[0], dtype=int)

    for i in range(X.shape[0]):
        # Compute the squared Euclidean distance between the ith data point and each centroid
        distances = np.sum((X[i] - centroids) ** 2, axis=1)
        
        # Find the index of the minimum distance
        idx[i] = np.argmin(distances)
    
    return idx

# Check your implementation using an example dataset

# In[3]:
# Load an example dataset that we will be using
X = load_data()

# In[4]:


print("First five elements of X are:\n", X[:5]) 
print('The shape of X is:', X.shape)


# In[5]:


# Select an initial set of centroids (3 Centroids)
initial_centroids = np.array([[3,3], [6,2], [8,5]])

# Find closest centroids using initial_centroids
idx = find_closest_centroids(X, initial_centroids)

# Print closest centroids for the first three elements
print("First three elements in idx are:", idx[:3])

# UNIT TEST
from public_tests import *

find_closest_centroids_test(find_closest_centroids)

# the `compute_centroids` below to recompute the value for each centroid
# compute_centroids

def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the 
    data points assigned to each centroid.
    
    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each 
                       example in X. Concretely, idx[i] contains the index of 
                       the centroid closest to example i
        K (int):       number of centroids
    
    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """
    
    # Useful variables
    m, n = X.shape
    
    # You need to return the following variables correctly
    centroids = np.zeros((K, n))
    
    for k in range(K):
        # Get all points assigned to centroid k
        points = X[idx == k]
        
        # Compute the mean of those points
        if len(points) > 0:
            centroids[k] = np.mean(points, axis=0)
    
    return centroids


# Check your implementation by running the cell below

# In[7]:


K = 3
centroids = compute_centroids(X, idx, K)

print("The centroids are:", centroids)

# UNIT TEST
compute_centroids_test(compute_centroids)

# In[8]:

def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """
    
    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids    
    idx = np.zeros(m)
    plt.figure(figsize=(8, 6))

    # Run K-Means
    for i in range(max_iters):
        
        #Output progress
        print("K-Means iteration %d/%d" % (i, max_iters-1))
        
        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)
        
        # plot progress
        if plot_progress:
            plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            
        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)
    plt.show() 
    return centroids, idx


# In[9]:

# Load an example dataset
X = load_data()

# Set initial centroids
initial_centroids = np.array([[3,3],[6,2],[8,5]])

# Number of iterations
max_iters = 10

# Run K-Means
centroids, idx = run_kMeans(X, initial_centroids, max_iters, plot_progress=True)

# Random initialization
# In[10]:

def kMeans_init_centroids(X, K):
    """
    This function initializes K centroids that are to be 
    used in K-Means on the dataset X
    
    Args:
        X (ndarray): Data points 
        K (int):     number of centroids/clusters
    
    Returns:
        centroids (ndarray): Initialized centroids
    """
    
    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    
    # Take the first K examples as centroids
    centroids = X[randidx[:K]]
    
    return centroids


# Rerun K-Means, but this time with random initial centroids. Run the cell below several times and observe how different clusters are created based on the initial points chosen.

# In[11]:
# Run this cell repeatedly to see different outcomes.

# Set number of centroids and max number of iterations
K = 3
max_iters = 10

# Set initial centroids by picking random examples from the dataset
initial_centroids = kMeans_init_centroids(X, K)

# Run K-Means
centroids, idx = run_kMeans(X, initial_centroids, max_iters, plot_progress=True)

# Image compression with K-means
# In[12]:


# Load an image of a bird
original_img = plt.imread('bird_small.png')

# In[13]:


# Visualizing the image
plt.imshow(original_img)

# In[14]:


print("Shape of original_img is:", original_img.shape)

# In[15]:

# Divide by 255 so that all values are in the range 0 - 1

original_img = original_img / 255

# Reshape the image into an m x 3 matrix where m = number of pixels
# (in this case m = 128 x 128 = 16384)
# Each row will contain the Red, Green, and Blue pixel values
# This gives us our dataset matrix X_img, on which we will use K-Means.

X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))

# Now, run K-Means on the pre-processed image.

# In[16]:

K = 16
max_iters = 10
 
initial_centroids = kMeans_init_centroids(X_img, K)

centroids, idx = run_kMeans(X_img, initial_centroids, max_iters)


# In[17]:


print("Shape of idx:", idx.shape)
print("Closest centroid for the first five elements:", idx[:5])

# In[18]:


# Plot the colors of the image and mark the centroids
plot_kMeans_RGB(X_img, centroids, idx, K)

# In[19]:


# Visualize the colors selected
show_centroid_colors(centroids)


# Compress the image
# In[20]:


# Find the closest centroid of each pixel
idx = find_closest_centroids(X_img, centroids)

# Replace each pixel with the color of the closest centroid
X_recovered = centroids[idx, :] 

# Reshape image into proper dimensions
X_recovered = np.reshape(X_recovered, original_img.shape) 

# In[21]:


# Display original image
fig, ax = plt.subplots(1,2, figsize=(16,16))
plt.axis('off')

ax[0].imshow(original_img)
ax[0].set_title('Original')
ax[0].set_axis_off()


# Display compressed image
ax[1].imshow(X_recovered)
ax[1].set_title('Compressed with %d colours'%K)
ax[1].set_axis_off()
