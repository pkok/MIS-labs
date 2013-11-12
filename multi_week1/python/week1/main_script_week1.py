# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import scipy.signal as sgn
from PIL import Image
from scipy.ndimage import filters

import sys
sys.path.insert(0, '../')
import tools
import week1

# PREFERENCES FOR DISPLAYING ARRAYS. FEEL FREE TO CHANGE THE VALUES TO YOUR LIKING
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=200)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Step A. Download images [Already implemented]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cd "CHANGE THIS PATH TO YOUR WORKING DIRECTORY $main_dir/python/week1"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Step B. Basic image operations [Already implemented]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# B.1 Read image
im = array(imread('../../data/objects/flower/1.jpg'))

# B.2 Show image
imshow(im)
axis('off')

# B.3 Get image size
H, W, C = im.shape    # H for height, W for width, C for number of color channels
print 'Height: ' + str(H) + ', Width: ' + str(W) + ', Channels: ' + str(C)

# B.4 Access image pixel
print im[0, 0, 1]    # Single value in the 2 color dimension. Remember, numbering start from 0 (thus 1 means "2")
print im[0, 0]       # Vector of RGB values in all 3 color dimensions

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Step C. Compute image histograms [You should implement]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Compute color histogram from channel R
# C.1 Vectorize first the array
im_r = im[:, :, 0].flatten()

# C.2 Compute histogram from channel R using the bincount command, as indicated in the handout
# histo_r = np.bincount ...

# C.3 Compute now the histograms from the other channels, that is G and B
# im_g = ...
# histo_g = ...
# im_b = ...
# histo_b = ...

# C.4 Concatenate histograms from R, G, B one below the other into a single histogram
# histo = ...

######
# C.5 PUT YOUR CODE INTO THE FUNCTION extractColorHistogram( im ) IN week1.py
######

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Step D. Compute distances between vectors [You should implement]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# D.1 Open images and extract their RGB histograms
im1 = imread('../../data/objects/flower/1.jpg')
histo1 = week1.extractColorHistogram(im1)
im2 = imread('../../data/objects/flower/3.jpg')
histo2 = week1.extractColorHistogram(im2)

# D.2 Compute euclidean distance: d=Σ(x-y)^2 
# Note: the ***smaller*** the value, the more similar the histograms
# dist_euc = ...# WRITE YOUR CODE HERE

# D.3 Compute histogram intersection distance: d=Σmin(x, y)
# Note: the ***larger*** the value, the more similar the histograms
# dist_hi = ...# WRITE YOUR CODE HERE

# D.4 Compute chi-2 similarity: d= Σ(x-y)^2 / (x+y)
# Note: the ***larger*** the value, the more similar the histograms
# dist_chi2 = ...# WRITE YOUR CODE HERE

# D.5 Compute hellinger distance: d= Σsqrt(x*y)
# Note: the ***larger*** the value, the more similar the histograms
# dist_hell = ...# WRITE YOUR CODE HERE

######
# D.6 PUT YOUR CODE INTO THE FUNCTION computeVectorDistance( vec1, vec2, dist_type ) IN week1.py
######

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Step E. Rank images [You should implement]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# E.1 Compute histograms for all images in the dataset
impaths = tools.getImagePathsFromObjectsDataset('flower') # [ALREADY IMPLEMENTED]
# histo = ...# WRITE YOUR CODE HERE
    
# E.2 Compute distances between all images in the dataset
# imdists = ... # WRITE YOUR CODE HERE
    
# E.3 Given an image, rank all other images
query_id = rnd.randint(0, 59) # get a random image for a query
# sorted_id = ... # Here you should sort the images according to how distant they are

# E.4 Showing results. First image is the query, the rest are the top-5 most similar images [ALREADY IMPLEMENTED]
fig = plt.figure()
ax = fig.add_subplot(2, 3, 1)
im = imread(impaths[query_id])
ax.imshow(im)
ax.axis('off')
ax.set_title('Query image')

for i in np.arange(1, 1+5):
    ax = fig.add_subplot(2, 3, i+1)
    im = imread(impaths[sorted_id[i-1]]) # The 0th image is the query itself
    ax.imshow(im)
    ax.axis('off')
    ax.set_title(impaths[sorted_id[i-1]])

######
# E.5 PUT YOUR CODE INTO THE FUNCTIONS computeImageDistances( images )
#     AND rankImages( imdists, query_id ) IN week1.py
######
    
# F. Gaussian blurring using gaussian filter for convolution

# F.1 Open an image
im = array(Image.open('../../data/objects/flower/1.jpg').convert('L'))
imshow(im, cmap='gray') # To show as grayscale image

# F.2 Compute gaussian filter
sigma = 10.0
# G = get_gaussian_filter... # WRITE YOUR CODE HERE

# F.3 Apply gaussian convolution filter to the image. See the result. Compare with Python functionality
im_gf = week1.apply_gaussian_conv(im, G) # [ALREADY IMPLEMENTED, YOU ONLY NEED TO INPUT YOUR GAUSSIAN FILTER G]
im_gf2 = filters.gaussian_filter(im, sigma) # The result using Python functionality

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.imshow(im_gf, cmap='gray')
ax = fig.add_subplot(1, 2, 2)
ax.imshow(im_gf2, cmap='gray')

# F.4 Compute first order gaussian derivative filter in one dimension, row or column
# dG = get_gaussian_der_filter ... # WRITE YOUR CODE HERE

# Apply first on the row dimension
im_drow = week1.apply_filter(im, dG, 'row') # [ALREADY IMPLEMENTED, YOU ONLY NEED TO INPUT YOUR GAUSSIAN DERIVATIVE dG YOU JUST IMPLEMENTED]
# Apply then on the column dimension
im_dcol = week1.apply_filter(im, dG, 'col') # [ALREADY IMPLEMENTED, YOU ONLY NEED TO INPUT YOUR GAUSSIAN DERIVATIVE dG YOU JUST IMPLEMENTED]

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.imshow(im_drow, cmap='gray')
ax = fig.add_subplot(1, 2, 2)
ax.imshow(im_dcol, cmap='gray')

# F.6 Compute the magnitude and the orientation of the gradients of an image
# im_dmag = ...

fig = plt.figure()
imshow(im_dmag, cmap='gray')

######
# F.6.1 PUT YOUR CODE INTO THE FUNCTIONS get_gaussian_filter(sigma),
#       get_gaussian_der_filter(sigma, order) AND gradmag(im_drow, im_dcol) IN week1.py
######

# F.7 Apply gaussian filters on impulse image. HERE YOU JUST NEED TO USE THE CODE
#     YOU HAVE ALREADY IMPLEMENTED

# F.7.1 Create impulse image
imp = np.zeros([15, 15])
imp[6, 6] = 1
imshow(imp, cmap='gray')

# F.7.1 Compute gaussian filters
sigma = 1.0
G = week1.get_gaussian_filter(sigma) # BY NOW YOU SHOULD HAVE THIS FUNCTION IMPLEMENTED

fig = plt.figure()
plt.plot(G)
fig.suptitle('My gaussian filter') # HERE YOU SHOULD GET A BELL CURVE

# F.7.2 Apply gaussian filters
imp_gfilt = week1.apply_gaussian_conv(imp, G) # [ALREADY IMPLEMENTED, ADDED HERE ONLY FOR VISUALIZATION PURPOSES]

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
imshow(imp_gfilt, cmap='gray')
ax.set_title('Gaussian convolution: my implementation')
ax = fig.add_subplot(1, 2, 2)
imshow(tools.gf_2d(sigma, H), cmap='gray')
ax.set_title('Gaussian Kernel already provided')

# F.7.3 Apply first order derivative gradient filter
dG = week1.get_gaussian_der_filter(sigma, 1) # BY NOW YOU SHOULD HAVE THIS FUNCTION IMPLEMENTED
imp_drow = week1.apply_filter(imp, dG, 'row') # [ALREADY IMPLEMENTED]
imp_dcol = week1.apply_filter(imp, dG, 'col') # [ALREADY IMPLEMENTED]

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
imshow(imp_drow, cmap='gray')
ax = fig.add_subplot(1, 2, 2)
imshow(imp_dcol, cmap='gray')

