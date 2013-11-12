# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#           NAME: week1.py
#         AUTHOR: Stratis Gavves
#  LAST MODIFIED: 18/03/10
#    DESCRIPTION: TODO
#
#------------------------------------------------------------------------------
import numpy as np
import urllib
import os
import sys
import math
import scipy.signal as sgn

DEFAULT_DIST_TYPE = 'euclidean'


def extractColorHistogram( im ):
    histo = []
    for channel in range(im.shape[-1]):
        histo.append(np.bincount(im[:, :, channel].flatten(), minlength=256)) 
    return np.concatenate(histo)


def computeVectorDistance( vec1, vec2, dist_type ):
    if dist_type == 'euclidean' or dist_type == 'l2':
        fn = lambda x, y: (x - y) ** 2
    elif dist_type == 'intersect' or dist_type == 'l1':
        fn = min
    elif dist_type == 'chi2':
        fn = lambda x, y: ((x - y) ** 2) / (x + y)
    elif dist_type == 'hellinger':
        fn = lambda x, y: math.sqrt(x * y)

    return map(fn, vec1, vec2)


def computeImageDistances( images ):
    return [[week1.computeVectorDistance(histo1, histo2, DEFAULT_DIST_TYPE)
        for histo2 in histo]
        for histo1 in histo]
    

def rankImages( imdists, query_id ):
    reverse_order = DIST_TYPE in ['intersect', 'l1', 'helliger']
    return sorted(range(len(imdists[query_id])),
            key=lambda x: imdists[query_id][x],
            reverse=reverse_order)


def get_gaussian_filter(sigma):
    # PRE [DO NOT TOUCH]
    sigma = float(sigma)
    G = []
    
    # WRITE YOUR CODE HERE FOR DEFINING THE HALF SIZE OF THE FILTER
    half_size = 3 * sigma
    x = np.arange(-half_size, half_size + 1)        

    # WRITE YOUR CODE HERE
    G = np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi))
                        
    # RETURN [DO NOT TOUCH]
    G = G / sum(G) # It is important to normalize with the total sum of G
    return G
    

def get_gaussian_der_filter(sigma, order):
    # PRE [DO NOT TOUCH]
    sigma = float(sigma)
    dG = []
    
    # WRITE YOUR CODE HERE
    half_size = 3 * sigma
    x = np.arange(-half_size, half_size + 1)
    
    if order == 1:
        dG = -x / (sigma ** 2) * get_gaussian_filter(sigma)
    elif order == 2:
        raise NotImplementedError()
        # dG = ...

    # RETURN [DO NOT TOUCH]
    return dG

def gradmag(im_dr, im_dc):
    im_dmag = np.sqrt(im_dr * im_dr + im_dc * im_dc)
    return im_dmag / sum(im_dmag)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# [ALREADY IMPLEMENTED. DO NOT TOUCH]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def apply_filter(im, myfilter, dim):
    H, W = im.shape
    if dim == 'col':
        im_filt = sgn.convolve(im.flatten(), myfilter, 'same')
        im_filt = np.reshape(im_filt, [H, W])
    elif dim == 'row':
        im_filt = sgn.convolve(im.T.flatten(), myfilter, 'same')
        im_filt = np.reshape(im_filt, [W, H]).T
    
    return im_filt

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# [ALREADY IMPLEMENTED. DO NOT TOUCH]    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def apply_gaussian_conv(im, G):
    im_gfilt = apply_filter(im, G, 'col')
    im_gfilt = apply_filter(im_gfilt, G, 'row')
    
    return im_gfilt


        
        
    
