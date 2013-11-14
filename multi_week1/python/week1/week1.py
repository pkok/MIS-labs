# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#           NAME: week1.py
#         AUTHOR: Stratis Gavves, Patrick de Kok, Georgios Methenitis
#  LAST MODIFIED: 14-10-2013
#    DESCRIPTION: TODO
#
#------------------------------------------------------------------------------
import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sgn


ROOT_OF_SQUARED_SUM = lambda h: math.sqrt(sum(map(lambda x: x * x, h)))
SUM = lambda h: float(sum(h))
DEFAULT_DIST_TYPE = 'euclidean'


def extractColorHistogram(im):
    histo = []
    pixels = float(im.shape[0] * im.shape[1])
    for channel in range(im.shape[-1]):
        histo.append(np.bincount(im[:, :, channel].flatten(), minlength=256)
                     / pixels)
    return np.concatenate(histo)


def d_euclidean(x, y):
    return -(x - y) ** 2


def d_l2(x, y):
    return (x * y)


d_intersect = min


def d_chi2(x, y):
    if x + y == 0:
        return 0
    return -((x - y) ** 2)


def d_helliger(x, y):
    return math.sqrt(x) * math.sqrt(y)


def computeVectorDistance(vec1, vec2, dist_type):
    if dist_type == 'euclidean':
        fn = d_euclidean
        normalize = ROOT_OF_SQUARED_SUM
    elif dist_type == 'l2':
        fn = d_l2
        normalize = ROOT_OF_SQUARED_SUM
    elif dist_type == 'intersect':
        fn = d_intersect
        normalize = SUM
    elif dist_type == 'chi2':
        fn = d_chi2
        normalize = SUM
    elif dist_type == 'hellinger':
        fn = d_helliger
        normalize = SUM
    return sum(map(fn, vec1 / normalize(vec1), vec2 / normalize(vec2)))


def test(id1, id2):
    print "test(%d, %d)" % (id1, id2)
    im1 = extractColorHistogram(get_image(id1))
    im2 = extractColorHistogram(get_image(id2))
    for key in ['euclidean', 'l2', 'intersect', 'chi2', 'hellinger']:
        print key, computeVectorDistance(im1, im2, key)


def get_image(id):
    return np.array(plt.imread('../../data/objects/flower/%d.jpg' % id))


def computeImageDistances(images):
    histo = [extractColorHistogram(np.array(plt.imread(image)))
             for image in images]
    return [[computeVectorDistance(histo1, histo2, DEFAULT_DIST_TYPE)
             for histo2 in histo]
            for histo1 in histo]


def rankImages(imdists, query_id):
    return sorted(range(len(imdists[query_id])),
                  key=lambda x: imdists[query_id][x],
                  reverse=True)


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
    G = G / sum(G)  # It is important to normalize with the total sum of G
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
