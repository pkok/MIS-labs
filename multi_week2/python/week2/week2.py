#------------------------------------------------------------------------------
#           NAME: week2.py
#         AUTHOR: Stratis Gavves
#  LAST MODIFIED: 18/03/10
#    DESCRIPTION: TODO
#
#------------------------------------------------------------------------------
import math
import os
import platform
import sys
try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
from PIL import Image
from pylab import *

sys.path.insert(0, '../')
import tools

CACHING = False
RENEW_CACHE = False
WRITTEN_TO_CACHE = False
CACHE_PATH = 'cache.pickle'

EMPTY_CACHE = {
    'match_sift': {},
    'compute_sift': {},
}

if CACHING and not RENEW_CACHE:
    try:
        with open(CACHE_PATH, 'r') as cache_file:
            try:
                cache = pickle.load(cache_file)
                for key, value in EMPTY_CACHE.items():
                    if key not in cache:
                        cache[key] = value
            except IOError:
                cache = EMPTY_CACHE
                os.remove(CACHE_PATH)
    except IOError:
        cache = EMPTY_CACHE
elif CACHING and RENEW_CACHE:
    cache = EMPTY_CACHE


def store_cache():
    global CACHING
    global WRITTEN_TO_CACHE
    if CACHING and WRITTEN_TO_CACHE:
        with open('cache.pickle', 'w') as cache_file:
            pickle.dump(cache, cache_file)
            WRITTEN_TO_CACHE = False


def arrayhash(array):
    writeable = array.flags.writeable
    array.flags.writeable = False
    h = hash(array.data)
    array.flags.writeable = writeable
    return h


##############################################################################
## YOUR IMPLEMENTATIONS
##############################################################################
def match_sift(sift1, sift2, dist_thresh=1.1):
    global CACHING, WRITTEN_TO_CACHE, RENEW_CACHE

    if CACHING:
        h1 = arrayhash(sift1)
        h2 = arrayhash(sift2)
        if (h1, h2) in cache['match_sift']:
            print "retrieving from cache: 'match_sift'"
            return cache['match_sift'][h1, h2]

    matches = -1 * np.ones((sift1.shape[0]), 'int')

    # FIRST NORMALIZE SIFT VECTORS
    sift1 = tools.normalizeL2(sift1, 0)
    sift2 = tools.normalizeL2(sift2, 0)

    # FOR ALL FEATURES IN SIFT1 FIND THE 2 CLOSEST FEATURES IN SIFT2
    # IF SIFT1_i IS MUCH CLOSER TO SIFT2_j1 THAN SIFT2_j2, THEN CONSIDER THIS
    # A MUCH MUCH CLOSER MEANS MORE THAN A PREDEFINED DISTANCE THRESHOLD
    dists = np.zeros(sift2.shape[0], 'float')
    ranked_ratio = float('infinity') * np.ones(sift1.shape[0], 'float')
    dist_l2 = lambda x, y: float(x * y)
    for i, s1 in enumerate(sift1):
        for j, s2 in enumerate(sift2):
            dists[j] = sum(map(dist_l2, s1, s2))
        closest_id = np.argmin(dists)
        closest = dists[closest_id]
        dists[closest_id] = float('infinity')
        second_closest = np.amin(dists)

        if closest / float(second_closest) < dist_thresh:
            matches[i] = closest_id
        ranked_ratio[i] = closest / float(second_closest)
        matches[i] = closest_id
    cache['match_sift'][arrayhash(sift1), arrayhash(sift2)] \
        = matches, ranked_ratio
    WRITTEN_TO_CACHE = True
    return matches, ranked_ratio


def project_point_via_homography(H, p):
    # INITIALIZATION [DO NOT TOUCH]
    newp = None
    # BRING POINT TO HOMOMORPHIC COORDINATES [DO NOT TOUCH]
    p = np.concatenate((p, [1]))

    # PUT YOUR CODE HERE
    newp = np.dot(H, p)

    # RE-NORMALIZATION BEFORE RETURNING THE POINTS [DO NOT TOUCH]
    newp = newp / newp[2]
    return newp


##############################################################################
### ALREADY IMPLEMENTED, DO NOT TOUCH
##############################################################################
def plot_matches(im1, im2, locs1, locs2, matches, color='r', newfig=True):
    im_both = append_images(im1, im2)
    if newfig is True:
        plt.figure()
        plt.imshow(Image.fromarray(np.uint8(im_both)))
        plt.axis('off')

    cols1 = np.array(im1).shape[1]
    plot_features(None, locs1, color='r')
    plot_features(None,
                  np.array((locs2[:, 0] + cols1, locs2[:, 1])).transpose(),
                  color='g')

    for i in range(0, locs1.shape[0]):
        if matches[i] != -1:
            m = matches[i]
            plt.plot([locs1[i][0], locs2[m][0] + cols1],
                     [locs1[i][1], locs2[m][1]], color)


def append_images(im1, im2):
    im1 = np.array(im1)
    im2 = np.array(im2)
    rows1 = np.array(im1).shape[0]
    rows2 = np.array(im2).shape[0]
    if rows1 < rows2:
        if len(im1.shape) == 3:
            padding = np.zeros((rows2-rows1, im1.shape[1], 3))
        else:
            padding = np.zeros((rows2-rows1, im1.shape[1]))

        im1 = np.concatenate((im1, padding), axis=0)
    elif rows1 > rows2:
        if len(im2.shape) == 3:
            padding = np.zeros((rows1-rows2, im2.shape[1], 3))
        else:
            padding = np.zeros((rows1-rows2, im2.shape[1]))

        im2 = np.concatenate((im2, padding), axis=0)

    return np.concatenate((im1, im2), axis=1)


def plot_features(im, locs, circle=False, color='r'):
    def draw_circle(c, r):
        t = np.arange(0, 1.01, .01) * 2 * math.pi
        x = r * np.cos(t) + c[0]
        y = r * np.sin(t) + c[1]
        plt.plot(x, y, color, linewidth=2)

    if im is not None:
        plt.imshow(im)

    plt.axis('off')
    plt.plot(locs[:, 0], locs[:, 1], '+' + color)
    if circle:
        for p in locs:
            draw_circle(p[:2], p[2])


def compute_sift(impath, edge_thresh=10, peak_thresh=5):
    global CACHING, WRITTEN_TO_CACHE, RENEW_CACHE

    params = ('--edge-thresh '
              + str(edge_thresh)
              + ' --peak-thresh '
              + str(peak_thresh))

    im1 = Image.open(impath).convert('L')

    if CACHING and not RENEW_CACHE:
        if im1 in cache['compute_sift']:
            print "retrieving from cache: 'compute_sift'"
            return cache['compute_sift'][hash(im1)]

    filpat1, filnam1, filext1 = tools.fileparts(impath)
    temp_im1 = 'tmp_' + filnam1 + '.pgm'
    im1.save(temp_im1)

    import struct
    is_64bit = struct.calcsize('P') * 8 == 64
    if platform.system() == 'Windows' and is_64bit:
        sift_exec = '..\\..\\external\\vlfeat-0.9.17\\bin\\win64\\sift.exe'
        command = sift_exec + ' \'' + os.getcwd() + '\\' + temp_im1 + '\' --output \'' + os.getcwd() + '\\' + filnam1 + '.sift.output' + '\' ' + params
    elif platform.system() == 'Windows' and not is_64bit:
        sift_exec = '..\\..\\external\\vlfeat-0.9.17\\bin\\win32\\sift.exe'
        command = sift_exec + ' \'' + os.getcwd() + '\\' + temp_im1 + '\' --output \'' + os.getcwd() + '\\' + filnam1 + '.sift.output' + '\' ' + params
    elif platform.system() == 'Linux':
        sift_exec = '..//..//external//vlfeat-0.9.17//bin//glnxa64//sift'
        command = sift_exec + ' \'' + os.getcwd() + '//' + temp_im1 + '\' --output \'' + os.getcwd() + '//' + filnam1 + '.sift.output' + '\' ' + params
    elif platform.system() == 'Darwin':
        sift_exec = '..//..//external//vlfeat-0.9.17//bin//maci64//sift'
        command = sift_exec + ' \'' + os.getcwd() + '//' + temp_im1 + '\' --output \'' + os.getcwd() + '//' + filnam1 + '.sift.output' + '\' ' + params

    os.system(command)
    frames, sift = read_sift_from_file(filnam1 + '.sift.output')
    os.remove(temp_im1)
    os.remove(filnam1 + '.sift.output')
    if CACHING:
        cache['compute_sift'][im1] = frames, sift
        WRITTEN_TO_CACHE = True
    return frames, sift


def read_sift_from_file(sift_path):
    f = np.loadtxt(sift_path)
    return f[:, :4], f[:, 4:]
