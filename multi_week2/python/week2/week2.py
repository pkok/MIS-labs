#------------------------------------------------------------------------------
#           NAME: week2.py
#         AUTHOR: Stratis Gavves
#  LAST MODIFIED: 18/03/10
#    DESCRIPTION: TODO
#
#------------------------------------------------------------------------------
import numpy as np
import math
import sys
import platform
from PIL import Image
from pylab import *
import os

sys.path.insert(0, '../')
import tools


##############################################################################
## YOUR IMPLEMENTATIONS
##############################################################################
def match_sift(sift1, sift2, dist_thresh=1.1):
    matches = -1 * np.ones((sift1.shape[0]), 'int')

    # FIRST NORMALIZE SIFT VECTORS
    # sift1 = tools.normalizeL...
    # sift2 = tools.normalizeL...

    # FOR ALL FEATURES IN SIFT1 FIND THE 2 CLOSEST FEATURES IN SIFT2

    # IF SIFT1_i IS MUCH CLOSER TO SIFT2_j1 THAN SIFT2_j2, THEN CONSIDER THIS
    # A MUCH MUCH CLOSER MEANS MORE THAN A PREDEFINED DISTANCE THRESHOLD

    return matches


def project_point_via_homography(H, p):
    # INITIALIZATION [DO NOT TOUCH]
    newp = None
    # BRING POINT TO HOMOMORPHIC COORDINATES [DO NOT TOUCH]
    p = np.concatenate((p, [1]))

    # PUT YOUR CODE HERE
    # newp = ...

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
    plot_features(None, locs1, 'r')
    plot_features(None,
                  np.array((locs2[:, 0] + cols1, locs2[:, 1])).transpose(),
                  'g')

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
    params = ('--edge-thresh '
              + str(edge_thresh)
              + ' --peak-thresh '
              + str(peak_thresh))

    im1 = Image.open(impath).convert('L')
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
    return frames, sift


def read_sift_from_file(sift_path):
    f = np.loadtxt(sift_path)
    return f[:, :4], f[:, 4:]
