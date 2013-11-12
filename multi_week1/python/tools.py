# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#           NAME: week1.py
#         AUTHOR: Stratis Gavves
#  LAST MODIFIED: 18/03/10
#    DESCRIPTION: TODO
#
#------------------------------------------------------------------------------
import urllib
import os
import numpy as np
import zipfile
import math
import matplotlib.pyplot as plt
from PIL import Image

def plot_features(im, locs, circle=False, color='r'):

    def draw_circle(c, r):
        t = np.arange(0, 1.01, .01) * 2 * math.pi
        x = r*np.cos(t) + c[0]
        y = r*np.sin(t) + c[1]
        plt.plot(x, y, color, linewidth=2)
    
    if im != None:
        plt.imshow(im)
        
    plt.axis('off')
    plt.plot(locs[:, 0], locs[:, 1], '+'+color)
    if circle == True:
        for p in locs:
            draw_circle(p[:2], p[2])    
    
def append_images(im1, im2):
    rows1 = np.array(im1).shape[0]
    rows2 = np.array(im2).shape[0]
    if rows1 < rows2:
        im1 = np.concatenate((im1, np.zeros((rows2-rows1, im1.shape[1]))), axis=0)
    elif rows1 > rows2:
        im2 = np.concatenate((im2, np.zeros((rows1-rows2, im2.shape[1]))), axis=0)        
        
    return np.concatenate((im1, im2), axis=1)

def plot_matches(im1, im2, locs1, locs2, matches, color='r'):
    
    im_both = append_images(im1, im2)
    plt.imshow(im_both)
    plt.axis('off')
    
    cols1 = np.array(im1).shape[1]
    plot_features(None, locs1, 'r')
    plot_features(None, np.array((locs2[:, 0]+cols1, locs2[:, 1])).transpose(), 'g')
    
    for i, m in enumerate(matches):
        plt.plot([locs1[i][0], locs2[m][0]+cols1], [locs1[i][1], locs2[m][1]], 'b')

def normalizeColsL2(vectors):
    vectors = vectors / np.sqrt((vectors ** 2).sum(-1))[..., np.newaxis]
    return vectors

def compute_sift(impath):
    params = '--edge-thresh 10 --peak-thresh 5'
    im1 = Image.open(impath).convert('L')
    filpat1, filnam1, filext1 = fileparts(impath)
    temp_im1 = 'tmp_' + filnam1 + '.pgm'
    im1.save(temp_im1)

    sift_exec = '..\\..\\external\\vlfeat-0.9.17\\bin\\win64\\sift.exe'
    command = sift_exec + ' ' + os.getcwd() + '\\' + temp_im1 + ' --output ' + os.getcwd() + '\\' + filnam1 + '.sift.output' + ' ' + params
    os.system(command)
    frames, sift = read_sift_from_file(filnam1 + '.sift.output')
    #frames = frames.transpose()
    #sift = sift.transpose()
    
    return frames, sift

def read_sift_from_file(sift_path):
    f = np.loadtxt(sift_path)
    return f[:, :4], f[:, 4:]

def fileparts(file_path):
    filpat = os.path.abspath(file_path)
    filpat, dummy = os.path.split(filpat)
    filnam = os.path.basename(file_path)
    filnam, filext = os.path.splitext(filnam)
    return filpat, filnam, filext

def downloadObjectsDataset():
    url = 'http://staff.science.uva.nl/~gavves/files/objects.zip'
    print 'Downloading data from server. Please wait a bit...'
    urllib.urlretrieve(url, '../../data/objects.zip')
    print 'Data have been downloaded. Extracting them in "../../data/objects"'
    if not os.path.exists('../../data/objects'):
        os.mkdir('../../data/objects')
        
    zfile = zipfile.ZipFile('../../data/objects.zip').extractall('../../data/')
    os.remove('../../data/objects.zip')
        
def getImagePathsFromObjectsDataset( obj ):
    impaths = []
    for i in np.arange(1, 61):
        impaths.append('../../data/objects/' + obj + '/' + str(i) + '.jpg')
    
    return impaths
    
def gf_2d(sigma, H):
    sigma = float(sigma)
    acme = np.ceil(3 * sigma) # We want the filter to have a size at least 3σ
        
    x = np.arange(-acme, acme + 1)
    y = x
    xx, yy = np.meshgrid(x, y)
    rr = xx*xx + yy*yy
    G = 1/(2*math.pi*sigma)*np.exp(-1/(2*sigma*sigma) * rr)
    G = G / sum(G.flatten())
    G_ = np.zeros([H, H])
    G_[H/2-acme - 1 : H/2-acme+2*acme, H/2-acme - 1 : H/2-acme+2*acme] = G
    G = G_
    
    return G   
    