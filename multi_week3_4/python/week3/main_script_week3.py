# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# PRE A : GO TO MAIN DIR FOR WEEK2, THAT IS WHERE MAIN_SCRIPT_WEEK2.PY IS
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import scipy.cluster.vq as cluster
import matplotlib.cm as cmx
from scipy import ndimage
from collections import defaultdict
import pickle
import math
import random
import sys
import os
import time

import week3
sys.path.insert(0, '../')
import tools

# Went to Random.org to determine seed from uniform interval [0, 99]
random.seed(26)

"""
###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PART 1. KMEANS
time.strftime("[%H:%M:%S] Computing part 1")

# GENERATE RANDOM DATA
x, labels = week3.generate_2d_data()

week3.plot_2d_data(x, labels, None, None)
plt.show()

# PART 1. STEP 0. PICK RANDOM CENTERS
time.strftime("[%H:%M:%S] Computing part 1, step 0")
week3.mykmeans(x, 3)
"""

"""
###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PART 2. COLOR BASED IMAGE SEGMENTATION

time.strftime("[%H:%M:%S] Computing part 2")
im = Image.open('../../data/coral.jpg')
imshow(im)
im = np.array(im)
im_flat = np.reshape(im, [im.shape[0] * im.shape[1], im.shape[2]])

N = 10000
im_flat_random = np.array(random.sample(im_flat, N))

K = 10
[codebook, dummy] = cluster.kmeans(... # RUN SCIPY KMEANS
[indexes, dummy] = cluster.vq(...      # VECTOR QUANTIZE PIXELS TO COLOR CENTERS

im_vq = codebook[indexes]
im_vq = np.reshape(im_vq, (im.shape))
im_vq = Image.fromarray(im_vq, 'RGB')

figure
subplot(1, 2, 1)
imshow(im)
subplot(1, 2, 2)
imshow(im_vq)
title('K=' + str(K))
"""

"""
###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PART 3. k-MEANS AND BAG-OF-WORDS
time.strftime("[%H:%M:%S] Computing part 3")
codebook = week3.load_codebook('../../data/codebook_100.pkl')
K = codebook.shape[0]
colors = week3.get_colors(K)

files = os.listdir('../../data/oxford_scaled/')

# PART 3. STEP 1. VISUALIZE WORDS ON IMAGES
time.strftime("[%H:%M:%S] Computing part 3, step 1")
word_patches = defaultdict(list)
patch_count = defaultdict(lambda: defaultdict(list))
files_random = random.sample(files, 5)

imfolder = '../../data/oxford_scaled/'
files = random.sample(os.listdir(imfolder), 5)
impaths = [imfolder + f for f in files]
for impath in impaths:
    frames, sift = week3.compute_sift(impath)      # COMPUTE SIFT
    [indexes, dummy] = cluster.vq(sift, codebook)  # VECTOR QUANTIZE SIFT TO WORDS

    # VISUALIZE WORDS
    patch_count[impath] = week3.show_words_on_image(impath,
                                                    K,
                                                    frames,
                                                    sift,
                                                    indexes,
                                                    colors,
                                                    patch_count[impath])
    patch_count[impath].default_factory = int
    for key in patch_count[impath]:
        word_patches[key] += patch_count[impath][key]
        patch_count[impath][key] = len(patch_count[impath][key])

# PART 3. STEP 2. PLOT COLORBAR
time.strftime("[%H:%M:%S] Computing part 3, step 2")
week3.get_colorbar(colors)

# PART 3. STEP 3. PLOT WORD CONTENTS
time.strftime("[%H:%M:%S] Computing part 3, step 3")
for k in random.sample(range(len(word_patches)), 5):
    WN = len(word_patches[k])
    plt.figure()
    plt.suptitle('Word ' + str(k))

    for i in range(WN):
        plt.subplot(int(math.ceil(np.sqrt(WN))),
                    int(math.ceil(np.sqrt(WN))),
                    i + 1)
        word_patch = word_patches[k][i].copy()
        plt.imshow(Image.fromarray(word_patch, 'RGB'))
        plt.axis('off')


# PART 4. BAG-OF-WORDS IMAGE REPRESENTATION
# USE THE np.bincount COUNTING THE INDEXES TO COMPUTE THE BAG-OF-WORDS REPRESENTATION,
time.strftime("[%H:%M:%S] Computing part 4")
for image in patch_count:
    max_key = max(patch_count[image])
    histogram = [patch_count[image][i] for i in range(max_key + 1)]
    plt.figure()
    plt.suptitle('Word histogram of image ' + image.split(os.sep)[-1])
    labels = np.arange(0, len(histogram), 10)
    ax = plt.axes()
    ax.set_xticks(labels)
    ax.set_ylabel("Word frequency")
    ax.set_xlabel("Word id")
    bars = plt.bar(np.arange(len(histogram)),
                   histogram, 1.0, color='gray', edgecolor='none')
    for bar, color in zip(bars, colors):
        bar.set_color(color)
"""

###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PART 5. PERFORM RETRIEVAL WITH THE BAG-OF-WORDS MODEL

# PART 5. STEP 1. LOAD BAG-OF-WORDS VECTORS FROM ../../data/bow/codebook_100/ using the week3.load_bow function

# PART 5. STEP 2. COMPUTE DISTANCE MATRIX

# PART 5. STEP 3. PERFORM RANKING SIMILAR TO WEEK 1 & 2 WITH QUERIES 'all_souls_000065.jpg', 'all_souls_0000XX.jpg', 'all_souls_0000XX.jpg'
query_id = ...
ranking = ...

# PART 5. STEP 4. COMPUTE THE PRECISION@5
files, labels, label_names = week3.get_oxford_filedata()
# ...
prec5 = week3.precision_at_N(0, gt_labels, ranking, 5)

# PART 5. STEP 4. IMPLEMENT & COMPUTE AVERAGE PRECISION

