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
print time.strftime("[%H:%M:%S] Computing part 1")

# GENERATE RANDOM DATA
x, labels = week3.generate_2d_data()

week3.plot_2d_data(x, labels, None, None)
plt.show()

# PART 1. STEP 0. PICK RANDOM CENTERS
print time.strftime("[%H:%M:%S] Computing part 1, step 0")
week3.mykmeans(x, 3)
"""

"""
###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PART 2. COLOR BASED IMAGE SEGMENTATION

print time.strftime("[%H:%M:%S] Computing part 2")
im = Image.open('../../data/coral.jpg')
plt.imshow(im)
im = np.array(im)
im_flat = np.reshape(im, [im.shape[0] * im.shape[1], im.shape[2]])

N = 1000
im_flat_random = np.array(random.sample(im_flat, N))

K = 10
[codebook, dummy] = cluster.kmeans(im_flat, K, iter = 100)  # RUN SCIPY KMEANS
[indexes, dummy] = cluster.vq(im_flat,  codebook)    # VECTOR QUANTIZE PIXELS TO COLOR CENTERS

im_vq = codebook[indexes]
im_vq = np.reshape(im_vq, (im.shape))
im_vq = Image.fromarray(im_vq, 'RGB')

#figure
plt.subplot(1, 2, 1)
plt.imshow(im)
plt.subplot(1, 2, 2)
plt.imshow(im_vq)
plt.title('K=' + str(K))

plt.show()
"""


"""
###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PART 3. k-MEANS AND BAG-OF-WORDS
# Went to Random.org to determine seed from uniform interval [0, 99]
random.seed(17)
print time.strftime("[%H:%M:%S] Computing part 3")
codebook = week3.load_codebook('../../data/codebook_100.pkl')
K = codebook.shape[0]
colors = week3.get_colors(K)

files = os.listdir('../../data/oxford_scaled/')

# PART 3. STEP 1. VISUALIZE WORDS ON IMAGES
print time.strftime("[%H:%M:%S] Computing part 3, step 1")
word_patches = defaultdict(list)
patch_count = defaultdict(lambda: defaultdict(list))
files_random = random.sample(files, 5)
print 'Random files: %s' % files_random

imfolder = '../../data/oxford_scaled/'
impaths = [imfolder + f for f in files_random]
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
print time.strftime("[%H:%M:%S] Computing part 3, step 2")
week3.get_colorbar(colors)

# PART 3. STEP 3. PLOT WORD CONTENTS
print time.strftime("[%H:%M:%S] Computing part 3, step 3")
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
"""

"""
# PART 4. BAG-OF-WORDS IMAGE REPRESENTATION
# USE THE np.bincount COUNTING THE INDEXES TO COMPUTE THE BAG-OF-WORDS REPRESENTATION,
print time.strftime("[%H:%M:%S] Computing part 4")
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
print time.strftime("[%H:%M:%S] Computing part 5")

# PART 5. STEP 1. LOAD BAG-OF-WORDS VECTORS FROM ../../data/bow/codebook_100/ using the week3.load_bow function
print time.strftime("[%H:%M:%S] Computing part 5, part 1")
DISTANCE_L2 = {'name': 'l2',
               'fn': lambda x, y: float(x * y),
               'norm': tools.normalizeL2}
DISTANCE_HIST = {'name': 'histogram',
                 'fn': min,
                 'norm': tools.normalizeL1}
DISTANCE_MEASURES = [DISTANCE_L2, DISTANCE_HIST]
BOW_SIZES = (10, 50, 100, 500, 1000)
BOW_FOLDER = "../../data/bow/codebook_%d/"
normalized_bows = {}
for bow_size in BOW_SIZES:
    normalized_bows[bow_size] = week3.bow_histograms(BOW_FOLDER % bow_size,
                                                     DISTANCE_MEASURES)

# PART 5. STEP 2. COMPUTE DISTANCE MATRIX
print time.strftime("[%H:%M:%S] Computing part 5, part 2")
dist_matrices = {}
for bow_size, normalized_bow in normalized_bows.items():
    dist_matrices[bow_size] = week3.distance_matrices(normalized_bow,
                                                      DISTANCE_MEASURES)
dist_hist = dist_matrices[100]['histogram']
dist_l2 = dist_matrices[100]['l2']

# PART 5. STEP 3. PERFORM RANKING SIMILAR TO WEEK 1 & 2 WITH QUERIES 'all_souls_000065.jpg', 'all_souls_0000XX.jpg', 'all_souls_0000XX.jpg'
print time.strftime("[%H:%M:%S] Computing part 5, part 3")

ranking_hist = {}
ranking_l2 = {}
QUERIES = (
    'all_souls_000006.jpg',
    'all_souls_000065.jpg',
    'all_souls_000075.jpg'
)
AVERAGE_PRECISION = ''
QUERY_PLOT_SYMBOLS = dict(zip(QUERIES, ('ro', 'gs', 'b^')))
QUERY_PLOT_SYMBOLS[AVERAGE_PRECISION] = 'k+'

ranking = {}
for query in QUERIES:
    ranking[query] = {}
    for bow_size in BOW_SIZES:
        ranking[query][bow_size] = {}
        for dist_measure in dist_matrices[bow_size]:
            ranking[query][bow_size][dist_measure] =\
                week3.rank_images(dist_matrices[bow_size][dist_measure], query)


IMG_FOLDER = '../../data/oxford_scaled/'
for query in QUERIES:
    ranking_hist[query] = ranking[query][100]['histogram']
    ranking_l2[query] = ranking[query][100]['l2']
    im = plt.imread(IMG_FOLDER + query)
    figure_h = plt.figure()
    figure_h.suptitle("Ranking images based on histogram distance")
    ax_h = figure_h.add_subplot(2, 3, 1)
    ax_h.imshow(im)
    ax_h.axis('off')
    ax_h.set_title("Query image")
    figure_l = plt.figure()
    figure_l.suptitle("Ranking images based on L2 distance")
    ax_l = figure_l.add_subplot(2, 3, 1)
    ax_l.imshow(im)
    ax_l.axis('off')
    ax_l.set_title("Query image")

    for i in range(5):
        im = plt.imread(IMG_FOLDER + ranking_hist[query][i])
        ax = figure_h.add_subplot(2, 3, i + 2)
        ax.imshow(im)
        ax.axis('off')
        ax.set_title(ranking_hist[query][i])

        im = plt.imread(IMG_FOLDER + ranking_l2[query][i])
        ax = figure_l.add_subplot(2, 3, i + 2)
        ax.imshow(im)
        ax.axis('off')
        ax.set_title(ranking_l2[query][i])


# PART 5. STEP 4. COMPUTE THE PRECISION@5
print time.strftime("[%H:%M:%S] Computing part 5, part 4")
files, labels, label_names = week3.get_oxford_filedata()
indexed_labels = dict(zip(files, labels))
precision = {}
PRECISION_LIMITS = (5, 10)
for n in PRECISION_LIMITS:
    precision[n] = {}
    for d_ in DISTANCE_MEASURES:
        d = d_['name']
        precision[n][d] = {}
        fig = plt.figure()
        fig.suptitle("Precision@%d with %s measure" % (n, d))
        precision[n][d][AVERAGE_PRECISION] = {}
        for s in BOW_SIZES:
            precision[n][d][AVERAGE_PRECISION][s] = 0.
        for q in QUERIES:
            precision[n][d][q] = {}
            for s in BOW_SIZES:
                precision[n][d][q][s] =\
                    week3.custom_precision_at_N(q,
                                                indexed_labels,
                                                ranking[q][s][d],
                                                n)
                precision[n][d][AVERAGE_PRECISION][s] +=\
                    precision[n][d][q][s]
            plot_values = zip(*precision[n][d][q].items())
            plt.plot(plot_values[0], plot_values[1],
                     QUERY_PLOT_SYMBOLS[q], label=q)
        for s in precision[n][d][AVERAGE_PRECISION]:
            precision[n][d][AVERAGE_PRECISION][s] /= len(QUERIES)
        plot_values = zip(*precision[n][d][AVERAGE_PRECISION].items())
        plt.plot(plot_values[0], plot_values[1],
                 QUERY_PLOT_SYMBOLS[AVERAGE_PRECISION], label='average')
        plt.legend(numpoints=1)

# PART 5. STEP 4. IMPLEMENT & COMPUTE AVERAGE PRECISION
print time.strftime("[%H:%M:%S] Done.")
plt.show()
