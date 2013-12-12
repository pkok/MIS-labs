import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import scipy as sc
import scipy.cluster.vq as cluster
import sys
import random
import os
import matplotlib.cm as cmx
import pickle
from collections import defaultdict
import math
sys.path.insert(0, '../')
import tools
import math
import pylab as pl
from sklearn import svm, datasets

import week56

random.seed(0)

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=200)

####
# WORKING CODE
#"""
files, labels, label_names, unique_labels, trainset, testset = week56.get_objects_filedata()

C = 100
bow = np.zeros([len(files), C])
cnt = -1
for impath in files:
    cnt = cnt + 1
    print str(cnt) + '/' + str(len(files)) + '): ' + impath
    filpat, filnam, filext = tools.fileparts(impath)
    filpat2, filnam2, filext2 = tools.fileparts(filpat)
    bow[cnt, :] = week56.load_bow('../../data/bow_objects/codebook_' + str(C) + '/' + filnam2 + '/' + filnam + '.pkl')

# Q1: IMPLEMENT HERE kNN CLASSIFIER.
# YOU CAN USE CODE FROM PREVIOUS WEEK
dist = np.zeros([len(testset), len(trainset)])
for i in range(len(testset)):
    b_i = tools.normalizeL1(bow[testset[i], :])
    for j in range(len(trainset)):
        b_j = tools.normalizeL1(bow[trainset[j], :])
        dist[i, j] = sum(np.minimum(b_i, b_j))

K = 9
QUERIES = {
    366: "goal/15",
    150: "bicycle/37",
    84: "beach/31",
    450: "mountain/37"
}
accuracy_1 = 0
accuracy_2 = 0

query_set = testset
query_set = QUERIES
for query_id in query_set:
    testset_id = np.argwhere(testset == query_id)[0, 0]
    ranking = np.argsort(dist[testset_id, :])
    ranking = ranking[::-1]
    nearest_labels = labels[trainset[ranking[:K]]]
    true_label = label_names[query_id]

    # Label predicted and accuracy according to method of Q1
    predicted_label1 = unique_labels[np.argmax(np.bincount(nearest_labels)) - 1]
    accuracy_1 += predicted_label1 == true_label

# Q2: USE DIFFERENT STRATEGY
#   Only ranking differs from Q1
    # Label predicted and accuracy according to method of Q2
    weighted_votes = np.zeros(len(unique_labels))
    for rank, ranked in enumerate(ranking[:K]):
        #weighted_votes[labels[trainset[ranked]]-1] += 1 - ((1. + rank) / (2 + rank))
        weighted_votes[labels[trainset[ranked]]-1] += ((1.) / (1 + rank))
    predicted_label2 = unique_labels[np.argmax(weighted_votes)]
    accuracy_2 += predicted_label2 == true_label

    # VISUALIZE RESULTS
    plt.figure()
    plt.subplot(2, 5, 1)
    plt.imshow(Image.open(files[query_id]))
    plt.title('Query\n' + true_label)
    plt.axis('off')

    for cnt in range(K):
        plt.subplot(2, 5, cnt + 2)
        plt.imshow(Image.open(files[trainset[ranking[cnt]]]))
        plt.title(unique_labels[nearest_labels[cnt] - 1])
        plt.axis('off')

print "Accuracy method 1: %d/%d\nAccuracy method 2: %d/%d" % (accuracy_1,
                                                              len(query_set),
                                                              accuracy_2,
                                                              len(query_set))
plt.show()
#"""


# UNTOUCHED CODE
"""
# Q3: For K = 9, COMPUTE THE CLASS ACCURACY FOR THE TESTSET
dist = np.zeros([len(testset), len(trainset)])
for i in range(len(testset)):
    for j in range(len(trainset)):
      dist[i.j] = ...

classAcc = np.zeros(len(unique_labels))
for c in range(len(unique_labels)):
  tp = ...
  fn = ...
  classAcc[c] = tp / (tp + fn + 0.0)

# REPORT THE CLASS ACC *PER CLASS* and the MEAN
# THE MEAN SHOULD BE (CLOSE TO): 0.31
"""


# UNTOUCHED CODE
"""
# Q4: DO CROSS VALIDATION TO DEFINE PARAMETER K
K = [1, 3, 5, 7, 9, 15]

# - SPLIT TRAINING SET INTO THREE PARTS.
# - RANDOMLY SELECT TWO PARTS TO TRAIN AND 1 PART TO VALIDATE
# - MEASURE THE MEAN CLASSIFICATION ACCURACY FOR ALL IMAGES IN THE VALIDATION PART
# - REPEAT FOR ALL POSSIBLE COMBINATIONS OF TWO PARTS
# - PICK THE BEST K AS THE VALUE OF K THAT WORKS BEST ON AVERAGE FOR ALL POSSIBLE
#   COMBINATIONS OF TRAINING-VALIDATION SETS

# PART 3. SVM ON TOY DATA
data, labels = week56.generate_toy_data()
svm_w, svm_b = week56.generate_toy_potential_classifiers(data,labels)
"""


# Q5: CLASSIFY ACCORDING TO THE 4 DIFFERENT CLASSIFIERS AND VISUALIZE THE RESULTS

# WORKING CODE
"""
data, labels = week56.generate_toy_data()
svm_w, svm_b = week56.generate_toy_potential_classifiers(data, labels)
pred = []
for i in range(len(data)):
  pred.append(np.sign((svm_w[0].T * data[i]) + svm_b[0]))


plt.figure()
plt.scatter(data[labels==1, 0], data[labels==1, 1], facecolor='r')
plt.scatter(data[labels==-1, 0], data[labels==-1, 1], facecolor='g')
plt.plot(data[pred==1, 0], data[pred==1, 1], marker='o', markersize=10, markeredgecolor='r', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)
plt.plot(data[pred==-1, 0], data[pred==-1, 1], marker='o', markersize=10, markeredgecolor='g', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)

plt.show()
"""




# UNTOUCHED CODE
"""
# Q6: USE HERE SVC function from sklearn to run a linear svm
# THEN USE THE PREDICT FUNCTION TO PREDICT THE LABEL FOR THE SAME DATA
svc = svm.SVC( ... )
pred = ...
"""


# UNTOUCHED CODE
"""
# PART 4. SVM ON RING DATA
data, labels = week56.generate_ring_data()

figure()
plt.scatter(data[labels==1, 0], data[labels==1, 1], facecolor='r')
plt.scatter(data[labels==-1, 0], data[labels==-1, 1], facecolor='g')

# Q7: USE LINEAR SVM AS BEFORE, VISUALIZE RESULTS and DRAW PREFERRED CLASSIFICATION LINE IN FIGURE

# Q8: (report only)

C = 1.0  # SVM regularization parameter
# Q9: TRANSFORM DATA TO POLAR COORDINATES FIRST
rad =
ang =
# PLOT POLAR DATA

data2 = np.vstack((rad, ang))
data2 = data2.T

# Q10: USE THE LINEAR SVM AS BEFORE (BUT ON DATA 2)

# PLOT THE RESULTS IN ORIGINAL DATA

# PLOT POLAR DATA


# PART 5. LOAD BAG-OF-WORDS FOR THE OBJECT IMAGES AND RUN SVM CLASSIFIER FOR THE OBJECTS

files, labels, label_names, unique_labels, trainset, testset = week56.get_objects_filedata()

K = 500
bow = np.zeros([len(files), K])
cnt = -1
for impath in files:
    cnt = cnt + 1
    print str(cnt) + '/' + str(len(files)) + '): ' + impath
    filpat, filnam, filext = tools.fileparts(impath)
    filpat2, filnam2, filext2 = tools.fileparts(filpat)
    bow[cnt, :] = week56.load_bow('../../data/bow_objects/codebook_' + str(K) + '/' + filnam2 + '/' + filnam + '.pkl')

# Q11: USE linear SVM, perform CROSS VALIDATION ON C = (.1,1,10,100), evaluate using MEAN CLASS ACCURACY

# Q12: Visualize the best performing SVM, what are good classes, bad classes, examples of images etc

# Q13: Compare SVM with k-NN



"""
