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

# For reproducability of results, use a seed for your random generator.
random.seed(0)

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=200)

ORIGINAL_METHOD = 1
SUGGESTED_METHOD = 2
BOTH_METHODS = ORIGINAL_METHOD | SUGGESTED_METHOD


def compute_dists(testset, trainset):
    dist = np.zeros([len(testset), len(trainset)])
    for i in range(len(testset)):
        b_i = tools.normalizeL1(bow[testset[i], :])
        for j in range(len(trainset)):
            b_j = tools.normalizeL1(bow[trainset[j], :])
            dist[i, j] = sum(np.minimum(b_i, b_j))
    return dist


def kNN_classifier(K, query_id, unique_labels, trainset, testset,
                   method=ORIGINAL_METHOD, dist=None, visualize=False):
    predicted_labels = []

    if dist is None:
        dist = compute_dists(testset, trainset)

    testset_id = np.argwhere(testset == query_id)[0, 0]
    ranking = np.argsort(dist[testset_id, :])
    ranking = ranking[::-1]
    tmp = ranking[:K]
    tmp = trainset[tmp]
    nearest_labels = labels[tmp]
    nearest_labels = labels[trainset[ranking[:K]]]

    # Label predicted and accuracy according to method of Q1
    if method & ORIGINAL_METHOD:
        predicted_labels.append(np.argmax(np.bincount(nearest_labels)))

    # Label predicted and accuracy according to method of Q2
    if method & SUGGESTED_METHOD:
        weighted_votes = np.zeros(len(unique_labels))
        for rank, ranked in enumerate(ranking[:K]):
            weighted_votes[labels[trainset[ranked]] - 1] += 1. / (1 + rank)
        predicted_labels.append(np.argmax(weighted_votes) + 1)

    # VISUALIZE RESULTS
    if visualize:
        plt.figure()
        plt.subplot(2, 5, 1)
        plt.imshow(Image.open(files[query_id]))
        plt.title('Query ' + str(query_id))
        plt.axis('off')

        for cnt in range(K):
            plt.subplot(2, 5, cnt + 2)
            plt.imshow(Image.open(files[trainset[ranking[cnt]]]))
            plt.title(unique_labels[nearest_labels[cnt] - 1])
            plt.axis('off')

    if len(predicted_labels) == 1:
        return predicted_labels[0]
    return predicted_labels


def class_accuracy(K, unique_labels, trainset, testset,
                   method=ORIGINAL_METHOD, dist=None):
    if dist is None:
        dist = compute_dists(testset, trainset)
    class_tp = np.zeros(len(unique_labels))
    for test_id in testset:
        predicted_label = kNN_classifier(K, test_id, unique_labels, trainset,
                                         testset, method=method, dist=dist)
        class_tp[labels[test_id] - 1] += labels[test_id] == predicted_label
    # REPORT THE CLASS ACC *PER CLASS* and the MEAN
    # THE MEAN SHOULD BE (CLOSE TO): 0.31
    class_total = np.bincount(labels[testset])[1:]
    class_acc = class_tp / class_total
    total_tp = sum(class_tp)
    total = sum(class_total)
    mean_acc = total_tp / float(total)
    return class_acc, mean_acc

"""
####
# WORKING CODE
files, labels, label_names, unique_labels, trainset, testset = week56.get_objects_filedata()

C = 100
bow = np.zeros([len(files), C])
cnt = -1
label_maxlen = max(map(len, unique_labels))
for impath in files:
    cnt = cnt + 1
    print '\r' + str(cnt) + '/' + str(len(files)) + '): ' + impath,
    filpat, filnam, filext = tools.fileparts(impath)
    filpat2, filnam2, filext2 = tools.fileparts(filpat)
    bow[cnt, :] = week56.load_bow('../../data/bow_objects/codebook_' + str(C) + '/' + filnam2 + '/' + filnam + '.pkl')
print ''

print 'Computing distance matrix...'
dist = compute_dists(testset, trainset)
K = 9
"""
# Q1: IMPLEMENT HERE kNN CLASSIFIER.
# YOU CAN USE CODE FROM PREVIOUS WEEK
# Q2: USE DIFFERENT STRATEGY
"""
# WORKING CODE
print "### Q1 and Q2 ###"
QUERIES = {
    366: "goal/15",
    150: "bicycle/37",
    84: "beach/31",
    450: "mountain/37"
}

accuracy = np.zeros(2)

query_set = testset
query_set = QUERIES

for query_id in query_set:
    true_label = labels[query_id]
    predicted_labels = kNN_classifier(K, query_id, unique_labels,
                                        trainset, testset,
                                        #visualize=True,
                                        method=BOTH_METHODS, dist=dist)
    accuracy += map(lambda x: x == true_label, predicted_labels)

for i, acc in enumerate(accuracy):
    print "Accuracy method %d: %d/%d" % (i, acc, len(query_set))

#plt.show()
"""

# Q3: For K = 9, COMPUTE THE CLASS ACCURACY FOR THE TESTSET
# WORKING CODE
"""
print "### Q3 ###"
class_acc1, mean_acc1 = class_accuracy(K, unique_labels, trainset, testset,
                                       method=ORIGINAL_METHOD, dist=dist)
class_acc2, mean_acc2 = class_accuracy(K, unique_labels, trainset, testset,
                                       method=SUGGESTED_METHOD, dist=dist)
# Pretty printing
print "Class accuracy for original and alternative kNN methods:"
printstr = "%%2i %%%is %%.1f   %%.1f" % label_maxlen
for i, l in enumerate(unique_labels):
    print printstr % (i + 1, l, class_acc1[i], class_acc2[i])
print "Mean accuracy: %.2f   %.2f" % (mean_acc1, mean_acc2)
"""


# Q4: DO CROSS VALIDATION TO DEFINE PARAMETER K
# WORKING CODE
"""
print "### Q4 ###"
Ks = (1, 3, 5, 7, 9, 15)

# - SPLIT TRAINING SET INTO THREE PARTS.
# - RANDOMLY SELECT TWO PARTS TO TRAIN AND 1 PART TO VALIDATE
shuffled = trainset[:]
random.shuffle(shuffled)
size = len(trainset) / 3
subsets = [shuffled[:size], shuffled[size:-size], shuffled[-size:]]
subset_ids = set(range(3))
# - REPEAT FOR ALL POSSIBLE COMBINATIONS OF TWO PARTS
class_acc = {K: np.zeros(len(unique_labels)) for K in Ks}
mean_acc = {K: 0 for K in Ks}
for i in subset_ids:
    validationset = subsets[i]
    trainset = np.concatenate(map(lambda x: subsets[x], subset_ids - set([i])))
    dist = compute_dists(validationset, trainset)
# - MEASURE THE MEAN CLASSIFICATION ACCURACY FOR ALL IMAGES IN THE VALIDATION PART
    for K in Ks:
        ca, ma = class_accuracy(K, unique_labels, trainset, validationset,
                               method=ORIGINAL_METHOD, dist=dist)
        class_acc[K] += ca
        mean_acc[K] += ma

for K in Ks:
    class_acc[K] /= 3.
    mean_acc[K] /= 3.

# - PICK THE BEST K AS THE VALUE OF K THAT WORKS BEST ON AVERAGE FOR ALL POSSIBLE
K_opt = max(mean_acc, key=lambda k: mean_acc[k])

print ((4+label_maxlen) * " ") + (len(Ks) * '  %3d  ') % Ks
printstr = "%%2i  %%%is" % label_maxlen + (len(Ks) * "  %.3f")
for i, l in enumerate(unique_labels):
    print printstr % ((i + 1, l) + tuple(class_acc[K][i] for K in Ks))
print ("%" + str(4+label_maxlen) + "s") % "Mean acc." + len(Ks) * "  %.3f" % tuple(mean_acc[K] for K in Ks)
print "Best K:", K_opt
"""

#   COMBINATIONS OF TRAINING-VALIDATION SETS
# PART 3. SVM ON TOY DATA
# Q5: CLASSIFY ACCORDING TO THE 4 DIFFERENT CLASSIFIERS AND VISUALIZE THE RESULTS
"""
# COMPLETED CODE
data, labels = week56.generate_toy_data()
svm_w, svm_b = week56.generate_toy_potential_classifiers(data, labels)
for x in xrange(4):
    pred = []
    error = 0.0
    for i in range(len(data)):
        distance = (svm_w[x].T * data[i]) + svm_b[x]
        error += math.sqrt(distance[0][0]**2 + distance[0][1]**2)
        pred.append(np.sign((svm_w[x].T * data[i]) + svm_b[x]))

    plt.figure()
    plt.scatter(data[labels==1, 0], data[labels==1, 1], facecolor='r')
    plt.scatter(data[labels==-1, 0], data[labels==-1, 1], facecolor='g')
    for i in range(len(data)):
        plt.title("error="+ str(error))
        if (pred[i][0][0] == 1 and pred[i][0][1] == 1):
            plt.plot(data[i][0], data[i][1], marker='o', markersize=10, markeredgecolor='r', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)
        elif (pred[i][0][0] == -1 and pred[i][0][1] == -1):
            plt.plot(data[i][0], data[i][1], marker='o', markersize=10, markeredgecolor='g', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)
    plt.show()
"""


# COMPLETED CODE
# Q6: USE HERE SVC function from sklearn to run a linear svm
# THEN USE THE PREDICT FUNCTION TO PREDICT THE LABEL FOR THE SAME DATA

data, labels = week56.generate_toy_data()
clf = svm.SVR()
clf.fit(data, labels)  
svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
gamma=0.0, kernel='rbf', max_iter=-1, probability=False, random_state=None,
shrinking=True, tol=0.001, verbose=True)
plt.figure()
plt.scatter(data[labels==1, 0], data[labels==1, 1], facecolor='r')
plt.scatter(data[labels==-1, 0], data[labels==-1, 1], facecolor='g')
for i in range(len(data)):
    if (clf.predict(data[i]) > 0):
        plt.plot(data[i][0], data[i][1], marker='o', markersize=10, markeredgecolor='r', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)
    else:
        plt.plot(data[i][0], data[i][1], marker='o', markersize=10, markeredgecolor='g', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)
plt.show()



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
