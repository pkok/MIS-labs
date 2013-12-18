import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
import random
import os
import math
sys.path.insert(0, '../')
import tools
from sklearn import svm, datasets
import copy

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


def load_bow(booksize, files, mute=False):
    bow = np.zeros([len(files), booksize])
    cnt = -1
    for impath in files:
        cnt = cnt + 1
        if not mute:
            print '\r' + str(cnt) + '/' + str(len(files)) + '): ' + impath,
        filpat, filnam, filext = tools.fileparts(impath)
        filpat2, filnam2, filext2 = tools.fileparts(filpat)
        bow[cnt, :] = week56.load_bow('../../data/bow_objects/codebook_' + str(booksize) + '/' + filnam2 + '/' + filnam + '.pkl')
    if not mute:
        print ''
    return bow


"""
####
# WORKING CODE
files, labels, label_names, unique_labels, trainset, testset = week56.get_objects_filedata()

bow = load_bow(100, files)
label_maxlen = max(map(len, unique_labels))

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


"""
#   COMBINATIONS OF TRAINING-VALIDATION SETS
# PART 3. SVM ON TOY DATA
# Q5: CLASSIFY ACCORDING TO THE 4 DIFFERENT CLASSIFIERS AND VISUALIZE THE RESULTS

# COMPLETED CODE
data, labels = week56.generate_toy_data()
svm_w, svm_b = week56.generate_toy_potential_classifiers(data, labels)
for x in xrange(4):
    pred = []
    max_error = [0.0, 0.0]
    for i in range(len(data)):
        distance = (svm_w[x].T * data[i]) + svm_b[x]
        max_error[int(labels[i])] = max(max_error[int(labels[i])], math.sqrt(distance[0][0]**2 + distance[0][1]**2))
        pred.append(np.sign((svm_w[x].T * data[i]) + svm_b[x]))

    plt.figure()
    plt.scatter(data[labels==1, 0], data[labels==1, 1], facecolor='r')
    plt.scatter(data[labels==-1, 0], data[labels==-1, 1], facecolor='g')

    for i in range(len(data)):
        plt.title("error_margin = "+ str(max_error))
        if (pred[i][0][0] == 1 and pred[i][0][1] == 1):
            plt.plot(data[i][0], data[i][1], marker='o', markersize=10, markeredgecolor='r', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)
        elif (pred[i][0][0] == -1 and pred[i][0][1] == -1):
            plt.plot(data[i][0], data[i][1], marker='o', markersize=10, markeredgecolor='g', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)
    plt.show()
"""

"""
# COMPLETED CODE
# Q6: USE HERE SVC function from sklearn to run a linear svm
# THEN USE THE PREDICT FUNCTION TO PREDICT THE LABEL FOR THE SAME DATA

data, labels = week56.generate_toy_data()
clf = svm.LinearSVC()
clf.fit(data, labels)
plt.figure()
plt.scatter(data[labels==1, 0], data[labels==1, 1], facecolor='r')
plt.scatter(data[labels==-1, 0], data[labels==-1, 1], facecolor='g')
for i in range(len(data)):
    if (clf.predict(data[i]) > 0):
        plt.plot(data[i][0], data[i][1], marker='o', markersize=10, markeredgecolor='r', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)
    else:
        plt.plot(data[i][0], data[i][1], marker='o', markersize=10, markeredgecolor='g', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)
plt.show()
"""

"""
# PART 4. SVM ON RING DATA

# COMPLETED CODE
# Q7: USE LINEAR SVM AS BEFORE, VISUALIZE RESULTS and DRAW PREFERRED CLASSIFICATION LINE IN FIGURE

data, labels = week56.generate_ring_data()
clf = svm.LinearSVC()
clf.fit(data, labels)
plt.figure()
plt.scatter(data[labels==1, 0], data[labels==1, 1], facecolor='r')
plt.scatter(data[labels==-1, 0], data[labels==-1, 1], facecolor='g')
false_negatives = 0.0
true_positives = 0.0
for i in range(len(data)):
    if (clf.predict(data[i]) > 0):
        if(labels[i] < 0):
            false_negatives += 1.0
        else:
            true_positives += 1.0
        plt.plot(data[i][0], data[i][1], marker='o', markersize=10, markeredgecolor='r', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)
    else:
        plt.plot(data[i][0], data[i][1], marker='o', markersize=10, markeredgecolor='g', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)
plt.title("accuracy = "+ str(true_positives / (true_positives + false_negatives)))
plt.show()
"""

"""
# COMPLETED CODE
# Q8: (report only)
# extra data for non linear classifier

data, labels = week56.generate_ring_data()
clf = svm.SVC()
clf.fit(data, labels)
plt.figure()
plt.scatter(data[labels==1, 0], data[labels==1, 1], facecolor='r')
plt.scatter(data[labels==-1, 0], data[labels==-1, 1], facecolor='g')
false_negatives = 0.0
true_positives = 0.0
for i in range(len(data)):
    if (clf.predict(data[i]) > 0):
        if(labels[i] < 0):
            false_negatives += 1.0
        else:
            true_positives += 1.0
        plt.plot(data[i][0], data[i][1], marker='o', markersize=10, markeredgecolor='r', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)
    else:
        plt.plot(data[i][0], data[i][1], marker='o', markersize=10, markeredgecolor='g', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)
plt.title("accuracy = "+ str(true_positives / (true_positives + false_negatives)))
plt.show()
"""

"""
# Q9: TRANSFORM DATA TO POLAR COORDINATES FIRST
data, labels = week56.generate_ring_data()
polar_data = copy.copy(data)
for i in xrange(len(data)):
    polar_data[i,0] = math.sqrt(data[i,0]**2 + data[i,1]**2)
    polar_data[i,1] = np.arctan2(data[i,1],data[i,0])

plt.figure()
plt.scatter(polar_data[labels==1, 0], polar_data[labels==1, 1], facecolor='r')
plt.scatter(polar_data[labels==-1, 0], polar_data[labels==-1, 1], facecolor='g')
plt.show()
"""
"""
# Q10: USE THE LINEAR SVM AS BEFORE (BUT ON DATA 2)
clf = svm.LinearSVC()
clf.fit(polar_data, labels)
plt.figure()
plt.scatter(polar_data[labels==1, 0], polar_data[labels==1, 1], facecolor='r')
plt.scatter(polar_data[labels==-1, 0], polar_data[labels==-1, 1], facecolor='g')
false_negatives = 0.0
true_positives = 0.0
for i in range(len(polar_data)):
    if (clf.predict(polar_data[i]) > 0):
        if(labels[i] < 0):
            false_negatives += 1.0
        else:
            true_positives += 1.0
        plt.plot(polar_data[i][0], polar_data[i][1], marker='o', markersize=10, markeredgecolor='r', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)
    else:
        plt.plot(polar_data[i][0], polar_data[i][1], marker='o', markersize=10, markeredgecolor='g', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)
plt.title("accuracy = "+ str(true_positives / (true_positives + false_negatives)))
plt.show()

# PLOT THE RESULTS IN ORIGINAL DATA
plt.figure()
plt.scatter(data[labels==1, 0], data[labels==1, 1], facecolor='r')
plt.scatter(data[labels==-1, 0], data[labels==-1, 1], facecolor='g')
for i in range(len(data)):
    if (clf.predict(polar_data[i]) > 0):
        plt.plot(data[i][0], data[i][1], marker='o', markersize=10, markeredgecolor='r', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)
    else:
        plt.plot(data[i][0], data[i][1], marker='o', markersize=10, markeredgecolor='g', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)
plt.show()
"""


# PART 5. LOAD BAG-OF-WORDS FOR THE OBJECT IMAGES AND RUN SVM CLASSIFIER FOR THE OBJECTS

random.seed(0)
files, labels, label_names, unique_labels, trainset, testset = week56.get_objects_filedata()

"""
# Q11: USE linear SVM, perform CROSS VALIDATION ON C = (.1,1,10,100), evaluate using MEAN CLASS ACCURACY
print "### Q11 ###"
booksizes = (10, 100, 500, 1000, 4000)
Cs = (.1,1,10,100)
mean_acc = {booksize: {C: 0 for C in Cs} for booksize in booksizes}
C_opt = {booksize: 0 for booksize in booksizes}

for booksize in booksizes:
    #1) Load the histograms for all images and all classes.
    #2) Run the linear SVM classifier for them using different codebook sizes.
    print "Using codebook", booksize
    bow = load_bow(booksize, files)

    #3) Tune with cross-validation on the training data the SVM parameter $C$ (use 3 splits of the train set).
    shuffled = trainset[:]
    random.shuffle(shuffled)
    size = len(trainset) / 3
    subsets = [shuffled[:size], shuffled[size:-size], shuffled[-size:]]
    subset_ids = set(range(3))
    # - REPEAT FOR ALL POSSIBLE COMBINATIONS OF TWO PARTS
    for i in subset_ids:
        train_ids = np.concatenate(map(lambda x: subsets[x], subset_ids - set([i])))

    #4) Classify all your test images for all your classes and report the mean classification accuracy.
        for C in Cs:
            clf = svm.LinearSVC(C=C)
            clf.fit(bow[train_ids], labels[train_ids])
            mean_acc[booksize][C] += clf.score(bow[subsets[i]], labels[subsets[i]])

    for C in Cs:
        mean_acc[booksize][C] /= 3.

    #5) What do you observe?
    C_opt[booksize] = max(mean_acc[booksize], key=lambda C: mean_acc[booksize][C])
    print "Best C for booksize", booksize, ":", C_opt[booksize]

#6) Repeat (1-5) for different codebook sizes and report which codebook works the best for you.
C_opt = max(C_opt, key=lambda booksize: C_opt[booksize])
print "Overall best book size:", C_opt
print "Booksize C.1    C1      C10    C1000",
for booksize in booksizes:
    print "\n%4d " % booksize,
    for C in Cs:
        print " %.3f " % mean_acc[booksize][C],
"""

"""
# Q12: Visualize the best performing SVM, what are good classes, bad classes, examples of images etc
print "### Q12 ###"
bow = load_bow(4000, files)
clf = svm.LinearSVC(C=0.1)
clf.fit(bow[trainset], labels[trainset])
class_test = [list() for i in range(len(unique_labels))]
for test_datum in testset:
    class_test[labels[test_datum] - 1].append(test_datum)
label_maxlen = max(map(len, unique_labels))
for i, test in enumerate(class_test):
    print ("%%%ds %%.3f" % label_maxlen) % (unique_labels[i], clf.score(bow[test], labels[test]))
print "Mean acc:", clf.score(bow[testset], labels[testset])
"""

"""
# Q13: Compare SVM with k-NN
booksizes = (10, 100, 500, 1000, 4000)
print "### Q13 ###"
for booksize in booksizes:
    bow = load_bow(booksize, files, mute=True)

    clf = svm.LinearSVC(C=0.1)
    clf.fit(bow[trainset], labels[trainset])
    svm_acc = clf.score(bow[testset], labels[testset])

    K = 9
    dist = compute_dists(testset, trainset)
    _, knn_acc = class_accuracy(K, unique_labels, trainset, testset,
                                        method=ORIGINAL_METHOD, dist=dist)
    print "%4d & %.2f & %.2f \\\\" % (booksize, svm_acc, knn_acc)
"""
