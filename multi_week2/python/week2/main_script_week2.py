# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# PRE A : GO TO MAIN DIR FOR WEEK2, THAT IS WHERE MAIN_SCRIPT_WEEK2.PY IS
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import sys
import os

from scipy import ndimage
from pylab import *
from numpy import *
from collections import defaultdict

import homography
import warp

import week2
sys.path.insert(0, '../')
import tools

try:
    import week1
except ImportError:
    print "You might want to copy week1.py from last week to this folder."

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# PRE B : SETTING PRINTING OPTIONS [FEEL FREE TO CHANGE THE VALUES TO YOUR LIKING]
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=200)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# STARTING THE ASSIGNMENT
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

##############################################################################
#### PART 2. SIFT FEATURES
##############################################################################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#### STEP A: READ AN IMAGE AND EXTRACT SIFT FEATURES FROM IT
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
impath1 = '../../data/oxford_scaled/all_souls_000026.jpg'
frames1, sift1 = week2.compute_sift(impath1)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#### STEP B: PLOT SIFT FEATURES
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
figure(1)
subplot(1, 2, 1)
im1 = Image.open(impath1)
week2.plot_features(im1, frames1, False, color='r')
subplot(1, 2, 2)
week2.plot_features(im1, frames1, True, color='r') # [YOU NEED TO FINISH THE IMPLEMENTATION FOR THIS]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#### STEP C: PLOT SIFT MATCHES
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
impath1 = '../../data/oxford_scaled/all_souls_000026.jpg'
frames1, sift1 = week2.compute_sift(impath1)

impath2 = '../../data/oxford_scaled/all_souls_000068.jpg'
frames2, sift2 = week2.compute_sift(impath2)

week2.store_cache()

figure(1)
subplot(1, 2, 1)
im1 = Image.open(impath1)
week2.plot_features(im1, frames1, False, 'r')
subplot(1, 2, 2)
im2 = Image.open(impath2)
week2.plot_features(im2, frames2, False, 'r')

# STEP C.1 NORMALIZE SIFT VECTORS
sift1 = tools.normalizeL2(sift1, 1)
sift2 = tools.normalizeL2(sift2, 1)

# STEP C.2 COMPUTE DISTANCES BETWEEN ALL VECTORS IN SIFT1 AND SIFT2.

# PUT YOUR CODE IN week2.match_sift, SO THAT IT CAN BE CALLED AS
dist_thresh = 1.1
matches, ranked_ratio = week2.match_sift(sift1, sift2, dist_thresh) # [YOU NEED TO IMPLEMENT THIS]
week2.store_cache()

while True:
    rawinput = raw_input('>> ')
    if rawinput == 'q':
        week2.store_cache()
        sys.exit()
    tmp_matches = -1 * np.ones(matches.shape)
    theta = float(rawinput)
    for i in range(len(matches)):
        if ranked_ratio[i] > theta:
            tmp_matches[i] = matches[i]
    print sum(tmp_matches != -1)
    # NOW YOU ARE ABLE TO PLOT THE MATCHES AND GET SOMETHING LIKE IN FIG.1 OF THE HANDOUT
    week2.plot_matches(im1, im2, frames1, frames2, tmp_matches) # [ALREADY IMPLEMENTED]
    plt.show()

"""
# !!! EXPERIMENT FOR DIFFERENT VALUES OF DIST_THRESH AND CONTINUE WITH THAT ONE.
# !!! REPORT THIS VALUE IN YOUR REPORT

##############################################################################
#### PART 3. SIFT FEATURES ARE INVARIANT TO
##############################################################################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#### STEP D: ROTATION (REPEAT FOR 15, 30, 45, 60, 75, 90)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
impath1 = '../../data/oxford_scaled/all_souls_000026.jpg'
frames1, sift1 = week2.compute_sift(impath1)

im1 = Image.open(impath1)

match_count = {}
for deg in np.arange(15, 90+15, 15):
    im_rot = im1.rotate(np.int(deg))
    im_rot.save('temp_rot.jpg')
    frames_rot, sift_rot = week2.compute_sift('temp_rot.jpg')
    matches = week2.match_sift(sift1, sift_rot, 1.1) # [YOU NEED TO IMPLEMENT THIS]
    week2.plot_matches(im1, im_rot, frames1, frames_rot, matches) # [ALREADY IMPLEMENTED]
    title('Rotating ' + str(deg) + ': Number of matches is ' + str(sum(matches!=-1)))
    match_count[deg] = sum(matches != -1)

week2.store_cache()
    
# MAKE A PLOT HERE. ON THE X-AXIS THERE SHOULD BE THE ROTATION CHANGES. ON THE Y-AXIS
fig = plt.figure()
ax = plt.axes()
ax.set_ylabel("Matches")
ax.set_xlabel("Rotation [deg]")
plt.plot(*zip(*match_count.items()))

# THERE SHOULD BE THE NUMBER OF MATCHES. BASED ON THE PLOT TELL IN YOUR OPINION IF SIFT
# IS ROTATION INVARIANT, THAT IS IF THE NUMBER OF MATCHES REMAINS SIMILAR FOR ROTATION CHANGES

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#### STEP E: SCALE  (REPEAT FOR x0.2, x0.5, x0.8, x1.2, x1.5, x1.8)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
impath1 = '../../data/oxford_scaled/all_souls_000026.jpg'
frames1, sift1 = week2.compute_sift(impath1)

im1 = Image.open(impath1)
H, W, C = np.array(im1).shape

match_count = {}
for alpha in np.arange(0.2, 2.0 + 0.2, 0.2):
    print alpha
    im_scl = im1.resize((int(alpha * W), int(alpha * H)))
    im_scl.save('temp_scl.jpg')
    frames_scl, sift_scl = week2.compute_sift('temp_scl.jpg')
    matches = week2.match_sift(sift1, sift_scl, 1.1) # [YOU NEED TO IMPLEMENT THIS]
    week2.plot_matches(im1, im_scl, frames1, frames_scl, matches) # [ALREADY IMPLEMENTED]
    title('Scaling x' + str(alpha) + ': Number of matches is ' + str(sum(matches!=-1)))
    match_count[alpha] = sum(matches != -1)
    
week2.store_cache()
# MAKE A PLOT HERE. ON THE X-AXIS THERE SHOULD BE THE SCALE CHANGES. ON THE Y-AXIS
# THERE SHOULD BE THE NUMBER OF MATCHES. BASED ON THE PLOT TELL IN YOUR OPINION IF SIFT
# IS SCALE INVARIANT, THAT IS IF THE NUMBER OF MATCHES REMAINS SIMILAR FOR SCALE CHANGES
fig = plt.figure()
ax = plt.axes()
ax.set_ylabel("Matches")
ax.set_xlabel("Scale")
plt.plot(*zip(*match_count.items()))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#### STEP F: PERSPECTIVE (REPEAT FOR ALL PERSPECTIVE CHANGED IMAGES)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# '../../data/other/all_souls_000026_prsp1.jpg'
# '../../data/other/all_souls_000026_prsp2.jpg'
# '../../data/other/all_souls_000026_prsp3.jpg'
# '../../data/other/all_souls_000026_prsp4.jpg'

impath1 = '../../data/oxford_scaled/all_souls_000026.jpg'
frames1, sift1 = week2.compute_sift(impath1)

im1 = Image.open(impath1)

match_count = {}
for i in range(4):
    im_prsp = Image.open('../../data/other/all_souls_000026_prsp' + str(i+1) + '.jpg')
    frames_prsp, sift_prsp = week2.compute_sift('../../data/other/all_souls_000026_prsp' + str(i+1) + '.jpg')
    matches = week2.match_sift(sift1, sift_prsp, 1.1) # [YOU NEED TO IMPLEMENT THIS]
    week2.plot_matches(im1, im_prsp, frames1, frames_prsp, matches) # [ALREADY IMPLEMENTED]
    title('Pespective ' + str(i+1) + ': Number of matches is ' + str(sum(matches!=-1)))
    match_count[i] = sum(matches != -1)
    
week2.store_cache()
# MAKE A PLOT HERE. ON THE X-AXIS THERE SHOULD BE THE PERSPECTIVE CHANGES. ON THE Y-AXIS
# THERE SHOULD BE THE NUMBER OF MATCHES. BASED ON THE PLOT TELL IN YOUR OPINION IF SIFT
# IS PERSPECTIVE INVARIANT, THAT IS IF THE NUMBER OF MATCHES REMAINS SIMILAR FOR PERSPECTIVE CHANGES
fig = plt.figure()
ax = plt.axes()
ax.set_ylabel("Matches")
ax.set_xlabel("Scale")
plt.plot(*zip(*match_count.items()))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#### STEP G: BRIGHTNESS (REPEAT FOR BRIGHTNESS CHANGES 0.5, 0.8, 1.2, 1.5, 1.8)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
impath1 = '../../data/oxford_scaled/all_souls_000026.jpg'
frames1, sift1 = week2.compute_sift(impath1)

im1 = Image.open(impath1)

match_count = {}
for beta in [0.5, 0.8, 1.2, 1.5, 1.8]:
    im_brt = ImageEnhance.Brightness(im1)
    im_brt = im_brt.enhance(beta)
    im_brt.save('temp_brt.jpg')
    frames_brt, sift_brt = week2.compute_sift('temp_brt.jpg')
    matches = week2.match_sift(sift1, sift_brt, 1.1) # [YOU SHOULD HAVE THIS IMPLEMENTED BY NOW]
    week2.plot_matches(im1, im_brt, frames1, frames_brt, matches) # [ALREADY IMPLEMENTED]
    title('Brightness x' + str(beta) + ': Number of matches is ' + str(sum(matches!=-1)))
    match_count[beta] = sum(matches != -1)

week2.store_cache()
# MAKE A PLOT HERE. ON THE X-AXIS THERE SHOULD BE THE BRIGHTNESS CHANGES. ON THE Y-AXIS
# THERE SHOULD BE THE NUMBER OF MATCHES. BASED ON THE PLOT TELL IN YOUR OPINION IF SIFT
# IS BRIGHTNESS INVARIANT, THAT IS IF THE NUMBER OF MATCHES REMAINS SIMILAR FOR BRIGHTNESS CHANGES
fig = plt.figure()
ax = plt.axes()
ax.set_ylabel("Matches")
ax.set_xlabel("Scale")
plt.plot(*zip(*match_count.items()))

##############################################################################
#### PART 4. USING SIFT FOR GEOMETRY
##############################################################################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#### STEP H: FIND THE POINTS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Open images
impath1 = '../../data/other/all_souls_000026.jpg'
im1 = Image.open(impath1)
frames1, sift1 = week2.compute_sift(impath1)

impath2 = '../../data/other/all_souls_000068.jpg'
im2 = Image.open(impath2)
frames2, sift2 = week2.compute_sift(impath2)

# Pre-computed homography matrix
H = np.array([[   0.554,    0.049,   72.393],
       [  -0.014,    0.642,  215.047],
       [  -0.   ,    0.   ,    1.   ]])

# VISUALIZING A POINT ON AN IMAGE

points = np.array((
    (562, 52), 
    (233, 170),
    (344, 217),
    (232, 440),
    (93, 87)))
colors = 'rgbmy'

figure(1)
imshow(im1)
axis('off')
for p, c in zip(points, colors):
    plot(p[0], p[1], c + 'o', markersize=10)

figure(2)
imshow(im2)
axis('off')
for p, c in zip(points, colors):
    # IN ORDER TO FIND THE LOCATION OF THE POINT TO THE NEW IMAGE, YOU NEED TO APPLY EQ.(5) FROM THE HANDOUT
    newp = week2.project_point_via_homography(H, p) # [YOU NEED TO FINISH THE IMPLEMENTATION OF THIS FUNCTION]

    # VISUALIZING THE SAME POINT, ESTIMATED ON THE SECOND IMAGE
    plot(newp[0], newp[1], c + 'd', markersize=10)
    axis('off')

"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#### STEP I: COMPUTE HOMOGRAPHY BETWEEN 2 IMAGES
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impath1 = '../../data/other/all_souls_000026.jpg'
im1 = Image.open(impath1)
frames1, sift1 = week2.compute_sift(impath1)

impath2 = '../../data/other/all_souls_000068.jpg'
im2 = Image.open(impath2)
frames2, sift2 = week2.compute_sift(impath2)

matches = week2.match_sift(sift1, sift2, 1.1) # BY NOW YOU SHOULD HAVE THIS ALREADY IMPLEMENTED
week2.store_cache()
H, inliers = homography.estimate_homography_with_ransac(frames1, frames2, matches) # ALREADY PROVIDED

# PLOT GOOD MATCHES
good_matches = -1 * np.ones(frames1.shape[0])
good_matches[inliers] = inliers
week2.plot_matches(im1, im2, frames1[ix], frames2[matches[ix]], good_matches)

# PLOT BAD MATCHES WITH A DIFFERENT COLOR
bad_matches = np.arange(frames1.shape[0])
bad_matches[inliers] = -1
week2.plot_matches(im1, im2, frames1[ix], frames2[matches[ix]], bad_matches, 'b', False)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#### STEP J: IMAGE RECOGNITION WITH GEOMETRIC VERIFICATION
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

files = os.listdir('../../data/oxford_scaled/')
query = 'all_souls_000026.jpg'
impath_qu = '../../data/oxford_scaled/' + query
frames_qu, sift_qu = week2.compute_sift(impath_qu)

# recognized = []
# min_inliers = ...

# LOOP OVER ALL FILES AND FIND THE ONES THAT MATCH WELL GEOMETRICALLY WITH THE QUERY. FIND A GOOD VALUE FOR
# A MINIMUM NUMBER OF INLIERS, USING THE estimate_homography_with_ransac function from above

# for f in files:
#    ...

