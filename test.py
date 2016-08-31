import knn as knn
import naivebayes as nb
import maximumlikelihood as mle
import pca as pca
import simplestats
import math

Xs = [  [1,2,3,4],
        [3,4,1,2],
        [5,7,3,4],
        [2,9,4,5],
        [5,1,6,7],
        [3,2,8,8],
        [4,1,1,1],
        [2,3,9,9]    ] # training data attributes
Ys = [  0,
        1,
        0,
        1,
        1,
        0,
        1,
        1,     ] # labels for the training data

active = [0, 1] # which attributes to actually consider - default is to use all the attributes
classes = [0, 1] # set of all classes to consider - default is to get the classes from Ys


testpoint = [1,3,10,11]



pcaobj = pca.pca(Xs, 3) # find the top 3 principal components

Xs = pca.project(Xs, pcaobj) # project the training data onto the principal components
testpoint = pca.project([testpoint], pcaobj)[0] # project the test point(s) onto the principal components

knnobj = knn.train(Ys, Xs, classes, active, 7, simplestats.euclidean)
print knn.classify(testpoint, knnobj)

nbobj = nb.train(Ys, Xs, classes, active)
print nb.classify(testpoint, nbobj)

mleobj = mle.train(Ys, Xs, classes, active)
print mle.classify(testpoint, mleobj)

