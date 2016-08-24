import knn as knn
import naivebayes as nb
import maximumlikelihood as mle
import simplestats
import math

Xs = [  [1,2],
        [3,4],
        [5,7],
        [2,9],
        [5,1],
        [3,2],
        [4,1],
        [2,3]    ] # training data attributes
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


testpoint = [1,3]

knnobj = knn.train(Ys, Xs, classes, active, 7, simplestats.euclidean)
print knn.classify(testpoint, knnobj)

nbobj = nb.train(Ys, Xs, classes, active)
print nb.classify(testpoint, nbobj)

mleobj = mle.train(Ys, Xs, classes, active)
print mle.classify(testpoint, mleobj)
