# simple-learnpy

Python library for some simple machine learning and classification, without numpy or scikit, which are often hard to install.

The goal is to provide the functionality of these classification techniques, without forcing a huge setup on the user - particularly, on Windows, scipy and scikit are unreasonably difficult to setup. These classification implementations, will operate entirely on arrays, and, where possible, will attempt to be fully functional, that is, without classes/objects.

Obviously, it will not be as efficient as Numpy, since it is written in Python and not C, but for results on simple data, this library will hopefully be much easier to use and be simpler to understand.

## Classifiers implemented

### Naive Bayes

### Maximum Likelihood Estimate

### K-Nearest Neighbors
Sample usage:
```python
import maximumlikelihood as mle
import naivebayes as nb
import knn as knn
import simplestats

Xs = [ 	[1,2],
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

nbobj = nb.train(Ys, Xs, classes, active)
nb.classify([1,3], nbobj) # returns the predicted class of a data point with attributes [1,3]

mleobj = mle.train(Ys, Xs, classes, active)
mle.classify([1,3], mleobj)

knnobj = knn.train(Ys, Xs, classes, active, 7, simplestats.euclidean) # must provide value of k and a distance function
knn.classify([1,3], knnobj)
```


Code presented in this repository is written by Julian Sexton, adapted from his code for a Machine Learning class at Stevens Institute of Technology. 
