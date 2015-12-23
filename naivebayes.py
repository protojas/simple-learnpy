from simplestats import *

def train(Ys, Xs, classes=[], active=[]):
    assert (Xs != []), "no training data"
    assert (len(Xs) == len(Ys)), "Xs and Ys have differing lengths"
        
    if active == []:
        active = range(len(Xs[0]))
    
    if classes == []:
        classset = {}
        map(classset.__setitem__, Ys, [])
        classes = classset.keys()

    Xs = transpose([transpose(Xs)[a] for a in active])

    split_X = []
    for c in range(len(classes)):
        split_X += [[Xs[i] for i in range(len(Xs)) if Ys[i] == classes[c]]]

    means = []
    covs = []
    priors = []
    covinvs = []
    for c in range(len(classes)):
        means += [map(mean, transpose(split_X[c]))]
        covs += [map(var, transpose(split_X[c]))]
        priors += [Ys.count(classes[c])/float(len(Ys))]
    covs = map(lambda x: diag(x), covs)
    for c in range(len(classes)):
        covinvs += [inv(covs[c])]
    return [means, covs, priors, covinvs, classes, active]

def classify(datum, trainobj):
    means = trainobj[0]
    covs = trainobj[1]
    priors = trainobj[2]
    covinvs = trainobj[3]
    classes = trainobj[4]
    active = trainobj[5]
    
    posteriors = []
    likelihoods = []

    datum = [datum[a] for a in active]
    for c in range(0, len(classes)):
        likelihoodtmp = 1
        for m in range(0, len(means)):
            likelihoodtmp *= normalX(datum, covs[c], means[c], covinv=covinvs[c])
        likelihoods += [likelihoodtmp]
    for c in range(0, len(classes)):
        posteriors += [likelihoods[c] * priors[c]]
    return classes[posteriors.index(max(posteriors))]
