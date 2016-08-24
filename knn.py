from simplestats import *

def train(Ys, Xs, classes=[], active=[], k=3, dist=euclidean):
    assert (Xs != []), "no training data"
    assert (len(Xs) == len(Ys)), "Xs and Ys have differing lengths"

    if active == []:
        active =  range(len(Xs[0]))
    if classes == []:
        classset = {}
        map(classset.__setitem__, Ys, [])
        classes = classset.keys()

    Xs = transpose([transpose(Xs)[a] for a in active])

    return Ys, Xs, classes, active, k, dist

def classify(datum, trainobj):
    Ys = trainobj[0]
    Xs = trainobj[1]
    classes = trainobj[2]
    active = trainobj[3]
    k = trainobj[4]
    dist = trainobj[5]

    datum = [datum[a] for a in active]

    ds = map(lambda p: dist(datum,p), Xs)

    ds = zip(Ys, ds)
    ds.sort(key=lambda x: x[1], reverse=True)
    ds = map(lambda x: x[0], ds[:k]) # k nearest neighbors' classes

    counts = map(lambda x: ds.count(x),classes)
    cs = zip(classes, counts, map(lambda x: ds.index(x), classes))
    cs = sorted(cs, cmp=bestclasscomp, reverse=True)
    return cs[0][0]

def bestclasscomp(i, j):
    if i[1] != j[1]:
        return i[1]-j[1]
    else:
        return i[2]-j[2]
