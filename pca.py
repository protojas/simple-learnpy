
import simplestats as ss
from pprint import pprint

#returns a list of eigenvalues and a list of eigenvectors for the covariance matrix of A
def princomp(A):
    T,V =  ss.eig(ss.cov(A))
    return T,ss.transpose(V)

#projects A onto the first k principal components
def pca(A, k):
    A = map(lambda y: ss.vecsub(y, ss.matmean(A)), A)
    E,V = princomp(A)
    V = [x for (y,x) in sorted(zip(E,V), key=lambda pair: pair[0], reverse=True)]
    E = sorted(E, reverse=True)
    E = E[:k]
    V = V[:k]
    return V

def project(pts, pcaobj):
    V = pcaobj
    return ss.transpose(ss.dot(V, ss.transpose(pts)))
