import itertools
from math import pi as PI_CONST, e as E_CONST, sqrt
import cmath
# returns the mean of a list
def mean(arr):
    return sum(arr)/float(len(arr))

# returns the euclidean distance between two lists
def euclidean(a,b):
    z = zip(a,b)
    z = map(lambda r: pow(r[0]-r[1],2),z)
    return sqrt(sum(z))

# flattens a list of lists
def flatten(a):
    return [i for s in a for i in s]

# returns the city block distance between two lists
def cityblock(a,b):
    z = zip(a,b)
    z = map(lambda r: math.abs(a-b),z)
    return sum(z)

# returns the variance of a list
def var(arr):
    m = mean(arr);
    if len(arr) == 1:
        return 0
    return sum(map(lambda x: ((x - m) * (x - m)),arr))/float(len(arr)-1)

# returns the covariance matrix over the columns of a matrix
def cov(M):
    width = len(M[0])
    MT = transpose(M)
    covm = []
    for i in range(width):
        tmp = []
        for j in range(width):
            tmp += [covariance(MT[i], MT[j])]
        covm += [tmp]
    return covm

# returns the covariance between two random variables
def covariance(X, Y):
    assert(len(X) == len(Y)), "vectors not of equal size"
    xmean = mean(X)
    ymean = mean(Y)
    if len(X) == 1:
        return 0
    return sum(map(lambda i: ((X[i] - xmean) * (Y[i] - ymean)), range(len(X))))/float(len(X)-1)

# returns the transpose of a matrix
def transpose(M):
    nwidth = len(M)
    nheight = len(M[0])
    MT = []
    for m in range(nheight):
        tmp = []
        for n in range(nwidth):
            tmp += [M[n][m]]
        MT += [tmp]
    return MT

# returns the dot product of two matrices if they can be multiplied
def dot(P, Q):
    pheight = len(P)
    pwidth = len(P[0])
    qheight = len(Q)
    qwidth = len(Q[0])
    assert (pwidth == qheight), "matrices cannot be multiplied due to mismatched dimension"

    prod = []
    for i in range(pheight):
        tmp = []
        for j in range(qwidth):
            tmp += [sum(map(lambda k: P[i][k] * Q[k][j],range(pwidth)))]
        prod += [tmp]
    return prod

# returns the difference X - Y
def vecsub(X, Y):
    assert (len(X) == len(Y)), "vectors have different dimensions"
    return map(lambda i: X[i] - Y[i], range(len(X)))

# returns the determinant of a matrix if it is square
def det(M):
    assert(len(M) == len(M[0])), "matrix is not square"
    n = len(M)

    perms = permute(n)
    det = 0
    for p in perms:
        prod = 1
        for i in range(n):
            prod *= M[i][p[i]]
        prod *= sgn(p)
        det += prod
    return det

# returns the parity of the permutation
def sgn(P):
    n = len(P)
    v = [False] * n

    ret = 1
    for k in range(n):
        if not v[k]:
            r = k
            L = 0
            while not v[r]:
                L += 1
                v[r] = True
                r = P[r]
            if L % 2 == 0:
                ret = -1 * ret
    return ret

# returns the length of a vector
def veclen(v): #should be [[x x x x x]] or [[x] [x] [x] [x]]
    assert len(v) == 1 or len(v[0]) == 1, "not a vector"
    if len(v[0]) == 1:
        v = transpose(v)
    return euclidean(v[0], [0]*len(v[0]))

# returns Hn as a result of the Arnoldi iteration
# currently unused
def arnoldi(A):
    b = [5] * len(A)
    qs = [[]] * (len(A[0])+1)
    qs[0] = map(lambda x: float(x) / veclen([b]), b)
    h = [ [ 0 for i in range(len(A)) ] for j in range(len(A)+1) ]
    for n in range(0, len(A)):
        v = dot(A, transpose([qs[n]]))
	for j in range(0, n+1):
            h[j][n] =  dot([qs[j]], v)[0][0]
            v = transpose([vecsub(transpose(v)[0],map(lambda x: x * h[j][n], qs[j]))])
	h[n+1][n] = veclen(v)
        qs[n+1] = map(lambda x: float(x) / h[n+1][n],transpose(v)[0])
    return h[:-1]

# note: does not support complex eigenvalues!! don't try to use it for that!
# this is meant only for symmetric matrices (namely, covariance matrices, for PCA)
def eig(A):
    if len(A) == 2 and len(A[0]) == 2:
        return eig22(A)
    T,V = qr(A, 4*len(A))
    return [T[i][i] for i in range(len(T))], V


# QR factorization
def householder(A):
    n = len(A)
    R = A
    Q = [[float(0)] * n for i in range(n)]

    for k in range(n-1):
        I = identity(n)

        x = column(R[k:],k)
        e = column(I[k:],k)

        a = -cmp(x[0][0], 0) * veclen(x)
        u = matsub(x, scale(a,e))
        lenu = veclen(u)
        v = [map(lambda p: p/lenu, u[0])]
        Qm = matsub(identity(len(u[0])), scale(float(2), dot(transpose(v),v)))
        Qt = [[ Qi(Qm,i,j,k) for i in range(n)] for j in range(n)]
        if k == 0:
            Q = Qt
            R = dot(Qt,A)
        else:
            Q = dot(Qt,Q)
            R = dot(Qt,R)
    return transpose(Q), R

def Qi(Qm, i, j, k):
    if i < k or j < k:
        return float(i == j)
    else:
        return Qm[i-k][j-k]

# iterates and does QR factorizations up to itermax iterations
def qr(A, itermax):
    T = A
    iter = 0
    Qprod = identity(len(T))
    while iter < itermax:
        Q,R = householder(T)
        T = dot(R,Q)
        Qprod = dot(Qprod, Q)
        iter += 1
    return T, Qprod
#returns A scaled by constant k
def scale(k,A):
    ret = A
    for j in range(len(ret)):
        ret = rowscale(ret, j, k)
    return ret
# returns A-B if possible
def matsub(A,B):
    assert len(A) == len(B) and len(A[0]) == len(B[0]), "matrices not same size"
    ret = []
    for i in range(len(A)):
        tmp = []
        for j in range(len(A[0])):
            tmp += [A[i][j] - B[i][j]]
        ret += [tmp]
    return ret

# adds two matrices together
def matadd(A,B):
    return matsub(A, scale(-1,B))
# eigenvalues for a 2x2 matrix
def eig22(A):
    assert len(A) == 2 and len(A[0]) == 2, "not a 2x2 matrix"

    T = float(A[0][0] + A[1][1]) #T = a + d
    D = float(det(A))
    eig1 = T/2 + cmath.sqrt(pow(T,2)/4 - D)
    eig2 = T/2 - cmath.sqrt(pow(T,2)/4 - D)

    return eig1,eig2

# gets a column of a matrix as a 1xn vector
def column(A,i):
    return [transpose(A)[i]]

# gets a row of a matrix as a 1xn vector
def row(A,j):
    return [A[j]]

# returns a list of all permutations of the set {0, 1, 2, ... n}
def permute(n):
    return map(list,list(itertools.permutations(range(n))))

# recursive implementation that fails at len=5 because of recursive limits in python
def permute_rec(arr, start, collect):
    if len(arr) == 1:
        collect += [start + arr]
        return collect
    else:
        return reduce(lambda x,y: x+y, map(lambda m: permute_rec(arr[:m] + arr[m+1:], start + [arr[m]], collect), range(len(arr))))

def normalX(X, cov, mean, covinv=[]):
    if (covinv == []):
        covinv = inv(cov)
    standardized = vecsub(X, mean)
    eexpo = map(lambda x: rowscale([x],0,-0.5)[0], dot([standardized], dot(covinv, transpose([standardized]))))
    ecoeff =  (1 / (((2 * PI_CONST) ** (float(len(cov))/2)) * sqrt(det(cov))))
    return ecoeff * (E_CONST ** det(eexpo))

# returns the inverse of a matrix if it exists
def inv(M):
    assert (hasinv(M)), "matrix is not square or is not invertible"
    n = len(M)
    i = 0
    j = 0
    R = []
    I = identity(n)
    #append the identity matrix
    for k in range(n):
        R += [M[k] + I[k]]
    RT = transpose(R)
    while j != n and i != n:
        curr = RT[j] # get the current column

        #check if the current column is all zeroes
        if reduce(lambda x,y: x and y, map(lambda m: m == 0, curr[i:])):
            j += 1
            i += 1
            continue

        # make the leading value equal to 1
        R = rowscale(R, i, 1/float(R[i][j]))

        # make the other rows have zero in this column
        for x in range(n):
            if x != i:
                R = rowadd(R, x, i, -1 * R[x][j])
        i += 1
        j += 1
        RT = transpose(R)

    # return the transformed identity
    return transpose(RT[n:])

# multiply row i by factor n
def rowscale(M, i, n):
    return M[:i] + [map(lambda x: n * x, M[i])] +  M[i+1:]

# add n * row j to row i and store in row i
def rowadd(M, i, j, n):
    jscale = map(lambda x: n * x, M[j])
    return M[:i] + [map(lambda y: jscale[y] + M[i][y],range(len(jscale)))] + M[i+1:]

#swap rows i and j in M
def rowswap(M, i, j):
    if i < j:
        p, q = i, j
    elif j < i:
        p, q = j, i
    else:
        return M
    return M[:p] + [M[q]] + M[p+1:q] + [M[p]] + M[q+1:]

# returns true if the matrix has an inverse and is square
def hasinv(M):
    return len(M) == len(M[0]) and det(M) != 0

# returns an N x N identity matrix multiplied by a constant k
def kidentity(N,k):
    I = []
    for i in range(N):
	I += [[float(0)]*i +[float(k)] + [float(0)]*(N-i-1)]
    return I

# returns an N x N identity matrix
def identity(N):
    return kidentity(N,1)

# creates a diagonal matrix from the given array
def diag(arr):
    N = len(arr)
    diag = 0
    tmp = []
    for i in range(N):
        Ei = [[0] * N] * i + [[0] * i + [1] + [0] * (N-i-1)] + [[0] * N] * (N-i-1)
        ei = transpose([Ei[i]])
        tmp += [dot(dot(Ei, transpose([arr])), transpose(ei))]
    return reduce(lambda p,q: addM(p,q), tmp)

# adds two same size matrices together
def addM(P, Q):
    assert len(P) == len(Q), "matrices have different row counts"
    assert len(P[0]) == len(Q[0]), "matrices have different column counts"
    return reduce(lambda z,w: z + w, [map(lambda x: map(lambda y: Q[x][y] + P[x][y], range(len(P[x]))),range(len(P)))])

# prints a matrix in a readable way
def prettyM(M):
    print "["
    for i in M:
        print(map(lambda x: round(x, 8),i))
    print "]"

# rounds all the things in a matrix to two decimal places
def roundM(M):
    return map(lambda x: map(lambda y: round(y, 2), x), M)

# returns the mean of all the datapoints in the matrix
def matmean(M):
    return map(mean, transpose(M))


