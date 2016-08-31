import numpy as np
from pprint import pprint

A = [[1, 2, 3, 7,3,4,2], [5, 2, 8, 4,2,3,2], [19, 22, 1, 14,3,2,7], [2, 2, 3 ,4,5,9,6],[6,6,6,7,8,8,10],[9,9,9,9,4,3,4],[10,11,12,14,15,2,1]]

print "prin"
pprint(map(lambda y: map(lambda x: round(x, 2),y), np.linalg.eig(np.cov(A,rowvar=False))[1]))
print "Q"
pprint(map(lambda y: map(lambda x: round(x, 2),y), np.linalg.qr(np.cov(A,rowvar=False))[0]))
print "R"
pprint(map(lambda y: map(lambda x: round(x, 2),y), np.linalg.qr(np.cov(A,rowvar=False))[1]))

