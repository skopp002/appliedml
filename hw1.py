import numpy as np
from numpy import linalg as LA

m=[[1,-4,2],[-4,1,-2],[2,-2,-2]]
#print("determinant is" ,np.linalg.det(m))


M = np.array(m)
w, v = LA.eig(M)

#print(w)

#print(v)

accs=np.array([16,24,16,12,16,11,14,15,9,14,7])
print(accs.mean())
print(np.std(accs))
