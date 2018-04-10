import numpy as np
#-------------------------------------------------------------------------------
def normalize_rows(A):
    A = np.nan_to_num(A)
    l = np.sqrt(np.sum(np.square(A),axis=1))
    i = l>0
    A[i,:] /= l[i,np.newaxis]
    return A, l
#-------------------------------------------------------------------------------
