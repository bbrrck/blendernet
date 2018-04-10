import numpy as np
#-------------------------------------------------------------------------------
def is_member(a,b):
    dim = a.shape
    aa = a.flatten()
    bb = b.flatten()
    tf = np.in1d(aa,bb)
    index = np.array([np.where(bb==i)[0] if t else -1 for i,t in zip(aa,tf)])
    return tf.reshape(dim), index.reshape(dim)
#-------------------------------------------------------------------------------
