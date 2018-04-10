import numpy as np
import pycurves.matrix as mat
#-------------------------------------------------------------------------------
# Code for computing per-vertex normals from
# https://sites.google.com/site/dlampetest/python/calculating-normals-of-a-triangle-mesh-using-numpy
#-------------------------------------------------------------------------------
def per_vertex_normals(V,F):
    N = np.zeros( V.shape, dtype=V.dtype )
    T = V[F]
    FN = np.cross( T[::,1]-T[::,0],  T[::,2]-T[::,0] )
    N[ F[:,0] ] += FN
    N[ F[:,1] ] += FN
    N[ F[:,2] ] += FN
    return mat.normalize_rows(N)[0]
#-------------------------------------------------------------------------------
