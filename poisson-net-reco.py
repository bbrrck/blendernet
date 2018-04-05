import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#-------------------------------------------------------------------------------
class curvestruct():
    id=0        # curve id
    bd=False    # boundary?
    cl=False    # closed?
    nn=0        # number of nodes
    nv=0        # number of datapoints
    lnodes=[]   # local node indices
    gnodes=[]   # global node indices
    D=[]        # data : distances
    T=[]        # data : tangents
    N=[]        # data : normalss
    gidx=[]     # global indices
#-------------------------------------------------------------------------------
def readRAWNET(filename):
    f = open(filename,"r")
    curves = []
    gnodes = set([])
    nc = int(f.readline())
    for c in range(0,nc):

        # init new curve
        curve = curvestruct()

        # curve id
        curve.id = c
        # curve : boundary or interior?
        curve.bd = int(f.readline())
        # curve : closed or open
        curve.cl = int(f.readline())

        # curve : number of nodes
        curve.nn = int(f.readline())
        # curve : local node indices
        curve.lnodes = np.fromfile(f, dtype=int, count=curve.nn, sep=" ")
        # curve : global node indices
        curve.gnodes = np.fromfile(f, dtype=int, count=curve.nn, sep=" ")

        # curve : number of datapoints
        curve.nv = int(f.readline())
        # datapoints (7D)
        DTN = np.fromfile(f, dtype=np.float32, count=7*curve.nv, sep=" ")
        DTN = DTN.reshape(-1, 7)
        # curve : distances
        curve.D = DTN[:,0]
        # curve : tangents
        curve.T = DTN[:,1:4]
        # curve : normals
        curve.N = DTN[:,4:7]

        # store
        curves.append(curve)
        gnodes.update(curve.gnodes)

        # # print stuff
        # print("\n--------\n\nCurve %d" % c)
        # print(curve.id)
        # print(curve.bd)
        # print(curve.cl)
        # print(curve.lnodes)
        # print(curve.gnodes)

    f.close()

    return curves, gnodes
#-------------------------------------------------------------------------------
def indexNetwork(curve,shift):

    # init the indexing
    idx = np.zeros(curve.nv,dtype=int);

    # loop over segments
    for i in range(1,curve.gnodes.size):

        # get local node indices
        n0 = curve.lnodes[i-1]
        n1 = curve.lnodes[i]
        numpts = n1-n0-1

        # store the global indexing
        idx[n0+1:n1] = shift + np.arange(0,numpts)

        # update the shift
        shift = shift + numpts;

    # replace node indices and shift
    idx[curve.lnodes] = curve.gnodes

    # if closed, trim last
    # if curve.cl:
        # idx = np.delete(idx,-1)

    return idx, shift
#-------------------------------------------------------------------------------
def main():
    curves, gnodes = readRAWNET("lilium.rawnet")
    nc = len(curves)
    numnodes = len(gnodes)

    rowcount=0
    for curve in curves:
        rowcount += curve.nv-1 if curve.cl else curve.nv

    rshift=0
    cshift=numnodes

    # Laplacian matrix : triplets for sparse : row index, col index, weights
    # three entries per row
    J = np.zeros(3*rowcount)
    I = np.zeros(3*rowcount)
    W = np.zeros(3*rowcount)
    entrycount=0
    # difference of tangents
    dT = np.zeros((rowcount,3));
    # edge matrix
    E = np.zeros((10000,5));
    # normal matrix
    N = np.zeros((10000,3));
    # counter : total number of edges
    eshift = 0;
    # loop over curves

    for curve in curves:

        curve.gidx, cshift = indexNetwork(curve,cshift)

        print(curve.gidx)

        # fill the normal matrix
        N[curve.gidx,:] = curve.N

        # fill the edge matrix
        ne = curve.nv-1
        e = eshift + np.arange(0,ne)
        E[e,0] = curve.gidx[:-1]
        E[e,1] = curve.gidx[1:]
        E[e,2] = curve.id
        E[e,3] = np.arange(0,ne)
        E[e,4] = curve.bd

        eshift += ne

        # prepare Laplacian matrix
        # inverse edge lengths
        EL = curve.D[1:] - curve.D[:-1]
        iEL = 1.0 / EL

        # inner vertices of the curve
        if curve.cl:
            # closed curve, all vertices are inner
            # (exclude last since the same as first)
            inner = np.arange(0,curve.nv-1)
        else:
            # open curves
            # (exclude first and last)
            inner = np.arange(1,curve.nv-1)

        # number of inner vertices
        ni = inner.size

        # loop over inner vertices
        i0 = inner-1
        i1 = inner
        i2 = inner+1

        i0[ i0==-1 ] = curve.nv-2
        i2[ i2==curve.nv-1 ] = 0

        # in : three entries per row
        entry0 = entrycount + np.arange( 0*ni, 1*ni)
        entry1 = entrycount + np.arange( 1*ni, 2*ni)
        entry2 = entrycount + np.arange( 2*ni, 3*ni)

        # in : increase number of entries
        entrycount = entrycount + 3*ni

        # in : row indices
        rows = rshift + np.arange(0,ni)

        # in : shift rows
        rshift += ni

        # in : I = row indices
        # each row = one equation in the Poisson system
        I[ entry0 ] = rows
        I[ entry1 ] = rows
        I[ entry2 ] = rows

        # in : J = column indices
        # each column = unique vertex in the network
        J[ entry0 ] = curve.gidx[i0]
        J[ entry1 ] = curve.gidx[i1]
        J[ entry2 ] = curve.gidx[i2]

        # in : W = Laplacian weights
        # computed as sums of inverse edge lengths
        W[ entry0 ] = - iEL[i0]
        W[ entry1 ] = + iEL[i0] + iEL[i1]
        W[ entry2 ] =           - iEL[i1]

        # in : dT = gradient of the tangent field
        dT[rows,:] = curve.T[i0,:] - curve.T[i1,:]

        # open curves : boundary conditions
        if curve.cl == 0:

            # bd : two entries per row
            entrybd0 = entrycount + np.array([0,1])
            entrybd1 = entrycount + np.array([2,3])

            # bd : increase number of entries
            entrycount = entrycount + 4;

            # bd : row indices
            rows = rshift + np.array([0,1])

            # bd : shift rows
            rshift += 2;

            # bd : I = boundary conditions
            I[ entrybd0 ] = rows
            I[ entrybd1 ] = rows

            # bd : J = boundary edges
            J[ entrybd0 ] = curve.gidx[[0,-2]]
            J[ entrybd1 ] = curve.gidx[[1,-1]]

            # bd : W = Laplacian weights
            W[ entrybd0 ] = -iEL[[0,-1]]
            W[ entrybd1 ] = +iEL[[0,-1]]

            # bd : dT = gradient of the tangent field
            dT[rows,:] = curve.T[[0,-1],:];

    # trim
    N = N[:cshift,:]
    E = E[:eshift]
    I = I[:entrycount]
    J = J[:entrycount]
    W = W[:entrycount]
    dT = dT[:rshift]

    # construct sparse Laplacian
    L = sp.coo_matrix((W, (I, J)))

    A = L.transpose() * L
    b = L.transpose() * dT

    print(L.shape)
    print(A.shape)
    print(b.shape)

    X = spsolve(A,b)
    # print(X)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0],X[:,1],X[:,2],c='r',marker='o')
    plt.show()

#-------------------------------------------------------------------------------
if __name__ == "__main__": main()
