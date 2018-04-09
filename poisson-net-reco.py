###
### Poisson network reconstruction from tangents and distances
###  see Sec. 5 in
###  https://tiborstanko.sk/assets/smi2017/shape-sensors.pdf
###
### (c) 2017-2018 Tibor Stanko [https://tiborstanko.sk]
###
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import pyviewer3d.viewer as vi
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
        curve.lnodes = np.fromfile(f, dtype=int, count=curve.nn, sep=" ")-1
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
def getCurveGlobalIndex(curve,shift):

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

    return idx, shift
#-------------------------------------------------------------------------------
def getNetworkGlobalIndex(curves,gnodes):

    # init shift
    numpts=len(gnodes)

    # get indexing for each curve
    for c in range(0,len(curves)) :
        curves[c].gidx, numpts = getCurveGlobalIndex(curves[c],numpts)

    return curves, numpts
#-------------------------------------------------------------------------------
def buildPoissonSystem(curves,gnodes):

    rowcount=0
    for curve in curves:
        rowcount += curve.nv-1 if curve.cl else curve.nv

    # Laplacian matrix : triplets for sparse
    # three entries per row
    J = np.zeros(3*rowcount)
    I = np.zeros(3*rowcount)
    W = np.zeros(3*rowcount)

    # divergence of tangents
    R = np.zeros((rowcount,3));

    rshift=0
    entrycount=0

    # loop over curves
    for curve in curves :

        # number of unique points in the curve
        k = curve.nv-1 if curve.cl else curve.nv

        # inverse edge lengths
        EL = curve.D[1:] - curve.D[:-1]
        iEL = 1.0 / EL

        # inner vertices of the curve
        if curve.cl:
            # closed curve, all vertices are inner
            # (exclude last since the same as first)
            inner = np.arange(0,k-1)
        else:
            # open curves
            # (exclude first and last)
            inner = np.arange(1,k-2)

        # number of inner vertices
        ni = inner.size

        # loop over inner vertices
        i0 = inner-1
        i1 = inner
        i2 = inner+1

        if curve.cl:
            i0[ i0==-1 ] = k-1
            i2[ i2==k ] = 0

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

        # in : R = divergence of the tangent field
        R[rows,:] = curve.T[i0,:] - curve.T[i1,:]

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

            # bd : R = gradient of the tangent field
            R[rows,:] = curve.T[[0,-1],:];

    # trim
    I = I[:entrycount]
    J = J[:entrycount]
    W = W[:entrycount]
    R = R[:rshift]

    # construct sparse Laplacian
    L = sp.coo_matrix((W, (I, J)))

    return L, R
#-------------------------------------------------------------------------------
def getNetworkEdgeMatrix(curves):
    E = np.zeros((10000,2),dtype=np.uint32)
    eshift=0
    for curve in curves:
        ne = curve.nv-1
        e = eshift + np.arange(0,ne)
        E[e,0] = curve.gidx[:-1]
        E[e,1] = curve.gidx[1:]
        # E[e,2] = curve.id
        # E[e,3] = np.arange(0,ne)
        # E[e,4] = curve.bd
        eshift += ne
    return E[:eshift]
#-------------------------------------------------------------------------------
def getNetworkNormals(curves,numpts):
    N = np.zeros((numpts,3))
    for curve in curves:
        N[curve.gidx,:] = curve.N
    return N
#-------------------------------------------------------------------------------
def testCube():
    V = np.array((
        (-1,-1,-1),
        (+1,-1,-1),
        (+1,+1,-1),
        (-1,+1,-1),
        (-1,-1,+1),
        (+1,-1,+1),
        (+1,+1,+1),
        (-1,+1,+1)
    ),dtype=np.float32)
    N = V
    E = np.array((
        (0,1),
        (1,2),
        (2,3),
        (3,0),
        (4,5),
        (5,6),
        (6,7),
        (7,4),
        (0,4),
        (1,5),
        (2,6),
        (3,7)
    ),dtype=np.uint32)
    return V,N,E
#-------------------------------------------------------------------------------
def main():
    curves, gnodes = readRAWNET("lilium.rawnet")
    curves, numpts = getNetworkGlobalIndex(curves,gnodes)
    L, R = buildPoissonSystem(curves,gnodes)

    # solve in the least-squares sense
    A = L.transpose() * L
    b = L.transpose() * R
    V = spsolve(A,b)
    E = getNetworkEdgeMatrix(curves)
    N = getNetworkNormals(curves,numpts)

    mV, F = vi.readOFF("lilium.off")
    mN = vi.per_vertex_normals(mV,F)

    viewer = vi.Viewer()
    # viewer.add_edges(V,N,E)
    viewer.add_curves(V,N,curves)
    # viewer.add_mesh(mV,mN,F)
    viewer.render()

#-------------------------------------------------------------------------------
if __name__ == "__main__": main()
