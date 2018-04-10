import numpy as np
#-------------------------------------------------------------------------------
class curvestruct():
    pass
#-------------------------------------------------------------------------------
def read_net(filename):
    f = open(filename,"r")
    nv = int(f.readline())
    X = np.fromfile(f, dtype=np.float32, count=6*nv, sep=" ").reshape(-1,6)
    V = X[:,:3]
    N = X[:,3:]
    nc = int(f.readline())
    curves = []
    for c in range(nc) :
        curve = curvestruct()
        curve.bd = int(f.readline())
        curve.gidx = np.array( f.readline().split(), dtype=np.uint32 )
        curve.nv = curve.gidx.size
        curve.cl = curve.gidx[0]==curve.gidx[-1]
        curves.append( curve )
    f.close()
    return V, N, curves
#-------------------------------------------------------------------------------
def write_net(filename,V,N,curves,gnodes):
    assert(V.shape[1] == 3)
    assert(N.shape[1] == 3)
    nv = V.shape[0]
    nn = len(gnodes)
    assert(N.shape[0] == nv)
    VN = np.hstack((V,N))
    try:
        f = open(filename,"wb+")
    except IOError:
        print("Could not open %s" % filename)
    # number of edges
    ne=0
    for curve in curves:
        ne += curve.nv-1
    # number of vertices and nodes
    np.savetxt(f,np.array([nv]),fmt='%d',delimiter=' ')
    # vertices : positions and normals
    np.savetxt(f,VN,fmt=' %+10.8f',delimiter=' ')
    # number of vertices and edges
    np.savetxt(f,np.array([len(curves)]),fmt='%d',delimiter=' ')
    # edges
    for curve in curves :
        np.savetxt(f,np.array([curve.bd]),fmt='%d')
        np.savetxt(f,curve.gidx.reshape(1,-1),fmt=' %d',delimiter='')
    f.close()
#-------------------------------------------------------------------------------
