import numpy as np

class curvestruct():
    bd=False    # boundary?
    gidx=[]     # global indices

def readNET(filename):
    f = open(filename,"r")
    nv = int(f.readline())
    X = np.fromfile(f, dtype=np.float32, count=6*nv, sep=" ").reshape(-1,6)
    V = X[:,:3]
    N = X[:,3:]
    ne_total = int(f.readline())
    ecount = 0
    while ecount < ne_total :
        curve = curvestruct()
        ln = f.readline().split()
        ne = int(ln[0])
        bd = int(ln[0])
        curve.bd = bd
        # TODO THIS IS ALL WRONG
        curve.gidx = np.fromfile(f,dtype=np.int32, count=ne+1)
        print(curve.gidx)
        ecount += ne
        print(ne)
    f.close()


def writeNET(filename,V,N,curves,gnodes):
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
    np.savetxt(f,np.array([ne]),fmt='%d',delimiter=' ')
    # edges
    for curve in curves :
        np.savetxt(f,np.array([curve.nv-1,curve.bd]).reshape(-1,2),fmt='%d',delimiter=' ')
        np.savetxt(f,np.array(list(curve.gidx)).reshape(1,-1),fmt='%d',delimiter=' ')
    f.close()
