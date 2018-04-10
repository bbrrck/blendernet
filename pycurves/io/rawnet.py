import numpy as np
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
def read_rawnet(filename):
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
    f.close()
    return curves, gnodes
#-------------------------------------------------------------------------------
def write_rawnet(filename):
    pass
#-------------------------------------------------------------------------------
