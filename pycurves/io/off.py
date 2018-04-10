import numpy as np
#-------------------------------------------------------------------------------
def read_off(filename):
    f = open(filename,"r")
    assert f.readline() == "OFF\n"
    nv, nf, zero = f.readline().split()
    nv = int(nv)
    nf = int(nf)
    V = np.fromfile(f, dtype=np.float32, count=3*nv, sep=" ")
    V = V.reshape(-1, 3)
    F = np.fromfile(f, dtype=np.uint32, count=4*nf, sep=" ")
    F = F.reshape(-1, 4)[:, 1:]
    return V, F
#-------------------------------------------------------------------------------
