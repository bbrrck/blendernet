import numpy as np
import pyviewer3d.viewer as vi
import pycurvenet.io as io

def main():
    V, F = vi.readOFF("lilium.off")
    N = vi.per_vertex_normals(V,F)

    #nV, nN, curves = 
    io.readNET("lilium.net")

    return

    viewer = vi.Viewer()
    viewer.add_curves(nV,nN,curves)
    viewer.add_mesh(V,N,F)
    viewer.render()

#-------------------------------------------------------------------------------
if __name__ == "__main__": main()
