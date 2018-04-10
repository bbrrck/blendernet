import numpy as np
import pycurves.io as io
import pycurves.matrix as matrix
import pycurves.mesh as mesh
import pycurves.viewer as vi
#-------------------------------------------------------------------------------
def main():
    nV, nN, curves = io.read_net("data/lilium.net")
    V, F = io.read_off("data/lilium.off")
    N = mesh.per_vertex_normals(V,F)
    # render
    viewer = vi.Viewer()
    viewer.add_curves(nV,nN,curves)
    viewer.add_mesh(V,N,F)
    viewer.render()
#-------------------------------------------------------------------------------
if __name__ == "__main__": main()
