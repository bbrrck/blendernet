import numpy as np
import pyviewer3d.viewer as vi

def main():
    V, F = vi.readOFF("lilium.off")
    N = vi.per_vertex_normals(V,F)
    viewer = vi.Viewer()
    viewer.add_mesh(V,N,F)
    viewer.render()

#-------------------------------------------------------------------------------
if __name__ == "__main__": main()
