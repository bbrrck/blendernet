###
### Blender script for exporting curve network with normals into a file
###
### (c) 2015-2018 Tibor Stanko [https://tiborstanko.sk]
###

# Blender :
# ALT+S (⌥+S) to save
# ALT+R (⌥+R) to reload
# ALT+P (⌥+P) to execute
#-------------------------------------------------------------------------------
DATANAME = "lilium"
EXPORT_DIR = "./"
#-------------------------------------------------------------------------------
import numpy as np
from collections import Counter
import os, os.path
import bpy, bmesh
import time, datetime
#-------------------------------------------------------------------------------
def ismember(a,b):
    dim = a.shape
    aa = a.flatten()
    bb = b.flatten()
    tf = np.in1d(aa,bb)
    index = np.array([np.where(bb==i)[0] if t else -1 for i,t in zip(aa,tf)])
    return tf.reshape(dim), index.reshape(dim)
#-------------------------------------------------------------------------------
def normalize_rows(A):
    A = np.nan_to_num(A)
    l = np.sqrt(np.sum(np.square(A),axis=1))
    i = l>0
    A[i,:] /= l[i,np.newaxis]
    return A, l
#-------------------------------------------------------------------------------
def getBlenderNetwork(dataname):

    # get mesh by the name
    obj = bpy.data.objects[dataname]
    mesh = obj.data
    # i) select specific groups
    VGroups = ['bd','in1','in2']
    # ii) or, get all groups
    VGroups = [g.name for g in obj.vertex_groups]

    # get bMesh representation
    # switch to edit mode
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(mesh)
    if hasattr(bm.verts, "ensure_lookup_table"):
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()

    # number of curves = number of vertex groups
    nc = len(VGroups)

    # init vertices and edges
    verts = set()
    edges = [set() for ci in range(nc)]

    # iterate through vertex groups
    c=0
    for gname in VGroups:
        # deselect all vertices
        bpy.ops.mesh.select_all(action='DESELECT')
        # set current vertex group to active
        bpy.ops.object.vertex_group_set_active(group=str(gname))
        # select vertices in the the active group
        bpy.ops.object.vertex_group_select()
        # get all currently selected verts
        verts.update( v.index for v in bm.verts if (v.select and not v.hide) )
        # get all currently selected edges
        edges[c].update( e.index for e in bm.edges if (e.select and not e.hide) )
        # deselect the current vertex group
        bpy.ops.object.vertex_group_deselect()
        # move on to the next curve
        c+=1

    # show the updates in the viewport
    bmesh.update_edit_mesh(mesh, True)

    # number of vertices
    nv = len(verts)

    # number of edges in each curve
    ne = np.zeros(nc)
    for ci in range(nc) :
        ne[ci] = len(edges[ci])

    # total number of edges
    ne_total = int(np.sum(ne))

    # init matrices : positions, normals, edges
    V = np.zeros((nv,3))
    N = np.zeros((nv,3))
    E = np.zeros((ne_total,4),dtype=int)

    # vertices : loop
    # store global (mesh) indices
    gindex = []
    r=0
    while len(verts) > 0 :
        # get next vertex
        v = verts.pop()
        # save global index
        gindex.append(v)
        # store position
        V[r,0] = bm.verts[v].co[0]
        V[r,1] = bm.verts[v].co[1]
        V[r,2] = bm.verts[v].co[2]
        # store normal
        N[r,0] = bm.verts[v].normal[0]
        N[r,1] = bm.verts[v].normal[1]
        N[r,2] = bm.verts[v].normal[2]
        # move on
        r+=1

    # edges : loop over curves (= vertex groups)
    r=0
    for ci in range(nc) :
        # loop over edges in curve c
        while len(edges[ci]) > 0 :
            # get next edge
            e = edges[ci].pop()
            # is boundary?
            bd = len( bm.edges[e].link_faces ) == 1
            # vertices : indices in the mesh
            v0 = bm.edges[e].verts[0].index
            v1 = bm.edges[e].verts[1].index
            # vertices : indices in the curve network
            g0 = gindex.index(v0)
            g1 = gindex.index(v1)
            # store edge data : curve index, is boundary, vertex indices
            E[r,0] = ci
            E[r,1] = bd
            E[r,2] = g0
            E[r,3] = g1
            # move on
            r+=1

    # stats into system terminal (not Blender console)
    print( datetime.datetime.fromtimestamp(time.time()).strftime('\n%Y-%m-%d %H:%M:%S') )
    print("%6d curves" % nc )
    print("%6d verts"  % nv )
    print("%6d edges " % ne_total, ne )

    return V,N,E
#-------------------------------------------------------------------------------
def writeBLENDERNET(filename,V,N,E):
    assert(V.shape[1] == 3)
    assert(N.shape[1] == 3)
    assert(E.shape[1] == 4)
    nv = V.shape[0]
    ne = E.shape[0]
    assert(N.shape[0] == nv)
    VN = np.hstack((V,N))
    try:
        f = open(EXPORT_DIR+filename,"wb+")
    except IOError:
        print("Could not open %s" % EXPORT_DIR+filename)
    # number of vertices and edges
    np.savetxt(f,np.array([nv,ne]).reshape(1,2),fmt='%d',delimiter=' ')
    # vertices : positions and normals
    np.savetxt(f,VN,fmt=' %+10.8f',delimiter=' ')
    # edges
    np.savetxt(f,E,fmt='%d',delimiter=' ')
    f.close()
#-------------------------------------------------------------------------------
def readBLENDERNET(filename):
    # output :
    # V – vertex positions
    # N – vertex normals
    # E – edge matrix, with columns
    #     [0] curve index
    #     [1] bool, boundary (1) or interior edge (0)
    #     [2] index of the 1st vertex
    #     [3] index of the 2nd vertex
    f = open(EXPORT_DIR+filename,"r")
    nv, ne = f.readline().split()
    nv = int(nv)
    ne = int(ne)
    VN = np.fromfile(f, dtype=np.float32, count=6*nv, sep=" ")
    VN = VN.reshape(-1, 6)
    V = VN[:,:3]
    N = VN[:,3:]
    E = np.fromfile(f, dtype=np.uint32, count=4*ne, sep=" ")
    E = E.reshape(-1, 4)
    f.close()
    return V,N,E
#-------------------------------------------------------------------------------
def sortCurve(c,E_net,nodes):
    # print curve number
    print("  curve %d" % c)
    # edges of this curve : bool
    e = E_net[:,0]==c
    # edges in this curve : indices
    E = E_net[e,2:]
    # check if boundary curve by looking at first edge's flag
    # assume curve either bd or int, edges NOT treated individually!
    bd = E_net[e,1][0]
    # init free edges
    free = np.ones(E.shape, dtype=bool)
    # which edges contain nodes and which nodes
    tf,index = ismember(E,nodes)
    # get node counts in the curve
    node_counter = Counter(E[tf])
    # get node with minimal valence
    v = min(node_counter)
    # init the curve indices
    sorted = np.array(v, dtype=int)

    iter=0
    while True :
        iter += 1
        # break infinite loop if needed
        if iter > 1e6 :
            break
        # find current vertex in the edge matrix of the curve
        edges,tmp = ismember(E,v)
        # get free, unprocessed edges which contain current vertex
        i,j = np.where(free & edges)
        # the end of the curve is reached if no such edges are found
        if i.size == 0 :
            print("  ...break, iter=%d" % iter)
            break
        # edge index
        e = i[0]
        # vertex index in the edge (0 or 1)
        k = j[0]
        # get the other vertex of the edge
        v = E[e,(k+1)%2]
        # store the new vertex
        sorted = np.append(sorted,v)
        # mark this edge as used
        free[e,:] = False

    return sorted, bd
#-------------------------------------------------------------------------------
def getNetworkNodes(E_net):
    # count vertices in the edges
    node_counter = Counter(E_net[:,2:].reshape(1,-1).flatten())
    # get nodes (valence > 2)
    return np.unique( [n for n in node_counter.elements() if node_counter[n] > 2] )
#-------------------------------------------------------------------------------
def getCurveNodes(sorted,nodes):
    # get the placement of nodes
    mask, nodeglobal = ismember(sorted,nodes)
    # local indices of nodes
    l = np.concatenate( np.where(mask) )
    # global indices of nodes
    g = np.concatenate( nodeglobal[mask] )
    # the curve is closed if it has identical first and last node
    c = g[0]==g[-1]
    return l, g, c
#-------------------------------------------------------------------------------
def getCurveTangentsAndDistances(V,closed):
    # generate tangents
    # closed curves: 1st and last vertex are the same -> next line is valid
    T = V[1:,:] - V[0:-1,:]
    # normalize and get lengths
    T, L = normalize_rows(T)
    # distances from lengths
    D = np.cumsum(L)
    D = np.insert(D, 0, 0, axis=0)
    # closed curve: duplicate the first tangent
    if closed :
        T = np.vstack((T,T[0,:]))
    # open curve: duplicate the last tangent
    else :
        T = np.vstack((T,T[-1,:]))
    return T, L, D
#-------------------------------------------------------------------------------
def writeCurveRAWNET(f,c,V_net,N_net,E_net,nodes):
    # sort curve points
    sorted, bd = sortCurve(c,E_net,nodes)
    # number of points
    nv = sorted.size
    # get nodes
    lnodes, gnodes, closed = getCurveNodes(sorted,nodes)
    # curve positions and normals
    V = V_net[sorted,:]
    N = N_net[sorted,:]
    # get tangents and distances from positions
    T, L, D = getCurveTangentsAndDistances(V,closed)
    # double check if all matrices have the right dimensions
    assert(D.shape[0] == nv)
    assert(T.shape[0] == nv)
    assert(T.shape[1] == 3)
    assert(N.shape[0] == nv)
    assert(N.shape[1] == 3)
    # curve datapoints : 7D (distance, tangent, normal)
    DTN = np.hstack((D.reshape(-1,1),T,N))
    # curve : boundary or interior, closed or open, number of nodes
    np.savetxt(f,np.array([bd,closed,lnodes.size]).reshape(-1,1),fmt='%d')
    # curve : nodes, local indexing
    np.savetxt(f,lnodes.reshape(1,-1),fmt='%d',delimiter=' ')
    # curve : nodes, global indexing
    np.savetxt(f,gnodes.reshape(1,-1),fmt='%d',delimiter=' ')
    # curve : number of datapoints
    np.savetxt(f,np.array([nv]),fmt='%d')
    # curve : datapoints
    np.savetxt(f,DTN,fmt=' %+10.8f',delimiter=' ')
#-------------------------------------------------------------------------------
def writeRAWNET(filename,V,N,E):
    # number of curves
    nc = np.max(E[:,0])+1
    # network nodes
    nodes = getNetworkNodes(E)
    # open export file
    fullpath = EXPORT_DIR+filename
    try:
        f = open(fullpath,"wb+")
    except IOError:
        print("IOError: Could not open %s" % fullpath)
    print("Opened %s" % fullpath)
    # write number of curves
    np.savetxt(f,np.array([nc]),fmt='%d')
    # loop over curves
    for c in range(0,nc) :
        # write current curve
        writeCurveRAWNET(f,c,V,N,E,nodes)
    # close the rawnet file
    f.close()
    print("Closed %s\n" % fullpath)
#-------------------------------------------------------------------------------
def main():
    if not os.path.isdir(EXPORT_DIR):
        print("ERROR : invalid export dir %s" % EXPORT_DIR)
        return
    print("----------------")
    # write data to a blendernet file
    V, N, E = getBlenderNetwork(DATANAME)

    # export as blendernet
    # writeBLENDERNET(DATANAME+".bnet",V,N,E)

    # read data from a blendernet file
    # V, N, E = readBLENDERNET(dataname+".bnet")

    # export as rawnet
    writeRAWNET(DATANAME+".rawnet",V,N,E)
#-------------------------------------------------------------------------------
if __name__ == "__main__": main()
