###
### Blender script for exporting curve network with normals into a file
###
### (c) 2015-2017 Tibor Stanko [tibor.stanko@gmail.com]
###

### Blender :
# ALT+S to save
# ALT+R to reload
# ALT+P to execute
import os, os.path
import bpy, bmesh
import time, datetime
import numpy as np

NAME = 'lilium'

# dire and file
DIRNAME  = "./"
FILENAME = DIRNAME + NAME + ".blendernet"

# get mesh by the name
obj = bpy.data.objects[NAME]
me = obj.data

# i) select specific groups
GROUPS = ['bd','in1','in2']
# ii) or, get all groups
GROUPS = [g.name for g in obj.vertex_groups]

# get bMesh representation
# switch to edit mode
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh( me )
if hasattr(bm.verts, "ensure_lookup_table"):
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

# number of curves = number of vertex groups
ccount = len(GROUPS)

# init vertices and edges
verts = set()
edges = [set() for ci in range(ccount)]

# iterate through vertex groups
ci=0
for gname in GROUPS:

    # deselect all vertices
    bpy.ops.mesh.select_all(action='DESELECT')

    # set current vertex group to active
    bpy.ops.object.vertex_group_set_active(group=str(gname))

    # select vertices in the the active group
    bpy.ops.object.vertex_group_select()

    # get all currently selected verts
    verts.update( v.index for v in bm.verts if (v.select and not v.hide) )

    # get all currently selected edges
    edges[ci].update( e.index for e in bm.edges if (e.select and not e.hide) )
    ci=ci+1

    # deselect the current vertex group
    bpy.ops.object.vertex_group_deselect()

# show the updates in the viewport
bmesh.update_edit_mesh(me, True)

# export : open file
export = open(FILENAME,"w")

# export : number of vertices
vcount = len(verts)
export.write("v %d\n" % vcount)

## export : vertex data
# init global indexing
gindex = []

# loop over vertices
while len(verts) > 0 :

    # get next vertex
    v = verts.pop()

    # save global index
    gindex.append(v)

    # write position
    co = bm.verts[v].co
    export.write( " %+8.8f %+8.8f %+8.8f  " % (co[0],co[1],co[2]) )

    # write normal
    no = bm.verts[v].normal
    export.write( " %+8.8f %+8.8f %+8.8f\n" % (no[0],no[1],no[2]) )

## export : curve data
# edge tag, interior or boundary
#   >> etag[False] = 'i'
#   >> etag[True]  = 'b'
etag = ['i','b']
ecount = np.zeros(ccount)

# loop over curves (= vertex groups)
for ci in range(ccount) :

    # number of edges in curve c
    ecount[ci] = len(edges[ci])

    # write number of edges
    export.write( 'e %d\n' % ecount[ci] )

    # loop over edges in curve c
    while len(edges[ci]) > 0 :

        # get next edge
        e = edges[ci].pop()

        # is boundary?
        bd = len( bm.edges[e].link_faces ) == 1

        # edge tag : 'b' or 'i'
        export.write( "  "+etag[bd] )

        # write vertex indices
        for v in bm.edges[e].verts :
            # find index in the global index list
            export.write( " %5d" % (gindex.index(v.index)) )

        # next line
        export.write("\n")

# close export file
export.close()

# stats
# this goes into system terminal (not Blender console)
print( datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') )
print("Exporting '" + NAME + "'")
print("%6d curves" % ccount )
print("%6d verts" % vcount )
np.set_printoptions(formatter={'all':lambda x:' '+str(int(x))+' '})
print("%6d edges " % np.sum(ecount), ecount )
print("written to " + FILENAME )
