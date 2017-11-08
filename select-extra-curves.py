###
### Blender script for selecting curves, which connect extraordinary vertices.
### Works only for specific meshes with 'aligned' extraordinary vertices.
###
### (c) 2016-2017 Tibor Stanko [tibor.stanko@gmail.com]
###

### Blender :
# ALT+S to save
# ALT+R to reload
# ALT+P to execute
import os
import os.path
import bpy,bmesh
import time,datetime
import numpy as np

### bmesh stuff
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(bpy.context.object.data)
if hasattr(bm.verts, "ensure_lookup_table"):
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

# vertex and face counts
vc = len(bm.verts)
fc = len(bm.faces)

# extract faces as numpy matrix
F = np.fromiter((x.index for f in bm.faces for x in f.verts),dtype=np.int64)
F.shape = (fc,4)

# get vertex valences
val = np.empty([1,vc])
for v in range(0,vc) :
    (i,j) = (F==v).nonzero()
    val[0,v] = len(i)

# find extraordinary vertices
ex = (val[0]!=4).nonzero()
ex = np.unique(ex)

for v0 in ex :
    (i,j) = (F==v0).nonzero()
    bm.verts[v0].select = True

    curve = [];
    curve.append(v0)
    bm.verts[v0].select = True
    bpy.context.scene.objects.active = bpy.context.scene.objects.active

    nei = np.unique(F[i,np.fmod([j-1,j+1],4)].reshape(-1,1))
    opp = np.unique(F[i,np.fmod([j+2],4)].reshape(-1,1))

    opp0 = opp
    v00 = v0

    iter = 0

    for v1 in nei :
        curve.append(v1)
        bm.verts[v1].select = True
        bpy.context.scene.objects.active = bpy.context.scene.objects.active

        opp = opp0
        v0 = v00
        while 1 :
            iter+=1
            if iter > 1000 :
                print('too many iterations')
                break

            #print(v1)

            if v1 in ex :
                print('reached next extra vertex')
                break

            (i,j) = (F==v1).nonzero()
            nei = np.unique(F[i,np.fmod([j-1,j+1],4)].reshape(-1,1))

            check = opp
            check = np.append(opp,v0)

            v0 = v1
            v1 = np.setdiff1d(nei,check)

            if len(v1) > 1 :
                print('len(v1) > 1')
                print(v0)
                print(v1)
                print(nei)
                print(check)
                break

            curve.append(v1)
            bm.verts[v1].select = True
            bpy.context.scene.objects.active = bpy.context.scene.objects.active

            opp = np.unique(F[i,np.fmod([j+2],4)].reshape(-1,1))

#trigger viewport update
bpy.context.scene.objects.active = bpy.context.scene.objects.active
