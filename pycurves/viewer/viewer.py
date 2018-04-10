import sys
import numpy as np
import OpenGL
from OpenGL.GL import *
from OpenGL.GL.shaders import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
OpenGL.ERROR_ON_COPY = True
OpenGL.ERROR_CHECKING = True
import glfw
from ctypes import c_void_p
import math
import pycurves.matrix
VIEWER_DIR = os.path.dirname(os.path.realpath(__file__)) + "/"
#-------------------------------------------------------------------------------
class Viewer():
    # def cube():
    #     V = np.array([
    #     # front
    #         [-1.0, -1.0, +1.0], [+1.0, -1.0, +1.0],
    #         [+1.0, +1.0, +1.0], [-1.0, +1.0, +1.0],
    #         # back
    #         [-1.0, -1.0, -1.0], [+1.0, -1.0, -1.0],
    #         [+1.0, +1.0, -1.0], [-1.0, +1.0, -1.0]
    #     ],  dtype=np.float32 )
    #     F = np.array([
    #         # front
    #         [0, 1, 2],[2, 3, 0],
    #         # top
    #         [1, 5, 6],[6, 2, 1],
    #         # back
    #         [7, 6, 5],[5, 4, 7],
    #         # bottom
    #         [4, 0, 3],[3, 7, 4],
    #         # left
    #         [4, 5, 1],[1, 0, 4],
    #         # right
    #         [3, 2, 6],[6, 7, 3]
    #     ], dtype=np.uint32 )
    #     return V, F
    #-------------------------------------------------
    def ortho( self, left, right, bottom, top, nearVal, farVal):
        result = np.identity(4)
        result[0,0] =  2. / (right - left)
        result[1,1] =  2. / (top - bottom)
        result[2,2] = -2. / (farVal - nearVal)
        result[0,3] = -(right + left) / (right - left)
        result[1,3] = -(top + bottom) / (top - bottom)
        result[2,3] = -(farVal + nearVal) / (farVal - nearVal)
        return result
    #-------------------------------------------------
    def frustum( self, left, right, bottom, top, nearVal, farVal):
        result = np.zeros([4,4])
        result[0,0] = (2. * nearVal) / (right - left)
        result[1,1] = (2. * nearVal) / (top - bottom)
        result[0,2] = (right + left) / (right - left)
        result[1,2] = (top + bottom) / (top - bottom)
        result[2,2] = -(farVal + nearVal) / (farVal - nearVal)
        result[3,2] = -1.
        result[2,3] = -(2. * farVal * nearVal) / (farVal - nearVal)
        return result
    #-------------------------------------------------
    def lookAt( self, az, el, zoom ) :
        target = np.array([0,0,0])
        direction = np.array([math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el) ])
        right = np.array([ math.sin( az - 0.5*math.pi ),  math.cos( az - 0.5*math.pi ), 0])
        up = np.cross( right, direction )
        origin = target-zoom*direction
        f = (target - origin)
        f = f / np.linalg.norm(f)
        s = np.cross(f,up)
        s = s / np.linalg.norm(s)
        u = np.cross(s,f)
        result = np.identity(4)
        result[0,0] = s[0]
        result[0,1] = s[1]
        result[0,2] = s[2]
        result[1,0] = u[0]
        result[1,1] = u[1]
        result[1,2] = u[2]
        result[2,0] = -f[0]
        result[2,1] = -f[1]
        result[2,2] = -f[2]
        result[0,3] = -np.dot(s,origin)
        result[1,3] = -np.dot(u,origin)
        result[2,3] =  np.dot(f,origin)
        return result
    #-------------------------------------------------

    def show_help( self ) :
        print(" ")
        print("------------------------------------------------")
        print("  Viewer controls : ")
        print("------------------------------------------------")
        print("  left click & drag  = rotate")
        print("  mouse scroll       = zoom")
        print("  pageup/pagedown    = zoom")
        print("  [ E ]              = toggle wireframe")
        print("  [ F ]              = flip normals")
        print("  [ N ]              = toggle color by normals")
        print("  [ S ]              = show normals")
        print("  [Esc]              = quit")
        print(" ")

    def recompute_matrices( self ) :
        fH = math.tan(self.viewAngle / 360. * math.pi) * self.dnear
        fW = fH * self.w / self.h
        if self.use_ortho :
            self.proj = self.ortho( -fW, fW, -fH, fH, self.dnear, self.dfar )
        else :
            self.proj = self.frustum( -fW, fW, -fH, fH, self.dnear, self.dfar )
        self.view = self.lookAt( self.az, self.el, self.zoom )
        self.model= self.scale * np.identity(4)
        self.model[3,3] = 1.0
        self.nmat = self.view
        self.invv = np.linalg.inv( self.view )

    def mouse_button_callback( self, window, button, action, mods ) :
        if button == glfw.MOUSE_BUTTON_LEFT :
            self.rotate = ~self.rotate
            if self.rotate :
                self.xpos, self.ypos = glfw.get_cursor_pos(window)

    def mouse_scroll_callback( self, window, xoff, yoff ) :
        if yoff < 0 :
            self.zoom *= 1.1
        else :
            self.zoom *= 0.9
        self.recompute_matrices()
        return

    def window_resize_callback( self, window, w, h ) :
        self.w = float(w)
        self.h = float(h)
        glViewport(0,0,w,h);
        self.recompute_matrices()

    def key_callback( self, window, key, scancode, action, mods ) :
        if key == glfw.KEY_PAGE_UP :
            self.zoom *= 0.9
            self.recompute_matrices()
            return
        if key == glfw.KEY_PAGE_DOWN :
            self.zoom *= 1.1
            self.recompute_matrices()
            return
        if key == glfw.KEY_E and action == glfw.PRESS :
            self.wire = not self.wire
            return
        if key == glfw.KEY_N and action == glfw.PRESS :
            self.render_normals = not self.render_normals
            return
        if key == glfw.KEY_P and action == glfw.PRESS :
            self.use_ortho = not self.use_ortho
            self.recompute_matrices()
            return
        if key == glfw.KEY_S and action == glfw.PRESS :
            self.show_normals = not self.show_normals
            self.recompute_matrices()
            return
        if key == glfw.KEY_F and action == glfw.PRESS :
            self.nrmls = -self.nrmls
            glBindBuffer( GL_ARRAY_BUFFER, self.vbo[1])
            glBufferData( GL_ARRAY_BUFFER,
                          self.nrmls.size * self.nrmls.itemsize, self.nrmls, GL_STATIC_DRAW)
            return

    def __init__(self,wname="Viewer",size=[1200,800]) :

        self.wname = wname
        self.w = size[0]
        self.h = size[1]

        self.verts = np.empty([0,3],dtype=np.float32)
        self.nrmls = np.empty([0,3],dtype=np.float32)
        self.faces = np.empty([0,3],dtype=np.uint32)
        self.edges = np.empty([0,2],dtype=np.uint32)
        self.curves = np.empty([0,4],dtype=np.uint32)

        self.rotate = False
        self.rotation = np.identity(4)
        self.wire = True
        self.render_normals = False
        self.show_normals = False
        self.use_ortho = False
        self.xpos = 0
        self.ypos = 0

        self.zoom = 2.0
        self.az =  45. / 180. * math.pi
        self.el = 200. / 180. * math.pi

        self.scale = 1.

        self.viewAngle = 30.
        self.dnear = 0.01
        self.dfar = 100.

        if not glfw.init():
            return

        self.show_help()

    # def add_patch(self,X,Y,Z,wireframe=False) :
    #
    #     # extract sampling density
    #     u = X.shape[0]
    #     v = X.shape[1]
    #
    #     # vertices
    #     V = np.empty([u*v,3],dtype=np.float32)
    #     V[:,0] = X.reshape(u*v)
    #     V[:,1] = Y.reshape(u*v)
    #     V[:,2] = Z.reshape(u*v)
    #
    #     # grid indices
    #     i0 = range(0,(u-1)*v)
    #     i1 = range(v, u   *v)
    #     frst = [v*i   for i in range(0,u  )]
    #     last = [v*i-1 for i in range(1,u+1)]
    #
    #     i00 = list( set(i0) - set(last) )
    #     i01 = list( set(i0) - set(frst) )
    #     i10 = list( set(i1) - set(last) )
    #     i11 = list( set(i1) - set(frst) )
    #
    #     # faces
    #     F = np.empty([2*(u-1)*(v-1),3],dtype=np.uint32)
    #     F[:,0] = i00 + i11
    #     F[:,1] = i01 + i10
    #     F[:,2] = i11 + i00
    #
    #     # edges
    #     E = np.empty([4*(u-1)*(v-1),2],dtype=np.uint32)
    #     E[:,0] = i00 + i01 + i11 + i10
    #     E[:,1] = i01 + i11 + i10 + i00
    #
    #     # N = per_vertex_normals(V,F)
    #
    #     # add the mesh
    #     self.add_mesh(V,N,F,E,wireframe)


    def add_mesh(self,V,N,F,E=None,wireframe=False) :

        if E is None :
            f0 = F[:,0]
            f1 = F[:,1]
            f2 = F[:,2]
            E = np.empty([F.shape[0]*3,2],dtype=np.uint32)
            E[:,0] = np.concatenate([f0, f1, f2])
            E[:,1] = np.concatenate([f1, f2, f0])

        # Shift if needed
        shift = self.verts.shape[0]
        F += shift
        E += shift

        # Store
        self.verts = np.concatenate( (self.verts, V.astype(np.float32) ), axis=0 )
        self.nrmls = np.concatenate( (self.nrmls, N.astype(np.float32) ), axis=0 )
        self.edges = np.concatenate( (self.edges, E.astype(np.uint32) ), axis=0 )
        if not wireframe :
            self.faces = np.concatenate( (self.faces, F.astype(np.uint32) ), axis=0 )

    def add_edges(self,V,N,E):
        self.verts = np.concatenate( (self.verts, V.astype(np.float32) ), axis=0 )
        self.nrmls = np.concatenate( (self.nrmls, N.astype(np.float32) ), axis=0 )
        self.edges = np.concatenate( (self.edges, E.astype(np.uint32) ), axis=0 )

    def add_curves(self,V,N,curves):
        self.verts = np.concatenate( (self.verts, V.astype(np.float32) ), axis=0 )
        self.nrmls = np.concatenate( (self.nrmls, N.astype(np.float32) ), axis=0 )
        for curve in curves :
            C = np.zeros((curve.nv-1,4))
            C[1:  , 0] = curve.gidx[ :-2]
            C[ :  , 1] = curve.gidx[ :-1]
            C[ :  , 2] = curve.gidx[1:  ]
            C[ :-1, 3] = curve.gidx[2:  ]

            if curve.cl :
                C[0  ,0] = curve.gidx[  -2]
                C[ -1,3] = curve.gidx[1   ]
            else :
                C[0  ,0] = curve.gidx[ 0   ]
                C[ -1,3] = curve.gidx[   -1]

            self.curves = np.concatenate( (self.curves, C.astype(np.uint32) ), axis=0 )


    def init_glfw(self) :

        # Glfw settings
        glfw.window_hint(glfw.SAMPLES,4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)

        # Create a windowed mode window and its OpenGL context
        self.window = glfw.create_window(self.w, self.h, self.wname, None, None)
        if not self.window:
            glfw.terminate()
            return

        # Make the window's context current
        glfw.make_context_current(self.window)

        # Glfw callbacks
        glfw.set_scroll_callback       ( self.window, self.mouse_scroll_callback )
        glfw.set_mouse_button_callback ( self.window, self.mouse_button_callback )
        glfw.set_window_size_callback  ( self.window, self.window_resize_callback )
        glfw.set_key_callback          ( self.window, self.key_callback )

        # OpenGL settings
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glClearColor(1.0,1.0,1.0,1.0)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glPolygonOffset(1.0,1.0)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    def create_shader_program(self,shadername,geom=False):

        # Create shader program
        program = glCreateProgram(1)

        # Vertex shader
        with open(VIEWER_DIR+"shaders/"+shadername+".vert.glsl", 'r') as VS_file:
            VS_str = VS_file.read()
        vs = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vs,VS_str);
        glCompileShader(vs);
        glAttachShader(program,vs);

        # Fragment shader
        with open(VIEWER_DIR+"shaders/"+shadername+".frag.glsl", 'r') as FS_file:
            FS_str = FS_file.read()
        fs = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fs,FS_str);
        glCompileShader(fs);
        glAttachShader(program,fs);

        # Geometry shader
        if geom :
            with open(VIEWER_DIR+"shaders/"+shadername+".geom.glsl", 'r') as GS_file:
                GS_str = GS_file.read()
            gs = glCreateShader(GL_GEOMETRY_SHADER);
            glShaderSource(gs,GS_str);
            glCompileShader(gs);
            glAttachShader(program,gs);

        # Link shader program
        glLinkProgram(program);

        # Detach and delete shaders
        glDetachShader(program,vs);
        glDeleteShader(vs);
        glDetachShader(program,fs);
        glDeleteShader(fs);
        if geom :
            glDetachShader(program,gs);
            glDeleteShader(gs);

        return program

    def set_matrices(self, uniform) :
        glUniformMatrix4fv( uniform.model, 1, True, self.model )
        glUniformMatrix4fv( uniform.view , 1, True, self.view )
        glUniformMatrix4fv( uniform.nmat , 1, True, self.nmat )
        glUniformMatrix4fv( uniform.proj , 1, True, self.proj )

    def render(self) :

        self.init_glfw()

        # Create shader programs
        programs = []
        programs.append( self.create_shader_program('mesh'))
        programs.append( self.create_shader_program('vnormals',True))
        programs.append( self.create_shader_program('curve',True))

        # Get locations of uniforms
        uniforms = []
        for p in programs :
            u = lambda:0
            u.model = glGetUniformLocation( p, 'model')
            u.view  = glGetUniformLocation( p, 'view')
            u.nmat  = glGetUniformLocation( p, 'nmat')
            u.proj  = glGetUniformLocation( p, 'proj')
            uniforms.append(u)

        uniforms[0].cmode = glGetUniformLocation( programs[0], 'cmode')

        # Scale to uniform length
        diag = self.verts.max(0)-self.verts.min(0)
        aabb = np.linalg.norm(diag)
        if aabb > 0 :
            self.scale = 1. / aabb

        # Snap to centroid
        self.verts -= 0.5*(self.verts.max(0)+self.verts.min(0))

        # Recompute Model, View, Projection matrices
        self.recompute_matrices()

        # Generate vertex arrays
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # Generate vertex buffers
        self.vbo = glGenBuffers(6)

        # 0 : position
        glBindBuffer( GL_ARRAY_BUFFER, self.vbo[0])
        glBufferData( GL_ARRAY_BUFFER,
                      self.verts.size * self.verts.itemsize, self.verts, GL_STATIC_DRAW)

        # 1 : normals
        glBindBuffer( GL_ARRAY_BUFFER, self.vbo[1])
        glBufferData( GL_ARRAY_BUFFER,
                      self.nrmls.size * self.nrmls.itemsize, self.nrmls, GL_STATIC_DRAW)

        # 2 : face indices
        glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, self.vbo[2])
        glBufferData( GL_ELEMENT_ARRAY_BUFFER,
                      self.faces.size * self.faces.itemsize, self.faces, GL_STATIC_DRAW)

        # 3 : edge indices
        glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, self.vbo[3])
        glBufferData( GL_ELEMENT_ARRAY_BUFFER,
                      self.edges.size * self.edges.itemsize, self.edges, GL_STATIC_DRAW)

        # 4 : vertex indices
        self.idx = np.arange(self.verts.size,dtype=np.uint32)
        glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, self.vbo[4])
        glBufferData( GL_ELEMENT_ARRAY_BUFFER,
                      self.idx.size * self.idx.itemsize, self.idx, GL_STATIC_DRAW)

        # 5 : vertex indices
        glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, self.vbo[5])
        glBufferData( GL_ELEMENT_ARRAY_BUFFER,
                      self.curves.size * self.curves.itemsize, self.curves, GL_STATIC_DRAW)

        # Loop until the user closes the window
        while not glfw.window_should_close(self.window) and glfw.get_key(self.window,glfw.KEY_ESCAPE) != glfw.PRESS:

            # Change rotation
            if self.rotate :
                xpos, ypos = glfw.get_cursor_pos( self.window )
                if self.xpos != xpos or self.ypos != ypos :

                    # azimuth
                    self.az -= 10*(self.xpos - xpos) / self.w
                    # elevation
                    self.el -= 10*(self.ypos - ypos) / self.h

                    # store new mouse coords
                    self.xpos = xpos
                    self.ypos = ypos

                    # recompute MVP matrices
                    self.recompute_matrices()

            # Clear the buffers
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Enable vertex arrays
            glEnableVertexAttribArray(0)
            glEnableVertexAttribArray(1)

            # 0 : positions
            glBindBuffer( GL_ARRAY_BUFFER, self.vbo[0] )
            glVertexAttribPointer( 0, 3, GL_FLOAT , False, 0, None )

            # 1 : normals
            glBindBuffer( GL_ARRAY_BUFFER, self.vbo[1] )
            glVertexAttribPointer( 1, 3, GL_FLOAT , False, 0, None )

            #-------------------------------------------------------------------
            # Shader : Mesh
            #-------------------------------------------------------------------
            # Select shader program
            glUseProgram(programs[0])
            # Set uniforms
            self.set_matrices(uniforms[0])
            # Draw wireframe
            if self.wire :
                # Bind element buffer
                glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, self.vbo[3])
                glUniform1i( uniforms[0].cmode, 1 )
                glDrawElements(GL_LINES, self.edges.size, GL_UNSIGNED_INT, None)
            # Draw triangles
            if self.render_normals :
                # Color by normals
                glUniform1i( uniforms[0].cmode, 2 )
            else :
                # Uniform color
                glUniform1i( uniforms[0].cmode, 0 )
            glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, self.vbo[2])
            glDrawElements(GL_TRIANGLES, self.faces.size, GL_UNSIGNED_INT, None)
            #-------------------------------------------------------------------

            #-------------------------------------------------------------------
            # Shader : Vertex normals
            #-------------------------------------------------------------------
            if self.show_normals :
                # Select shader program
                glUseProgram(programs[1])
                # Set uniforms
                self.set_matrices(uniforms[1])
                # Draw
                glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, self.vbo[4])
                glDrawElements( GL_POINTS, self.idx.size, GL_UNSIGNED_INT, None)
            #-------------------------------------------------------------------

            #-------------------------------------------------------------------
            # Shader : Curves
            #-------------------------------------------------------------------
            # Select shader program
            glUseProgram(programs[2])
            # Set uniforms
            self.set_matrices(uniforms[2])
            # Draw
            glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, self.vbo[5])
            glDrawElements( GL_LINES_ADJACENCY, self.curves.size, GL_UNSIGNED_INT, None)
            #-------------------------------------------------------------------

            # Disable vertex arrays
            glDisableVertexAttribArray(0)
            glDisableVertexAttribArray(1)

            # Swap front and back buffers
            glfw.swap_buffers(self.window)

            # Poll for and process events
            glfw.poll_events()

        # Delete stuff
        glDeleteBuffers(6,self.vbo)
        glDeleteVertexArrays(1,(self.vao,))
        for program in programs:
            glDeleteProgram(program)
        glfw.terminate()
