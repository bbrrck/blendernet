#version 330
layout (lines_adjacency) in;
layout (triangle_strip,max_vertices=24) out;

uniform mat4 model, view, nmat, proj;

in vec3 position_world[];
in vec3 normal_world[];

out vec3 position_eye;
out vec3 normal_eye;

vec4 prism4[8];
vec3 prism_eye[8];
vec3 normals_eye[8];

void emit_face(int a,int b,int c,int d)
{
    position_eye = prism_eye[a];
    normal_eye = normals_eye[a];
    gl_Position = prism4[a];
    EmitVertex();

    position_eye = prism_eye[b];
    normal_eye = normals_eye[b];
    gl_Position = prism4[b];
    EmitVertex();

    position_eye = prism_eye[c];
    normal_eye = normals_eye[c];
    gl_Position = prism4[c];
    EmitVertex();

    position_eye = prism_eye[d];
    normal_eye = normals_eye[d];
    gl_Position = prism4[d];
    EmitVertex();

    EndPrimitive();
}

void main(void)
{
    vec3 p0=position_world[0], p1=position_world[1], p2=position_world[2], p3=position_world[3];
    vec3 n0=normal_world[0], n1=normal_world[1], n2=normal_world[2], n3=normal_world[3];

    vec3 t01 = normalize(p1-p0);
    vec3 t12 = normalize(p2-p1);
    vec3 t23 = normalize(p3-p2);

    vec3 b1;
    vec3 b2;

    if( abs(dot(t01,t12)) < 0.5 ) { // special case, 1 is a corner
        b1 = t01;
    } else {
        vec3 t02 = normalize(p2-p0);
        b1 = cross(n1,t02);
    }

    if( abs(dot(t12,t23)) < 0.5 ) { // special case, 2 is a corner
        b2 = -t23;
    } else {
        vec3 t13 = normalize(p3-p1);
        b2 = cross(n2,t13);
    }

    b1 = normalize(b1);
    b2 = normalize(b2);

    normals_eye[0] = -b1 -n1;
    normals_eye[1] = +b1 -n1;
    normals_eye[2] = +b1 +n1;
    normals_eye[3] = -b1 +n1;
    normals_eye[4] = -b2 -n2;
    normals_eye[5] = +b2 -n2;
    normals_eye[6] = +b2 +n2;
    normals_eye[7] = -b2 +n2;

    float curvethickness=1.0;
    float thick = 0.002*curvethickness;
    // transform
    vec3 p = p1;
    for( int i=0; i<8; i++ ) {
        vec3 p;
        if( i<4 )
            p = p1;
        else
            p = p2;
        normals_eye[i] = normalize( normals_eye[i] );
        prism_eye[i] = vec3(view * vec4(p + thick*normals_eye[i],1.0));
        prism4[i] = proj * vec4(prism_eye[i],1.0);
    }

    emit_face(0,1,3,2);
    emit_face(4,5,7,6);
    emit_face(0,1,4,5);
    emit_face(1,2,5,6);
    emit_face(2,3,6,7);
    emit_face(3,0,7,4);
}
