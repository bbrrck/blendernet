#version 330
layout (points) in;
layout (triangle_strip, max_vertices=5) out;
in vec3 normal_proj[];
in vec3 vColor[];
out vec3 fColor;
void main()
{
    const float nscale = 0.05f;
    const float thick  = 0.001f;
    // color is fixed for all vertices
    fColor = vColor[0];
    // vertex and its translated in the direction of the vertex normal
    vec4 v0 = gl_in[0].gl_Position;
    vec4 v1 = v0 + nscale * vec4(normal_proj[0],0.0f);
    // in plane
    vec3 l = normalize(vec3(v1-v0));
    vec3 n = cross(l,vec3(0.0,0.0,1.0));
    gl_Position = vec4(vec3(v0)+thick*n,v0.w); EmitVertex();
    gl_Position = vec4(vec3(v1)+thick*n,v1.w); EmitVertex();
    gl_Position = vec4(vec3(v1)-thick*n,v1.w); EmitVertex();
    gl_Position = vec4(vec3(v0)-thick*n,v0.w); EmitVertex();
    gl_Position = vec4(vec3(v0)+thick*n,v0.w); EmitVertex();
    EndPrimitive();
}
