#version 330
uniform mat4 model, view, nmat, proj;
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
out vec3 normal_proj;
void main()
{
    gl_Position = proj * view * model * vec4(position,1.0);
    normal_proj = vec3(proj * nmat * vec4(normal,0.0));
}
