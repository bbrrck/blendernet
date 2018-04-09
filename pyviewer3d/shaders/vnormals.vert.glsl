#version 330
uniform mat4 model, view, proj, normalmat;
uniform vec3 color;
in vec3 position, normal;
out vec3 vColor;
out vec3 normal_proj;
void main()
{
    gl_Position = proj * view * model * vec4(position,1.0);
    normal_proj = vec3(proj * normalmat * vec4(normal,0.0));
    vColor = color;
}
