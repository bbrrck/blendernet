#version 330
uniform mat4 model, view, nmat, proj;

layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;

out vec3 position_world;
out vec3 normal_world;

// out vec3 position_eye;
// out vec3 normal_eye;

void main()
{
    position_world = vec3(model*vec4(position,1.0));
    normal_world = vec3(nmat*vec4(normal,0.0));

    // position_eye = vec3 (view * model * vec4 (position, 1.0));
    // normal_eye = vec3 (nmat * vec4 (normal,0.0));
    // normal_eye = normalize(normal_eye);
    // gl_Position = proj * vec4 (position_eye,1.0);
}
