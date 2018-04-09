#version 330
uniform float gamma;
in vec3 fColor;
out vec4 oColor;
void main()
{
    vec3 color = pow(fColor,vec3(1./gamma));
    oColor = vec4(color,1.0);
}
