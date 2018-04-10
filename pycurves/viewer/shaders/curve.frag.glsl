#version 330
uniform mat4 model, view, nmat, proj;
in vec3 normal_eye;
in vec3 position_eye;
out vec4 color;
void main(void)
{
    const vec4 Ka = vec4( vec3(0.2), 1.0);
    const vec4 Kd = vec4( vec3(0.5), 1.0);
    const vec4 Ks = vec4( vec3(0.2), 1.0);
    const float spec_exp = 35.0f;
    const vec3 light_position_eye = vec3(.0,.0,3.);

    vec3 to_light = normalize(light_position_eye-position_eye);
    vec3 normal = normalize(normal_eye);
    float d = max(0.0,dot(to_light,normal));
    vec3 to_eye = normalize(-position_eye);
    vec3 refl = reflect(-to_light,normal);
    float s = pow(max(dot(to_eye,refl),0.0),spec_exp);
    vec3 phong = vec3(Ka + Kd*d + Ks*s);
    color = vec4(phong,1.0);
}
