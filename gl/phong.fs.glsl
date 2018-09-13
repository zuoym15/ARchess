#version 430
in vec3 normal;
in vec3 fragPos;

layout (location = 3) uniform vec3 lightPos;
layout (location = 4) uniform vec3 color;

out vec4 fragColor;

void main()
{
    vec3 norm = normalize(normal);
    vec3 lightDir = normalize(lightPos - fragPos);

    float diff = max(dot(norm, lightDir), 0.0);
    fragColor = vec4(diff * color + pow(diff, 5) * vec3(1), 1.0);
}