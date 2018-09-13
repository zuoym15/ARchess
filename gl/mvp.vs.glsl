#version 430
layout (location = 0) in vec3 normal0;
layout (location = 1) in vec3 aPos;

layout (location = 0) uniform mat4 model;
layout (location = 1) uniform mat4 view;
layout (location = 2) uniform mat4 projection;

out vec3 normal;
out vec3 fragPos;

void main()
{
    fragPos = vec3(model * vec4(aPos, 1.0));
    gl_Position = projection * view * vec4(fragPos, 1.0);
    normal = mat3(transpose(inverse(model))) * normal0;
}