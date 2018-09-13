#version 430

in vec2 fragPos;

layout (location = 0) uniform sampler2D image;

out vec4 fragColor;

void main()
{
    fragColor = texture(image, (fragPos + 1) / 2);
}
