import numpy as np
import pywavefront
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.arrays.vbo import VBO


def compile_shader(vertex_shader, fragment_shader):
    with open(vertex_shader) as f:
        vertex_shader = ''.join(f.readlines())
    with open(fragment_shader) as f:
        fragment_shader = ''.join(f.readlines())
    return shaders.compileProgram(shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                  shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))


class OBJModel:
    vs_source = 'gl/mvp.vs.glsl'
    fs_source = 'gl/phong.fs.glsl'

    def __init__(self, obj_file):
        obj = pywavefront.Wavefront(obj_file, strict=True, collect_faces=True)
        assert len(obj.mesh_list) == 1 and len(obj.mesh_list[0].materials) == 1
        material = obj.mesh_list[0].materials[0]
        #
        self.num_vertices = len(material.vertices) // material.vertex_size
        self.vbo = VBO(np.array(material.vertices, dtype=np.float32), GL_STATIC_DRAW, GL_ARRAY_BUFFER)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo.bind()
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, self.vbo)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, self.vbo + 3 * 4)
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)

        self.shader = compile_shader(self.vs_source, self.fs_source)

    def render(self, model, view, perjection, light_position, color):
        glEnable(GL_DEPTH_TEST)

        glUseProgram(self.shader)

        glUniformMatrix4fv(0, 1, True, model)
        glUniformMatrix4fv(1, 1, True, view)
        glUniformMatrix4fv(2, 1, True, perjection)
        glUniform3fv(3, 1, light_position)
        glUniform3fv(4, 1, color)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.num_vertices)


class BackGroundImage:
    vs_source = 'gl/empty.vs.glsl'
    fs_source = 'gl/image.fs.glsl'

    def __init__(self, width, height):
        vertices = [-1, -1, 0,
                    1, -1, 0,
                    1, 1, 0,
                    1, 1, 0,
                    -1, 1, 0,
                    -1, -1, 0]

        self.num_vertices = len(vertices) // 3
        self.vbo = VBO(np.array(vertices, dtype=np.float32), GL_STATIC_DRAW, GL_ARRAY_BUFFER)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo.bind()
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, self.vbo)
        glEnableVertexAttribArray(0)

        glBindVertexArray(0)

        self.shader = compile_shader(self.vs_source, self.fs_source)

        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGB8, width, height)

        # img = Image.open('board.jpg').transpose(Image.FLIP_TOP_BOTTOM)
        # glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
        #              img.size[0], img.size[1], 0,
        #              GL_RGB, GL_UNSIGNED_BYTE, img.tobytes())

    def set_image(self, image):
        glBindTexture(GL_TEXTURE_2D, self.texture)
        np.flipud(image)

        glTexSubImage2D(GL_TEXTURE_2D, 0,
                        0, 0, image.shape[1], image.shape[0],
                        GL_BGR, GL_UNSIGNED_BYTE, np.flipud(image).tobytes())

    def render(self):
        glDisable(GL_DEPTH_TEST)

        glUseProgram(self.shader)

        glUniform1i(0, 0)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.num_vertices)
