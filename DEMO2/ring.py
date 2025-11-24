import pygame as pg
import moderngl as mgl
import numpy as np
import glm
import math
import random

class Ring:
    """Anell amb volum, Toon Shading i Outline (Cel Shading)."""

    def __init__(self, ctx, camera, radius=0.9, thickness=0.15, height=0.1, segments=24, color=(1.0, 0.2, 0.2),
                 position=glm.vec3(0, 0, 0), altura= 1):
        self.ctx = ctx
        self.camera = camera
        self.color = color
        self.position = position
        self.m_model = glm.translate(glm.mat4(), self.position)
        self.contagion_radius = radius + thickness
        self.altura = altura

        # --- Generació de la geometria 3D (Posició, Normal, Normal Suave) ---
        # Format del buffer: 3f (pos) + 3f (norm) + 3f (smooth_norm)
        vertex_data = []

        r_outer = radius + thickness / 2
        r_inner = radius - thickness / 2

        for i in range(segments):
            theta1 = (i / segments) * 2 * math.pi
            theta2 = ((i + 1) / segments) * 2 * math.pi

            c1, s1 = math.cos(theta1), math.sin(theta1)
            c2, s2 = math.cos(theta2), math.sin(theta2)

            # --- POSICIONES ---
            # 1 = actual, 2 = següent
            # ob = Outer-Bottom, ot = Outer-Top, ib = Inner-Bottom, it = Inner-Top
            
            # Segment actual (1)
            p_ob1 = (c1 * r_outer, 0, s1 * r_outer)
            p_ot1 = (c1 * r_outer, height, s1 * r_outer)
            p_ib1 = (c1 * r_inner, 0, s1 * r_inner)
            p_it1 = (c1 * r_inner, height, s1 * r_inner)

            # Segment següent (2)
            p_ob2 = (c2 * r_outer, 0, s2 * r_outer)
            p_ot2 = (c2 * r_outer, height, s2 * r_outer)
            p_ib2 = (c2 * r_inner, 0, s2 * r_inner)
            p_it2 = (c2 * r_inner, height, s2 * r_inner)

            # --- NORMALES PLANAS (Hard Shading) ---
            n_up   = (0, 1, 0)
            n_down = (0, -1, 0)
            n_out1 = (c1, 0, s1)
            n_out2 = (c2, 0, s2)
            n_in1  = (-c1, 0, -s1)
            n_in2  = (-c2, 0, -s2)

            # --- NORMALES SUAVES (Smooth Shading para Outline) ---
            # Promig de les cares adjacents a cada vèrtex per inflar en diagonal
            sn_ot1 = glm.normalize(glm.vec3(c1, 1, s1))   # Out + Up
            sn_ob1 = glm.normalize(glm.vec3(c1, -1, s1))  # Out + Down
            sn_it1 = glm.normalize(glm.vec3(-c1, 1, -s1)) # In + Up
            sn_ib1 = glm.normalize(glm.vec3(-c1, -1, -s1))# In + Down
            
            sn_ot2 = glm.normalize(glm.vec3(c2, 1, s2))
            sn_ob2 = glm.normalize(glm.vec3(c2, -1, s2))
            sn_it2 = glm.normalize(glm.vec3(-c2, 1, -s2))
            sn_ib2 = glm.normalize(glm.vec3(-c2, -1, -s2))

            # Helper per afegir vèrtex al buffer: Pos + Norm + SmoothNorm
            def add_vert(pos, norm, smooth_norm):
                vertex_data.extend(pos)
                vertex_data.extend(norm)
                vertex_data.extend((smooth_norm.x, smooth_norm.y, smooth_norm.z))

            # --- TRIANGULACIÓ (2 triangles per cara) ---

            # 1. Cara Superior (Top) - Normal UP
            add_vert(p_it1, n_up, sn_it1); add_vert(p_ot2, n_up, sn_ot2); add_vert(p_ot1, n_up, sn_ot1)
            add_vert(p_it1, n_up, sn_it1); add_vert(p_it2, n_up, sn_it2); add_vert(p_ot2, n_up, sn_ot2)

            # 2. Cara Inferior (Bottom) - Normal DOWN
            add_vert(p_ib1, n_down, sn_ib1); add_vert(p_ob1, n_down, sn_ob1); add_vert(p_ob2, n_down, sn_ob2)
            add_vert(p_ib1, n_down, sn_ib1); add_vert(p_ob2, n_down, sn_ob2); add_vert(p_ib2, n_down, sn_ib2)

            # 3. Cara Exterior (Outer) - Normals OUT
            add_vert(p_ob1, n_out1, sn_ob1); add_vert(p_ot1, n_out1, sn_ot1); add_vert(p_ot2, n_out2, sn_ot2)
            add_vert(p_ob1, n_out1, sn_ob1); add_vert(p_ot2, n_out2, sn_ot2); add_vert(p_ob2, n_out2, sn_ob2)

            # 4. Cara Interior (Inner) - Normals IN
            add_vert(p_ib1, n_in1, sn_ib1); add_vert(p_it2, n_in2, sn_it2); add_vert(p_it1, n_in1, sn_it1)
            add_vert(p_ib1, n_in1, sn_ib1); add_vert(p_ib2, n_in2, sn_ib2); add_vert(p_it2, n_in2, sn_it2)

        self.vbo = self.ctx.buffer(np.array(vertex_data, dtype='f4').flatten())
        
        self.shader = self.get_shader()
        self.vao = self.ctx.vertex_array(
            self.shader,
            # 3f pos, 3f normal, 3f smooth_normal
            [(self.vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_smooth_normal')]
        )
        self.update_uniforms()

    def get_shader(self):
        return self.ctx.program(
            vertex_shader='''
                #version 330
                in vec3 in_position;
                in vec3 in_normal;
                in vec3 in_smooth_normal; // Normal per inflar (outline)

                uniform mat4 m_proj;
                uniform mat4 m_view;
                uniform mat4 m_model;
                uniform float outline_width; // Grosor del borde

                out vec3 v_normal;
                out vec3 v_frag_pos;

                void main() {
                    vec3 pos = in_position;

                    // INFLAT PER AL BORDE NEGRE
                    if (outline_width > 0.0) {
                        pos += in_smooth_normal * outline_width;
                    }

                    vec4 world_pos = m_model * vec4(pos, 1.0);
                    v_frag_pos = world_pos.xyz;

                    // Normal matrix per il·luminació (sense inflar)
                    mat3 normal_matrix = mat3(transpose(inverse(m_model)));
                    v_normal = normalize(normal_matrix * in_normal);

                    gl_Position = m_proj * m_view * world_pos;
                }
            ''',
            fragment_shader='''
                #version 330
                in vec3 v_normal;
                in vec3 v_frag_pos;

                uniform vec3 light_pos;
                uniform vec3 view_pos;
                uniform vec3 ring_color;
                uniform float outline_width;

                out vec4 fragColor;

                void main() {
                    // --- PASADA 1: CONTORNO ---
                    if (outline_width > 0.0) {
                        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
                        return;
                    }

                    // --- PASADA 2: TOON SHADING ---
                    
                    vec3 norm = normalize(v_normal);
                    vec3 light_dir = normalize(light_pos - v_frag_pos);
                    vec3 view_dir = normalize(view_pos - v_frag_pos);

                    // 1. Intensitat Base (Lambert)
                    float intensity = max(dot(norm, light_dir), 0.05);
                    intensity = pow(intensity, 0.7); // No lineal

                    float light_level;

                    // 2. Escalera de Nivells
                    if (intensity > 0.95)      light_level = 1.0;
                    else if (intensity > 0.7)  light_level = 0.8;
                    else if (intensity > 0.3)  light_level = 0.6;
                    else if (intensity > 0.1)  light_level = 0.3;
                    else                       light_level = 0.1; // Sombra oscura

                    // 3. Especular Toon (Brillo)
                    vec3 halfwayDir = normalize(light_dir + view_dir);
                    float NdotH = max(dot(norm, halfwayDir), 0.0);
                    float spec = (pow(NdotH, 64.0) > 0.9) ? 0.5 : 0.0;

                    // 4. Combinar
                    vec3 result = ring_color * light_level + vec3(spec);

                    // 5. Correcció Gamma
                    result = pow(result, vec3(1.0 / 2.2));

                    fragColor = vec4(result, 1.0);
                }
            '''
        )

    def update_uniforms(self):
        self.shader['m_proj'].write(self.camera.m_proj)
        self.shader['m_view'].write(self.camera.m_view)
        self.shader['m_model'].write(self.m_model)
        self.shader['ring_color'].value = self.color
        self.shader['view_pos'].value = tuple(self.camera.position)

    def update(self, position):
        position.y = self.altura
        self.position = position

    def destroy(self):
        self.vbo.release()
        self.shader.release()
        self.vao.release()

    def render(self, light_pos):
        self.m_model = glm.translate(glm.mat4(), self.position)

        # Actualitzem uniforms
        self.shader['m_proj'].write(self.camera.m_proj)
        self.shader['m_view'].write(self.camera.m_view)
        self.shader['m_model'].write(self.m_model)
        self.shader['light_pos'].value = light_pos
        self.shader['view_pos'].value = tuple(self.camera.position)
        
        # Activar Culling
        self.ctx.enable(mgl.CULL_FACE)

        # --- PASADA 1: CONTORNO ---
        self.ctx.cull_face = 'front' # Dibuixar cares posteriors
        if 'outline_width' in self.shader:
            # Borde una mica més prim per l'anell
            self.shader['outline_width'].value = 0.02 
        self.vao.render(mode=mgl.TRIANGLES)

        # --- PASADA 2: OBJECTE ---
        self.ctx.cull_face = 'back' # Dibuixar cares anteriors
        if 'outline_width' in self.shader:
            self.shader['outline_width'].value = 0.0
        self.vao.render(mode=mgl.TRIANGLES)