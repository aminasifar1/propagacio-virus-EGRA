import pygame as pg
import moderngl as mgl
import numpy as np
import glm
import math

class Escenario:
    def __init__(self, ctx, camera, vertex_data, bounding_box, texture_path=None):
        self.ctx = ctx
        self.camera = camera
        self.bounding_box = bounding_box
        
        # 1. Cargar Textura
        self.texture = self.load_texture(texture_path)
        
        # 2. Buffer y VAO
        # Ahora el buffer contiene: Pos(3f) + Norm(3f) + UV(2f) = 8 floats por vértice
        self.vbo = self.ctx.buffer(vertex_data)
        
        self.shader = self.get_shader()
        
        # El formato es '3f 3f 2f' -> in_position, in_normal, in_texcoord
        self.vao = self.ctx.vertex_array(
            self.shader,
            # El formato ahora es '3f 3f 2f 3f' (Pos, Norm, UV, Color)
            [(self.vbo, '3f 3f 2f 3f', 'in_position', 'in_normal', 'in_texcoord', 'in_color')]
        )
        
        self.m_model = glm.mat4()
        
        # Luces
        self.light_angle = 0.0
        self.light_radius = 15.0
        self.light_speed = 0.0005

    def load_texture(self, path):
        """Carga una imagen y crea una textura ModernGL."""
        if not path:
            # Textura blanca por defecto si no hay archivo
            texture = self.ctx.texture((2, 2), 3, data=b'\xff\xff\xff' * 4)
        else:
            try:
                # Cargar con Pygame
                img = pg.image.load(path).convert()
                # Voltear verticalmente (OpenGL espera el origen abajo-izq)
                img = pg.transform.flip(img, False, True)
                
                # Crear textura
                texture = self.ctx.texture(img.get_size(), 3, pg.image.tostring(img, 'RGB'))
                # Generar mipmaps para que se vea bien de lejos
                texture.build_mipmaps()
                texture.filter = (mgl.LINEAR_MIPMAP_LINEAR, mgl.LINEAR)
            except Exception as e:
                print(f"Error cargando textura {path}: {e}")
                texture = self.ctx.texture((2, 2), 3, data=b'\xff\x00\x00' * 4) # Roja si falla
        
        return texture

    def get_shader(self):
        return self.ctx.program(
            vertex_shader='''
                #version 330
                in vec3 in_position;
                in vec3 in_normal;
                in vec2 in_texcoord;
                in vec3 in_color;    // <--- Recibimos el color del material
                
                uniform mat4 m_proj;
                uniform mat4 m_view;
                uniform mat4 m_model;
                
                out vec3 v_normal;
                out vec3 v_frag_pos;
                out vec2 v_uv;
                out vec3 v_color;    // <--- Lo pasamos al fragment
                
                void main() {
                    vec4 world_pos = m_model * vec4(in_position, 1.0);
                    v_frag_pos = world_pos.xyz;
                    mat3 normal_matrix = mat3(transpose(inverse(m_model)));
                    v_normal = normalize(normal_matrix * in_normal);
                    
                    v_uv = in_texcoord;
                    v_color = in_color; // Pasamos el color
                    
                    gl_Position = m_proj * m_view * world_pos;
                }
            ''',
            fragment_shader='''
                #version 330
                in vec3 v_normal;
                in vec3 v_frag_pos;
                in vec2 v_uv;
                in vec3 v_color;
                
                uniform vec3 light_pos;
                uniform vec3 view_pos;
                uniform sampler2D u_texture;
                
                out vec4 fragColor;
                
                void main() {
                    // 1. Obtener color base (Textura * Color Material)
                    vec4 texColor = texture(u_texture, v_uv);
                    vec3 baseColor = texColor.rgb * v_color;

                    // 2. Configuración de iluminación (Ajustado para parecerse más a Blender)
                    vec3 norm = normalize(v_normal);
                    vec3 light_dir = normalize(light_pos - v_frag_pos);
                    vec3 view_dir = normalize(view_pos - v_frag_pos);

                    // --- Ambient ---
                    // Subimos un poco el ambient para que las sombras no sean negras puras
                    // Blender usa iluminación ambiental global, aquí simulamos con 0.5
                    vec3 ambient = 0.5 * baseColor;
                    
                    // --- Diffuse ---
                    float diff = max(dot(norm, light_dir), 0.0);
                    vec3 diffuse = diff * baseColor * 0.6; // Bajamos difusa para no quemar
                    
                    // --- Specular ---
                    // El "brillo" del plástico/madera
                    vec3 reflect_dir = reflect(-light_dir, norm);
                    // Blinn-Phong es mejor que Phong (usamos halfwayDir) para brillos más naturales
                    vec3 halfwayDir = normalize(light_dir + view_dir);  
                    float spec = pow(max(dot(norm, halfwayDir), 0.0), 32.0);
                    vec3 specular = spec * vec3(0.2); // Especular suave

                    vec3 result = ambient + diffuse + specular;

                    // --- 3. CORRECCIÓN GAMMA (CRUCIAL) ---
                    // Los monitores esperan sRGB. Convertimos de Lineal a sRGB.
                    // Esto "aclara" los tonos medios y hace que los colores "revivan".
                    float gamma = 2.2;
                    result = pow(result, vec3(1.0 / gamma));

                    fragColor = vec4(result, 1.0);
                }
            '''
        )

    def update_light_position(self):
        self.light_angle += self.light_speed * (self.app.delta_time * 1000)
        light_x = self.light_radius * math.cos(self.light_angle)
        light_z = self.light_radius * math.sin(self.light_angle)
        return (light_x, 12.0, light_z)

    def render(self):
        # 1. Usar (bindear) la textura en la unidad 0
        self.texture.use(location=0)
        
        # 2. Actualizar uniforms
        self.shader['m_model'].write(self.m_model)
        self.shader['light_pos'].value = self.update_light_position()
        self.shader['view_pos'].value = tuple(self.camera.position)
        self.shader['m_proj'].write(self.camera.m_proj)
        self.shader['m_view'].write(self.camera.m_view)
        
        # 3. Asignar el sampler a la unidad 0
        self.shader['u_texture'].value = 0

        self.vao.render(mode=mgl.TRIANGLES)