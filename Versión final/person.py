import pygame as pg
import moderngl as mgl
import numpy as np
import glm
import sys
import time
import math
import random
from facultat import Sala, Clase, Pasillo
from escenario import Escenario
from camera import Camera
from ring import Ring
from animacion import PuffSystem


class Person:
    def __init__(self, motor, ctx, camera, vertex_data, facultad, schedule=None, sala='pasillo', ground_y=0.1, position=None, group=None):
        self.ctx = ctx
        self.camera = camera

        self.vbo = self.ctx.buffer(vertex_data)

        self.mundo = facultad
        self.schedule = schedule
        self.present = True
        self.sala = None
        self.ring = None
        self.puff = PuffSystem(self.ctx, self.camera)
        self.motor = motor
        self.group_id = group
        self.preclass_plan = None

        # Personalizacion
        self.height = min(max(np.random.normal(1.75, 0.1), 1.6), 1.9)
        self.speed = np.random.normal(0.5, 0.05)

        # Starting position and destiny
        # self._cambiar_sala(sala)
        if position:
            self.position = position
        else:
            p0 = glm.vec3(-26.55, -0.15, -17.65)
            p1 = glm.vec3(-26.55, -0.15, -46.85)
            p2 = glm.vec3(3.35, -0.15, -46.85)
            L01 = glm.length(p1 - p0)
            L12 = glm.length(p2 - p1)
            L = L01 + L12
            if L == 0:
                return glm.vec3(p0)

            r = random.random() * L
            a, b = (p0, p1) if r < L01 else (p1, p2)
            t = random.random()
            self.position = a + t * (b - a)
            
        self.m_model = glm.translate(glm.mat4(1.0), self.position)

        self.rotation_angle = 0.0

        # Estado del movimiento
        self.en_movimiento = False
        self.pasos = []
        self.indice_paso = 0
        self.camino_actual = []
        self.indice_camino = 0
        self.sentado = False
        self.objetivo_asiento = None
        self.destino_sala = None

        # Bounding Box (AABB)
        self.bb_half = glm.vec3(0.25, self.height, 0.25)
        self.shader_lines = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec3 in_pos;
                uniform mat4 m_model;
                uniform mat4 m_view;
                uniform mat4 m_proj;
                void main() {
                    gl_Position = m_proj * m_view * m_model * vec4(in_pos, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                uniform vec3 u_color;
                out vec4 f_color;
                void main() {
                    f_color = vec4(u_color, 1.0);
                }
            '''
        )
        
        hx = self.bb_half.x; hy = self.bb_half.y; hz = self.bb_half.z
        bb_verts = np.array([
            -hx, 0.0, -hz,   hx, 0.0, -hz, hx, 0.0, -hz,    hx, 0.0, hz,
            hx, 0.0, hz,     -hx, 0.0, hz, -hx, 0.0, hz,    -hx, 0.0, -hz,
            -hx, hy, -hz,   hx, hy, -hz, hx, hy, -hz,    hx, hy, hz,
            hx, hy, hz,     -hx, hy, hz, -hx, hy, hz,    -hx, hy, -hz,
            -hx, 0.0, -hz,   -hx, hy, -hz, hx, 0.0, -hz,    hx, hy, -hz,
            hx, 0.0, hz,     hx, hy, hz, -hx, 0.0, hz,    -hx, hy, hz,
        ], dtype='f4')

        self.bb_vbo = self.ctx.buffer(bb_verts.tobytes())
        self.bb_vao = self.ctx.simple_vertex_array(self.shader_lines, self.bb_vbo, 'in_pos')

    # --- DEFINICIÓN DEL SHADER (Método Estático) ---
    # Definido aquí para mayor coherencia. DEMO.py lo llamará una vez.
    @staticmethod
    def get_shader(ctx):
        return ctx.program(
            vertex_shader='''
                #version 330
                in vec3 in_position;
                // in vec3 in_normal;
                in vec2 in_texcoord;
                in vec3 in_color;
                in vec3 in_smooth_normal; // Normal suave
                
                uniform mat4 m_proj;
                uniform mat4 m_view;
                uniform mat4 m_model;
                uniform float outline_width;
                
                out vec3 v_normal;
                out vec3 v_frag_pos;
                out vec2 v_uv;
                out vec3 v_color;
                
                void main() {
                    vec3 pos = in_position;
                    if (outline_width > 0.0) {
                        pos += in_smooth_normal * outline_width;
                    }
                    vec4 world_pos = m_model * vec4(pos, 1.0);
                    v_frag_pos = world_pos.xyz;
                    
                    mat3 normal_matrix = mat3(transpose(inverse(m_model)));
                    
                    // TRUCO: Usamos la normal SUAVE para la iluminación.
                    v_normal = normalize(normal_matrix * in_smooth_normal);
                    
                    v_uv = in_texcoord;
                    v_color = in_color;
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
                uniform float outline_width;
                
                out vec4 fragColor;
                
                void main() {
                    if (outline_width > 0.0) {
                        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
                        return;
                    }

                    // --- TOON SHADING ---
                    vec4 texColor = texture(u_texture, v_uv);
                    vec3 baseColor = texColor.rgb * v_color;

                    vec3 norm = normalize(v_normal);
                    vec3 light_dir = normalize(light_pos - v_frag_pos);
                    vec3 view_dir = normalize(view_pos - v_frag_pos);

                    float intensity = max(dot(norm, light_dir), 0.05);
                    intensity = pow(intensity, 0.7); // Curva suave

                    float light_level;
                    if      (intensity > 0.95) light_level = 1.0;
                    else if (intensity > 0.80) light_level = 0.9;
                    else if (intensity > 0.60) light_level = 0.8;
                    else if (intensity > 0.40) light_level = 0.7;
                    else if (intensity > 0.20) light_level = 0.6;
                    else if (intensity > 0.10) light_level = 0.5;
                    else                       light_level = 0.3;

                    vec3 halfwayDir = normalize(light_dir + view_dir);
                    float NdotH = max(dot(norm, halfwayDir), 0.0);
                    float spec = (pow(NdotH, 64.0) > 0.9) ? 0.3 : 0.0;

                    vec3 result = baseColor * light_level + vec3(spec);
                    result = pow(result, vec3(1.0 / 2.2));

                    fragColor = vec4(result, 1.0);
                }
            '''
        )

    # ... (Resto de métodos de lógica de movimiento igual que antes) ...
    
    def infectar(self, distance):
        self.ring = Ring(self.ctx, self.camera, radius=distance, thickness=0.15, height=0.1, position=self.position, altura=self.position.y)
        puff_position = self.position + glm.vec3(0, 1.0, 0)
        self.puff.create_puff(puff_position, num_particles=12)

    def camino(self, destino, duracion_paso=1.0):
        pos_actual = glm.vec3(self.m_model[3].x, self.m_model[3].y, self.m_model[3].z)
        destino = glm.vec3(destino)
        direccion = destino - pos_actual
        distancia_total = glm.length(direccion)
        if distancia_total == 0: self.indice_paso = 0; self.en_movimiento = False; return
        dir_norm = glm.normalize(direccion)
        max_paso = self.height / 2.0
        n_pasos = int(distancia_total // max_paso)
        resto = distancia_total % max_paso
        self.pasos = []
        for i in range(1, n_pasos + 1): self.pasos.append(pos_actual + dir_norm * (i * max_paso))
        if resto > 0: self.pasos.append(pos_actual + dir_norm * (n_pasos * max_paso + resto))
        self.indice_paso = 0
        self.paso(self.pasos[0], duracion=duracion_paso)

    def paso(self, destino, duracion=1.0):
        self.en_movimiento = True
        self._move_elapsed = 0.0
        self.duracion = max(float(duracion), 1e-6)  # segundos del mundo (real-equivalentes)

        self.pos_inicial = glm.vec3(self.m_model[3].x, self.m_model[3].y, self.m_model[3].z)
        self.pos_final = glm.vec3(destino)

    def plan_preclass_departure(self, calendar, next_room):
        # calendar.t_day es segundos desde 08:00 dentro del día
        slot_sim_sec = calendar.slot_sim_seconds

        u_real_min = random.uniform(0.0, 30.0)
        delay_sim = (u_real_min / 30.0) * slot_sim_sec

        self.preclass_plan = {
            "target_room": next_room,
            "go_time": calendar.t_day + delay_sim,
            "weekday": calendar.weekday_key(),
            "slot": calendar.current_slot(),
        }

    def maybe_execute_preclass_plan(self, calendar):
        if not self.preclass_plan:
            return False

        # invalidar si cambió de día/slot (por seguridad)
        if self.preclass_plan["weekday"] != calendar.weekday_key():
            self.preclass_plan = None
            return False

        if calendar.t_day >= self.preclass_plan["go_time"]:
            room = self.preclass_plan["target_room"]
            self.preclass_plan = None

            # Aquí disparas tu lógica existente para ir a un destino:
            # por ejemplo, ajustar schedule/objetivo
            self.schedule = [room]
            return True

        return False

    def actualizar_movimiento(self, delta_time):
        if not self.en_movimiento:
            return

        self._move_elapsed += delta_time
        t = self._move_elapsed / self.duracion
        if t >= 1.0:
            t = 1.0
            self.en_movimiento = False

        pos = glm.mix(self.pos_inicial, self.pos_final, t)
        pos.y += 4 * 0.2 * t * (1 - t)
        self.m_model = glm.translate(glm.mat4(), pos)
        self.position = pos

        if not self.en_movimiento:
            if self.pasos and self.indice_paso + 1 < len(self.pasos):
                self.indice_paso += 1
                self.paso(self.pasos[self.indice_paso], self.speed)
            else:
                if hasattr(self, "camino_actual") and self.indice_camino >= len(self.camino_actual):
                    self._terminar_camino()

    def _definir_camino_hacia(self, destino):
        if not self.sala:
            self.camino_actual = [666]
            self.indice_camino = 0; self.destino_sala = "pasillo"
        else:
            sala_origen = self.mundo[self.sala]; sala_destino = self.mundo[destino]
            id_actual = self._waypoint_mas_cercano(sala_origen)
            if isinstance(sala_origen, Clase):
                if id_actual == sala_origen.salida_id: self.camino_actual = [sala_origen.salida_id, sala_origen.entrada_id]
                else: self.camino_actual = sala_origen.get_path(id_actual, sala_origen.salida_id[random.randint(0,len(sala_destino.salida_id)-1)])
                self.indice_camino = 0; self.destino_sala = "pasillo"; return
            elif isinstance(sala_origen, Pasillo):
                self.camino_actual = sala_origen.get_path(id_actual, sala_destino.entrada_id[random.randint(0,len(sala_destino.entrada_id)-1)])
                self.indice_camino = 0; self.destino_sala = destino           
    
    def _waypoint_mas_cercano(self, sala):
        min_dist = float("inf"); id_mas_cercano = None
        for wid, wp in sala.waypoints.items():
            d = glm.distance(wp.position, self.position)
            if d < min_dist: min_dist = d; id_mas_cercano = wid
        return id_mas_cercano

    def _terminar_camino(self):
        if self.destino_sala:
            self._cambiar_sala(self.destino_sala)
            self.destino_sala = None; self.camino_actual = []; self.indice_camino = 0
            if isinstance(self.mundo[self.sala], Clase): self.sentado = False
        else: 
            self.camino_actual = []
            self.sentado = True
    
    def _cambiar_sala(self, destino):
        if self.sala == destino: return
        if self.sala in self.mundo: self.mundo[self.sala].salir(self)
        if destino in self.mundo: self.mundo[destino].entrar(self)
        self.sala = destino

    def update(self, delta_time):
        self.puff.update(delta_time)
        if self.ring: self.ring.update(self.position) 
        self.actualizar_movimiento(delta_time)
        if self.en_movimiento: return
        if not self.schedule: return
        sala_objetivo = self.schedule[0]; sala_actual = self.sala
        if sala_actual == sala_objetivo:
            if not self.sentado:
                if not self.camino_actual:
                    sala = self.mundo[sala_actual]
                    self.objetivo_asiento = sala.asiento_libre_aleatorio()
                    if self.objetivo_asiento is not None:
                        id_actual = self._waypoint_mas_cercano(self.mundo[sala_actual])
                        self.camino_actual = sala.get_path(id_actual, self.objetivo_asiento)
                        self.indice_camino = 0
                    else: return
                if self.indice_camino < len(self.camino_actual):
                    next_id = self.camino_actual[self.indice_camino]
                    destino_wp = self.mundo[sala_actual].get_wp(next_id)
                    self.camino(destino_wp.position, duracion_paso=self.speed)
                    self.indice_camino += 1
                else: self.camino_actual = []
            return
        else:
            if not self.camino_actual: self._definir_camino_hacia(sala_objetivo)
            if self.camino_actual and self.indice_camino < len(self.camino_actual):
                if self.sala:
                    next_id = self.camino_actual[self.indice_camino]
                    destino_wp = self.mundo[self.sala].get_wp(next_id)
                    self.camino(destino_wp.position, duracion_paso=self.speed)
                    self.indice_camino += 1
                else:
                    next_id = self.camino_actual[self.indice_camino]
                    destino_wp = self.mundo["pasillo"].get_wp(next_id)
                    self.camino(destino_wp.position, duracion_paso=self.speed)
                    self.indice_camino += 1

    def get_model_matrix(self):
        m_model = glm.mat4(1.0)
        m_model = glm.translate(m_model, self.position)
        m_model = glm.rotate(m_model, self.rotation_angle, glm.vec3(0, 1, 0))
        m_model = glm.scale(m_model, glm.vec3((1, self.height / 1.71, 1)))
        return m_model
    
    def destroy(self):
        self.vbo.release(); self.bb_vbo.release(); self.shader_lines.release(); self.bb_vao.release()

    def render(self, shader, vao_tri, vao_line, light_pos, person_texture=None):
        self.puff.render()

        m_model = self.get_model_matrix()
        shader['m_model'].write(m_model)
        shader['light_pos'].value = light_pos
        shader['view_pos'].value = tuple(self.camera.position)
        shader['m_proj'].write(self.camera.m_proj)
        shader['m_view'].write(self.camera.m_view)

        # Configurar Textura
        if person_texture:
            person_texture.use(location=0)
            shader['u_texture'].value = 0
        elif hasattr(self.motor, 'object') and self.motor.object.texture:
             self.motor.object.texture.use(location=0)
             shader['u_texture'].value = 0

        self.ctx.enable(mgl.CULL_FACE)

        # Pasada 1: Contorno
        self.ctx.cull_face = 'front'
        if 'outline_width' in shader:
            shader['outline_width'].value = 0.01
        vao_tri.render(mode=mgl.TRIANGLES)

        # Pasada 2: Persona (Toon)
        self.ctx.cull_face = 'back'
        if 'outline_width' in shader:
            shader['outline_width'].value = 0.0
        vao_tri.render(mode=mgl.TRIANGLES)

        if self.ring: self.ring.render(light_pos)

        # Debug BBOX
        try: show_bb = bool(self.motor.show_bboxes)
        except: show_bb = False
        if show_bb:
            self.shader_lines['m_model'].write(glm.translate(glm.mat4(1.0), self.position))
            self.shader_lines['m_view'].write(self.camera.m_view)
            self.shader_lines['m_proj'].write(self.camera.m_proj)
            self.shader_lines['u_color'].value = (0.0, 1.0, 0.0)
            self.ctx.line_width = 2.0
            self.bb_vao.render(mode=mgl.LINES)