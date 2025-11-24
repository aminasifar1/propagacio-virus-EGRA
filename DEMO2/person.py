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
    def __init__(self, motor, ctx, camera, vertex_data, facultad, schedule=None, sala='pasillo', ground_y=0.1, position=None):
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

        # Personalizacion
        self.height = min(max(np.random.normal(1.75, 0.1), 1.6), 1.9)
        self.speed = np.random.normal(0.5, 0.05)

        # Starting position and destiny
        self._cambiar_sala(sala)
        if position:
            self.position = position
        else:
            self.position = self.mundo[self.sala].get_wp(self.mundo[self.sala].salida_id).position
        self.ground_y = ground_y
        self.position.y = self.ground_y
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

    # ... (Métodos infectar, camino, paso, actualizar_movimiento, _definir_camino..., _waypoint..., _terminar..., _cambiar... igual que antes) ...
    # REPETIMOS LOS MÉTODOS DE LÓGICA PARA QUE EL FICHERO SEA COMPLETO
    
    def infectar(self, distance):
        self.ring = Ring(self.ctx, self.camera, radius=distance, thickness=0.15, height=0.1, position=self.position, altura=self.ground_y)
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
        self.t_inicio = pg.time.get_ticks()
        self.duracion = duracion / self.motor.speed * 1000
        self.pos_inicial = glm.vec3(self.m_model[3].x, self.m_model[3].y, self.m_model[3].z)
        self.pos_final = glm.vec3(destino)

    def actualizar_movimiento(self):
        if not self.en_movimiento: return
        t = (pg.time.get_ticks() - self.t_inicio) / self.duracion
        if t >= 1.0: t = 1.0; self.en_movimiento = False
        pos = glm.mix(self.pos_inicial, self.pos_final, t)
        pos.y += 4 * 0.2 * t * (1 - t)
        self.m_model = glm.translate(glm.mat4(), pos)
        self.position = pos
        if not self.en_movimiento:
            if self.pasos and self.indice_paso + 1 < len(self.pasos):
                self.indice_paso += 1; self.paso(self.pasos[self.indice_paso], self.speed)
            else:
                if hasattr(self, "camino_actual") and self.indice_camino >= len(self.camino_actual): self._terminar_camino()

    def _definir_camino_hacia(self, destino):
        sala_origen = self.mundo[self.sala]; sala_destino = self.mundo[destino]
        id_actual = self._waypoint_mas_cercano(sala_origen)
        if isinstance(sala_origen, Clase):
            if id_actual == sala_origen.salida_id: self.camino_actual = [sala_origen.salida_id, sala_origen.entrada_id]
            else: self.camino_actual = sala_origen.get_path(id_actual, sala_origen.salida_id)
            self.indice_camino = 0; self.destino_sala = "pasillo"; return
        if isinstance(sala_origen, Pasillo):
            self.camino_actual = sala_origen.get_path(id_actual, sala_destino.entrada_id)
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
        else: self.camino_actual = []
    
    def _cambiar_sala(self, destino):
        if self.sala == destino: return
        if self.sala in self.mundo: self.mundo[self.sala].salir(self)
        if destino in self.mundo: self.mundo[destino].entrar(self)
        self.sala = destino

    def update(self, delta_time):
        self.puff.update(delta_time)
        if self.ring: self.ring.update(self.position) 
        self.actualizar_movimiento()
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
                next_id = self.camino_actual[self.indice_camino]
                destino_wp = self.mundo[self.sala].get_wp(next_id)
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

        # --- ACTIVAR ILUMINACIÓN SUAVE ---
        if 'use_smooth_lighting' in shader:
            shader['use_smooth_lighting'].value = 1.0

        self.ctx.enable(mgl.CULL_FACE)

        # Pasada 1: Contorno
        self.ctx.cull_face = 'front'
        if 'outline_width' in shader:
            shader['outline_width'].value = 0.015 
        vao_tri.render(mode=mgl.TRIANGLES)

        # Pasada 2: Persona (Toon)
        self.ctx.cull_face = 'back'
        if 'outline_width' in shader:
            shader['outline_width'].value = 0.0
        vao_tri.render(mode=mgl.TRIANGLES)

        # Limpieza (opcional, por seguridad)
        if 'use_smooth_lighting' in shader:
            shader['use_smooth_lighting'].value = 0.0

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