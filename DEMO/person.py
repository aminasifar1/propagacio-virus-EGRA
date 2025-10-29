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


class Person:
    def __init__(self, ctx, camera, tri_data, normals, line_data, facultad, schedule=None, sala='pasillo', ground_y=0.1, is_infected=False):
        """
        Crea una persona que camina siguiendo waypoints.
        ground_y: Altura del suelo donde deben caminar las personas
        is_infected: Si la persona comença amb el 'ring'
        """
        self.ctx = ctx
        self.camera = camera
        self.tri_vbo = self.ctx.buffer(tri_data)
        self.nrm_vbo = self.ctx.buffer(normals)
        self.line_vbo = self.ctx.buffer(line_data)
        self.mundo = facultad
        self.schedule = schedule
        self.present = True
        self.sala = None

        # Personalizacion
        self.height = min(np.random.normal(1.777, 0.1), 1.95)
        self.speed = np.random.normal(0.5, 0.05)

        # Starting position and destiny
        self._cambiar_sala(sala)
        self.position = self.mundo[self.sala].get_wp(self.mundo[self.sala].salida_id).position
        self.ground_y = ground_y
        self.position.y = self.ground_y
        self.m_model = glm.translate(glm.mat4(1.0), self.position)

        self.rotation_angle = 0.0

        # Estado del movimiento y navegación
        self.en_movimiento = False       # si está realizando un paso animado
        self.pasos = []                  # lista de sub-pasos (para el salto)
        self.indice_paso = 0
        self.camino_actual = []          # lista de IDs de waypoints a seguir
        self.indice_camino = 0
        self.sentado = False             # si ya está sentado
        self.objetivo_asiento = None     # id del asiento hacia el que se dirige
        self.destino_sala = None         # sala a la que se está moviendo

        # Control de infeccion
        self.is_infected = is_infected

    # =========================================================
    # ANIMACIÓN DE MOVIMIENTO (ya existente en tu código)
    # =========================================================
    def camino(self, destino, duracion_paso=1.0):
        """
        Divide el recorrido hacia 'destino' en pasos de longitud máxima = altura/2.
        Inicia el primer paso automáticamente.
        """
        pos_actual = glm.vec3(self.m_model[3].x, self.m_model[3].y, self.m_model[3].z)
        destino = glm.vec3(destino)
        direccion = destino - pos_actual
        distancia_total = glm.length(direccion)

        if distancia_total == 0:
            return

        # Vector de dirección normalizado
        dir_norm = glm.normalize(direccion)

        # Longitud máxima por paso
        max_paso = self.height / 2.0

        # Número entero de pasos
        n_pasos = int(distancia_total // max_paso)
        resto = distancia_total % max_paso

        # Generar destinos intermedios
        self.pasos = []
        for i in range(1, n_pasos + 1):
            self.pasos.append(pos_actual + dir_norm * (i * max_paso))
        if resto > 0:
            self.pasos.append(pos_actual + dir_norm * (n_pasos * max_paso + resto))

        # Reiniciar contador y empezar el primero
        self.indice_paso = 0
        self.paso(self.pasos[0], duracion=duracion_paso)

    def paso(self, destino, duracion=1.0):
        """Inicia un movimiento con trayectoria de arco desde la posición actual hacia 'destino'."""
        self.en_movimiento = True
        self.t_inicio = pg.time.get_ticks()
        self.duracion = duracion * 1000  # a milisegundos
        self.pos_inicial = glm.vec3(self.m_model[3].x, self.m_model[3].y, self.m_model[3].z)
        self.pos_final = glm.vec3(destino)

    def actualizar_movimiento(self):
        """Actualiza la posición si el objeto está en movimiento."""
        if not self.en_movimiento:
            return

        tiempo_actual = pg.time.get_ticks()
        t = (tiempo_actual - self.t_inicio) / self.duracion
        if t >= 1.0:
            t = 1.0
            self.en_movimiento = False

        # Interpolación lineal
        pos = glm.mix(self.pos_inicial, self.pos_final, t)

        # Altura parabólica para simular arco
        altura = 0.2
        pos.y += 4 * altura * t * (1 - t)

        # Actualizar matriz de modelo y posición
        self.m_model = glm.translate(glm.mat4(), pos)
        self.position = pos  # mantener sincronizado

        # Si termina el paso, iniciar el siguiente (si hay más)
        if not self.en_movimiento:
            if self.pasos and self.indice_paso + 1 < len(self.pasos):
                self.indice_paso += 1
                self.paso(self.pasos[self.indice_paso], self.speed)
            else:
                # Ha terminado todos los pasos → fin del camino actual
                if hasattr(self, "camino_actual") and self.indice_camino >= len(self.camino_actual):
                    self._terminar_camino()      

    # =========================================================
    # LÓGICA DE NAVEGACIÓN Y HORARIO
    # =========================================================
    def _definir_camino_hacia(self, destino):
        """Calcula el camino desde la sala actual hasta la sala de destino."""
        sala_origen = self.mundo[self.sala]
        sala_destino = self.mundo[destino]

        # 1. Si está en una clase → ir a la salida
        if isinstance(sala_origen, Clase):
            self.camino_actual = sala_origen.get_path(sala_origen.salida_id, sala_origen.entrada_id)
            self.indice_camino = 0
            self.destino_sala = "pasillo"
            return

        # 2. Si está en un pasillo → ir hacia la entrada del destino
        if isinstance(sala_origen, Pasillo):
            entrada_destino = sala_destino.entrada_id
            self.camino_actual = sala_origen.get_path(sala_origen.salida_id, entrada_destino)
            self.indice_camino = 0
            self.destino_sala = destino

    def _terminar_camino(self):
        """Acciones al terminar un camino."""
        if self.destino_sala:
            self._cambiar_sala(self.destino_sala)
            self.destino_sala = None
            self.camino_actual = []
            self.indice_camino = 0
            print(f"[PERSONA] Ahora está en {self.sala}")

            # Si acaba de entrar en una clase, activar búsqueda de asiento
            if isinstance(self.mundo[self.sala], Clase):
                self.sentado = False
        else:
            # Movimiento interno (por ejemplo hacia un asiento)
            # self.sentado = True
            self.camino_actual = []
            print(f"[PERSONA] Sentada en {self.sala}.")
    
    def _cambiar_sala(self, destino):
        """Cambia la persona de una sala a otra actualizando ambas listas."""
        if self.sala == destino:
            return  # ya está en esa sala

        # Sacarla de la sala actual (si existe)
        if self.sala in self.mundo:
            self.mundo[self.sala].salir(self)

        # Entrarla en la nueva sala
        if destino in self.mundo:
            self.mundo[destino].entrar(self)

        # Actualizar el atributo local
        self.sala = destino

        print(f"[PERSONA] Ahora está en la sala {self.sala}")        

    # =========================================================
    # UPDATE Y RENDER
    # =========================================================
    def update(self, delta_time):
        """
        Controla la lógica general de la persona:
        - Actualiza animación de pasos.
        - Decide nuevos destinos según el horario.
        """
        # Actualizar animación si está en movimiento
        self.actualizar_movimiento()
        if self.en_movimiento:
            return

        # Si no hay horario, no hace nada
        if not self.schedule:
            return

        sala_objetivo = self.schedule[0]  # en la demo solo hay una
        sala_actual = self.sala

        # === 1. Si ya está en la sala correcta ===
        if sala_actual == sala_objetivo:
            if not self.sentado:
                # Si no hay un camino actual, buscar un asiento libre
                if not self.camino_actual:
                    sala = self.mundo[sala_actual]
                    self.objetivo_asiento = sala.asiento_libre_aleatorio()
                    if self.objetivo_asiento is not None:
                        # sala.ocupar_asiento(self.objetivo_asiento)
                        self.camino_actual = sala.get_path(sala.salida_id, self.objetivo_asiento)
                        self.indice_camino = 0
                    else:
                        print(f"[{sala_actual}] No hay asientos libres.")
                        return

                # Moverse al siguiente waypoint del camino
                if self.indice_camino < len(self.camino_actual):
                    next_id = self.camino_actual[self.indice_camino]
                    destino_wp = self.mundo[sala_actual].get_wp(next_id)
                    self.camino(destino_wp.position, duracion_paso=self.speed)
                    self.indice_camino += 1
                else:
                    # self.sentado = True
                    self.camino_actual = []
                    print(f"[{sala_actual}] Persona sentada.")
            return

        # === 2. Si está en otra sala ===
        else:
            if not self.camino_actual:
                self._definir_camino_hacia(sala_objetivo)

            if self.camino_actual and self.indice_camino < len(self.camino_actual):
                next_id = self.camino_actual[self.indice_camino]
                destino_wp = self.mundo[self.sala].get_wp(next_id)
                self.camino(destino_wp.position, duracion_paso=self.speed)
                self.indice_camino += 1

    def get_model_matrix(self):
        """Retorna la matriz de modelo para esta persona."""
        m_model = glm.mat4(1.0)
        m_model = glm.translate(m_model, self.position)
        m_model = glm.rotate(m_model, self.rotation_angle, glm.vec3(0, 1, 0))
        m_model = glm.scale(m_model, glm.vec3(self.height / 1.71))
        return m_model

    def render(self, shader, vao_tri, vao_line, light_pos):
        """Renderiza la persona i el seu anell si existeix."""

        # Renderitzar la persona (malla)
        m_model = self.get_model_matrix()
        shader['m_model'].write(m_model)
        shader['light_pos'].value = light_pos
        shader['view_pos'].value = tuple(self.camera.position)
        # Actualitzar vista/projecció per si la càmera s'ha mogut
        shader['m_proj'].write(self.camera.m_proj)
        shader['m_view'].write(self.camera.m_view)

        vao_tri.render(mode=mgl.TRIANGLES)
        self.ctx.line_width = 1.0
        vao_line.render(mode=mgl.LINES)
