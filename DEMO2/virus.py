import pygame as pg
import moderngl as mgl
import numpy as np
import glm
import sys
import time
import math
import random
from ring import Ring, Particles
from person import Person

import random
import glm

class Virus:
    """
    Comprova col·lisions per transferir infecció.
    """

    def __init__(self,app,td,tt,ip,r, infection_distance,aire = 0.00006,disipar = 0.00005,evolve = 10):
        #self.puff_system = app.puff_system
        self.tick_duration = td
        self.tick_timer = tt 
        self.infection_probability = ip
        self.radio = r
        self.infection_distance = infection_distance
        self.evolve = evolve
        self.rastros = []
        self.contagio_aire = aire
        self.disipar = disipar

    def update(self,td,tt,ip,r):
        self.tick_duration = td
        self.tick_timer = tt 
        self.infection_probability = ip
        self.radio = r

    def infectar(self,person):
        """Infecta una persona i crea l'efecte puff."""
        if not person.ring:
            person.infectar(self.infection_distance)
        print("Una persona s'ha infectat!")
        #puff_position = person.position + glm.vec3(0, 1.0, 0)
        #self.puff_system.create_puff(puff_position, num_particles=12)

    # -------------------------------------------------------
    # COMPROVACIÓ COL·LISIÓ RING - AABB
    # -------------------------------------------------------
    def ring_collides_aabb(self, ring, person):
        """
        Comprova si el bounding box (AABB) de la persona toca l'arc de la persona infectada.
        """

        # Coordenades del centre del ring
        cx, cy, cz = ring.position.x, ring.position.y, ring.position.z
        radius = ring.contagion_radius
        ring_bottom = ring.position.y
        ring_top = ring.position.y + ring.altura  # altura real del ring

        # Bounding box de la persona
        px, py, pz = person.position.x, person.position.y, person.position.z
        hx, hy, hz = person.bb_half.x, person.bb_half.y, person.bb_half.z

        # AABB Min / Max
        box_min = glm.vec3(px - hx, py, pz - hz)
        box_max = glm.vec3(px + hx, py + hy*2, pz + hz)

        # Distància mínima entre el cilindre i l'AABB projectada a XZ
        dx = max(box_min.x - cx, 0, cx - box_max.x)
        dz = max(box_min.z - cz, 0, cz - box_max.z)

        # Comprovació vertical
        dy = 0
        if ring_top < box_min.y:
            dy = box_min.y - ring_top
        elif ring_bottom > box_max.y:
            dy = ring_bottom - box_max.y

        # Col·lisió si distància horitzontal és <= radi i vertical està alineat
        return (dx*dx + dz*dz <= radius*radius) and (dy == 0)

    # -------------------------------------------------------
    # CHECK INFECTIONS
    def check_infections(self,mundo):
        """Comprova col·lisions per transferir infecció."""

        for rastro in self.rastros:
            check = rastro.evolve()
            if check == -1:
                self.rastros.remove(rastro)
                rastro.destroy()


        for nombre in mundo:
            # Disminuir el nivel de contagio por aire en la sala
            mundo[nombre].contagio_aire -= self.disipar
            if mundo[nombre].contagio_aire < 0.0:
                mundo[nombre].contagio_aire = 0.0
            
            infected_people = []
            uninfected_people = []
            for p in mundo[nombre].personas:
                if p.ring:
                    infected_people.append(p)
                else:
                    uninfected_people.append(p)

            # Incrementar el nivel de contagio por aire en la sala
            if len(infected_people) > 0:
                mundo[nombre].contagio_aire += self.contagio_aire * len(infected_people)
                if mundo[nombre].contagio_aire > 1.0:
                    mundo[nombre].contagio_aire = 1.0
        
            for infected in infected_people:
                infection_radius = infected.ring.contagion_radius
                nuevo = Rastro(infection_radius, infected,
                                self.infection_probability,
                                evolution_rate=self.evolve,
                                tick_duration=self.tick_duration,
                                infection_distance=self.infection_distance,
                                color=getattr(infected.ring, 'color', None))
                self.rastros.append(nuevo)
            if nombre == "pasillo":
                print("Prob contagio sala",nombre,":",mundo[nombre].contagio_aire)

            if not uninfected_people:
                continue

            for infected in infected_people:
                infection_radius = infected.ring.contagion_radius
                for uninfected in uninfected_people[:]: # iterar sobre una copia de la lista para poder modificarla mientras se itera
                    
                    dist = glm.length(infected.position - uninfected.position)
                    
                    if dist < infection_radius:
                        if random.random() < self.infection_probability:
                            self.infectar(uninfected)
                            uninfected_people.remove(uninfected)
            
            for uninfected in uninfected_people:
                if random.random() < mundo[nombre].contagio_aire:
                    self.infectar(uninfected)
                    uninfected_people.remove(uninfected)
            

        for rastro in self.rastros:
            for uninfected in uninfected_people:

                dist = glm.length(rastro.position - uninfected.position)
                
                if dist < rastro.radius:
                    if random.random() < rastro.infection_rate:
                        self.infectar(uninfected)
                        uninfected_people.remove(uninfected)    

    def update_particles(self, delta_time: float):
        """Actualiza el sistema de partículas de todos los rastros cada frame."""
        for rastro in self.rastros:
            try:
                rastro.particles.update(delta_time)
            except Exception:
                pass

    def render(self,light_pos):
        for rastro in self.rastros:
            try:
                rastro.render(light_pos)
            except Exception:
                # protect rendering loop from per-rastro errors
                pass


class Rastro:
    def __init__(self, rad, persona: Person, infection_rate: float, evolution_rate: int, tick_duration: float = 0.2, particles_per_step: int = 2, infection_distance: float = None, color=None):
        self.O_radius = rad
        self.radius = rad
        self.infection_rate = infection_rate
        self.position = persona.position
        self.evolution = [1 - (1 / evolution_rate) * i for i in range(evolution_rate + 1)]
        self.tick_duration = tick_duration
        self.particles_per_step = particles_per_step

        # Infection visual parameters
        self.infection_distance = infection_distance if infection_distance is not None else rad
        self.color = color if color is not None else (1.0, 0.5, 0.0)

        # Use Particles generator from ring.py
        self.particles = Particles(persona.ctx, persona.camera)
        # Use Particles generator from ring.py (solid mode for clearer contagion visualization)
        #self.particles = Particles(persona.ctx, persona.camera, default_solid=True)
        # Emit initial burst using infection_distance and ring color
        self.particles.emit(self.position, num=self.particles_per_step, color=self.color, radius=self.infection_distance)
        # Using solid mode
        #self.particles.emit(self.position, num=self.particles_per_step, color=self.color, radius=self.infection_distance, solid=True)

        # Limit concurrent particles per rastro to reduce load
        self.max_particles = max(4, self.particles_per_step * 3)

    def evolve(self):
        # emit particles at current position (bounded by max_particles)
        to_emit = min(self.particles_per_step, max(0, self.max_particles - len(self.particles.particles)))
        if to_emit > 0:
            self.particles.emit(self.position, num=to_emit, color=self.color, radius=self.infection_distance)

        # # advance particle system
        # try:
        #     self.particles.update(self.tick_duration)
        # except Exception:
        #     pass

        # evolve infection radius
        if len(self.evolution) > 1:
            self.evolution.pop(0)
            self.radius = self.O_radius * self.evolution[0]
        else:
            self.radius = 0

        if self.radius == 0:
            self.destroy()
            return -1
        return 0

    def destroy(self):
        # # release particle GL resources
        # for p in list(self.particles.particles):
        #     try:
        #         p.vbo.release(); p.shader.release(); p.vao.release()
        #     except Exception:
        #         pass
        # self.particles.particles.clear()

        self.particles.particles.clear()

    def render(self, light_pos=None):
        try:
            self.particles.render()
        except Exception:
            pass

    def update(self):
        self.evolve()