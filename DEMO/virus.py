import pygame as pg
import moderngl as mgl
import numpy as np
import glm
import sys
import time
import math
import random
from ring import Ring
from person import Person

class Virus:
    def __init__(self,app,td,tt,ip,r, infection_distance,evolve = 0):
        #self.puff_system = app.puff_system
        self.tick_duration = td
        self.tick_timer = tt 
        self.infection_probability = ip
        self.radio = r
        self.infection_distance = infection_distance
        self.evolve = evolve
        self.rastros = []
    
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
        puff_position = person.position + glm.vec3(0, 1.0, 0)
        #self.puff_system.create_puff(puff_position, num_particles=12)

    
    def check_infections(self,mundo):
        """Comprova col·lisions per transferir infecció."""

        for rastro in self.rastros:
            check = rastro.evolve()
            if check == -1:
                self.rastros.remove(rastro)
                rastro.destroy()

        infected_people = []
        uninfected_people = []
        for nombre in mundo:
            for p in mundo[nombre].personas:
                if p.ring:
                    infected_people.append(p)
                else:
                    uninfected_people.append(p)
        
        for infected in infected_people:
            infection_radius = infected.ring.contagion_radius
            nuevo = Rastro(infection_radius,infected,
                           self.infection_probability,
                           evolution_rate = 10)
            self.rastros.append(nuevo)

        if not uninfected_people:
            return
        

        for infected in infected_people:
            infection_radius = infected.ring.contagion_radius
            for uninfected in uninfected_people:
                
                dist = glm.length(infected.position - uninfected.position)
                
                if dist < infection_radius:
                    if random.random() < self.infection_probability:
                        self.infectar(uninfected)
                        uninfected_people.remove(uninfected)

        for rastro in self.rastros:
            for uninfected in uninfected_people:

                dist = glm.length(rastro.position - uninfected.position)
                
                if dist < rastro.radius:
                    if random.random() < rastro.infection_rate:
                        self.infectar(uninfected)
                        uninfected_people.remove(uninfected)
        

    def render(self,light_pos):
        for rastro in self.rastros:
            a = rastro.render(light_pos)

class Rastro:
    def __init__(self,rad,persona: Person ,infection_rate : float,evolution_rate : int):
        self.radius = rad
        self.infection_rate = infection_rate
        self.position = persona.position
        self.evolution = [self.radius-(self.radius/evolution_rate)*i for i in range(evolution_rate+1)]
        self.ring = Ring(persona.ctx, persona.camera,
                         radius=self.radius, thickness=0.15, height=0.1,
                         position=persona.position,
                         altura=persona.ground_y)

    def evolve(self):
        self.evolution.pop(0)
        self.radius = self.evolution[0]
        if self.radius == 0:
            self.destroy()
            return -1
        return 0
    
    def destroy(self):
        self.ring.destroy()

    def render(self, light_pos):
        self.ring.render(light_pos)

    def update(self):
        self.evolve()
        self.ring.update()
