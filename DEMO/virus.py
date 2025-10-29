import pygame as pg
import moderngl as mgl
import numpy as np
import glm
import sys
import time
import math
import random
from ring import Ring

class Virus:
    def __init__(self,app,td,tt,ip,r):
        #self.puff_system = app.puff_system
        self.tick_duration = td
        self.tick_timer = tt 
        self.infection_probability = ip
        self.radio = r
    
    def update(self,td,tt,ip,r):
        self.tick_duration = td
        self.tick_timer = tt 
        self.infection_probability = ip
        self.radio = r

    def infectar(self,person):
        """Infecta una persona i crea l'efecte puff."""
        if not person.ring:
            person.ring = Ring(
                person.ctx, person.camera,
                radius=0.9, thickness=0.15, height=0.1,
                position=person.position + glm.vec3(0, 0.05, 0)  # Offset Y per l'anell
            )
        print("Una persona s'ha infectat!")
        puff_position = person.position + glm.vec3(0, 1.0, 0)
        #self.puff_system.create_puff(puff_position, num_particles=12)

    
    def check_infections(self,mundo):
        """Comprova col·lisions per transferir infecció."""

        infected_people = []
        uninfected_people = []
        for nombre in mundo:
            for p in mundo[nombre].personas:
                if p.ring:
                    infected_people.append(p)
                else:
                    uninfected_people.append(p)
        
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
