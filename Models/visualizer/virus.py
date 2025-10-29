import pygame as pg
import moderngl as mgl
import numpy as np
import glm
import sys
import time
import math
import random

class Virus:
    def __init__(self,app,td,tt,ip,r):
        self.puff_system = app.puff_system
        self.tick_duration = td
        self.tick_timer = tt 
        self.infection_probability = ip
        self.radio = r
    
    def update(self,td,tt,ip,r):
        self.tick_duration = td
        self.tick_timer = tt 
        self.infection_probability = ip
        self.radio = r
    
    def check_infections(self,people):
        """Comprova col·lisions per transferir infecció."""
        if not people:
            return

        infected_people = []
        uninfected_people = []
        for p in people:
            if p.ring:
                infected_people.append(p)
            else:
                uninfected_people.append(p)
        
        if not uninfected_people:
            return

        newly_infected = []

        for infected in infected_people:
            infection_radius = infected.ring.contagion_radius
            for uninfected in uninfected_people:
                if uninfected in newly_infected:
                    continue
                
                dist = glm.length(infected.position - uninfected.position)
                
                if dist < infection_radius:
                    if random.random() < self.infection_probability:
                        newly_infected.append(uninfected)
        
        # Infectar als nous i crear efecte puff
        for person_to_infect in newly_infected:
            person_to_infect.infect()
            # CREAR EFECTO PUFF EN LA POSICIÓN DE LA PERSONA
            puff_position = person_to_infect.position + glm.vec3(0, 1.0, 0)
            self.puff_system.create_puff(puff_position, num_particles=12)