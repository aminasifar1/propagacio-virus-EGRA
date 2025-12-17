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

        self.app = app

        # Debug grid: radio en celdas (puedes ajustar 6–12)
        self.debug_grid = False
        self.debug_grid_radius_cells = 8

        # Snapshot global (unión de todas las salas en el último tick)
        self._debug_grid_snapshot = {"cell_size": None, "occupied": set(), "cell_y": {}}

        # GL resources para dibujar el grid
        self._grid_prog = None
        self._grid_vbo = None
        self._grid_vao = None

        # STEP 2: trail por distancia (no por tick)
        self.trail_spacing = 0.35  # distancia entre marcas del rastro (ajusta: 0.30 más denso, 0.60 más ligero)
        self.trail_max_rastros_per_tick_per_person = 3  # evita picos si alguien teleporta/cambia sala

        # pid -> last_position (vec3)
        self._trail_last_pos = {}

    def _spawn_trail_for_infected(self, infected: Person, sala_name: str):
        """STEP 2: genera rastros detrás del infectado según distancia recorrida.
        Mantiene el look (rastros densos), pero evita que se acumulen encima.
        """
        pid = id(infected)
        current = glm.vec3(infected.position)

        # inicialización
        if pid not in self._trail_last_pos:
            self._trail_last_pos[pid] = glm.vec3(current)
            return

        last = self._trail_last_pos[pid]
        delta = current - last
        dist = glm.length(delta)

        if dist < self.trail_spacing:
            return

        # si se ha movido lo suficiente, ponemos UNA marca en la posición actual
        infection_radius = infected.ring.contagion_radius if infected.ring else self.infection_distance
        nuevo = Rastro(
            infection_radius,
            infected,
            self.infection_probability,
            evolution_rate=self.evolve,
            tick_duration=self.tick_duration,
            infection_distance=self.infection_distance,
            color=getattr(infected.ring, 'color', None),
            position_override=current,      # SOLO posición actual
            sala_name=sala_name,
        )
        self.rastros.append(nuevo)

        # actualizamos el último punto
        self._trail_last_pos[pid] = glm.vec3(current)

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

        # --- DEBUG GRID: acumuladores globales ---
        debug_occupied_all = set() if self.debug_grid else None
        debug_cell_y_all = {} if self.debug_grid else None
        debug_cell_size = max(0.5, float(self.infection_distance))

        # for rastro in self.rastros:
        #     check = rastro.evolve()
        #     if check == -1:
        #         self.rastros.remove(rastro)
        #         rastro.destroy()

        new_rastros = []
        for rastro in self.rastros:
            if rastro.evolve() == -1:
                rastro.destroy()
            else:
                new_rastros.append(rastro)
        self.rastros = new_rastros


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
                # infection_radius = infected.ring.contagion_radius
                # nuevo = Rastro(infection_radius, infected,
                #                 self.infection_probability,
                #                 evolution_rate=self.evolve,
                #                 tick_duration=self.tick_duration,
                #                 infection_distance=self.infection_distance,
                #                 color=getattr(infected.ring, 'color', None))
                # self.rastros.append(nuevo)
                self._spawn_trail_for_infected(infected, sala_name=nombre)

            if not uninfected_people:
                continue

            # for infected in infected_people:
            #     infection_radius = infected.ring.contagion_radius
            #     for uninfected in uninfected_people[:]: # iterar sobre una copia de la lista para poder modificarla mientras se itera
                    
            #         dist = glm.length(infected.position - uninfected.position)
                    
            #         if dist < infection_radius:
            #             if random.random() < self.infection_probability:
            #                 self.infectar(uninfected)
            #                 uninfected_people.remove(uninfected)

            cell_size = max(0.5, float(self.infection_distance))
            grid = SpatialHashGrid2D(cell_size)
            grid.build(mundo[nombre].personas)

            # --- DEBUG: snapshot del grid para render ---
            if self.debug_grid:
                debug_occupied_all.update(grid.cells.keys())
                for k, y in grid.cell_y.items():
                    debug_cell_y_all.setdefault(k, y)

            for infected in infected_people:
                infection_radius = infected.ring.contagion_radius
                for candidate in grid.neighbors(infected.position):
                    if candidate.ring:
                        continue
                    # solo infectamos si está en la lista local de no infectados
                    # (esto mantiene la lógica por sala)
                    if candidate not in uninfected_people:
                        continue

                    dist = glm.length(infected.position - candidate.position)
                    if dist < infection_radius and random.random() < self.infection_probability:
                        self.infectar(candidate)
                        if candidate in uninfected_people:
                            uninfected_people.remove(candidate)
            
            for uninfected in uninfected_people[:]:
                if random.random() < mundo[nombre].contagio_aire:
                    self.infectar(uninfected)
                    uninfected_people.remove(uninfected)

            # for rastro in self.rastros:
            #     for uninfected in uninfected_people[:]:
            #         dist = glm.length(rastro.position - uninfected.position)
            #         if dist < rastro.radius:
            #             if random.random() < rastro.infection_rate:
            #                 self.infectar(uninfected)
            #                 uninfected_people.remove(uninfected)    

            # STEP 5: grid también para rastros
            # (filtramos rastros por sala para no chequear contra todos)
            rastros_sala = [r for r in self.rastros if getattr(r, "sala_name", None) == nombre or getattr(r, "sala_name", None) is None]

            for rastro in rastros_sala:
                for candidate in grid.neighbors(rastro.position):
                    if candidate.ring:
                        continue
                    if candidate not in uninfected_people:
                        continue

                    dist = glm.length(rastro.position - candidate.position)
                    if dist < rastro.radius and random.random() < rastro.infection_rate:
                        self.infectar(candidate)
                        if candidate in uninfected_people:
                            uninfected_people.remove(candidate)

        # --- DEBUG GRID: guardar snapshot global ---
        if self.debug_grid:
            self._debug_grid_snapshot = {
                "cell_size": float(debug_cell_size),
                "occupied": debug_occupied_all,
                "cell_y": debug_cell_y_all,
            }

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

    def _ensure_grid_renderer(self):
        if self._grid_prog is not None:
            return

        GRID_VERT = """
        #version 330
        in vec3 in_pos;
        uniform mat4 mvp;
        void main() {
            gl_Position = mvp * vec4(in_pos, 1.0);
        }
        """
        GRID_FRAG = """
        #version 330
        uniform vec3 color;
        out vec4 fragColor;
        void main() {
            fragColor = vec4(color, 1.0);
        }
        """
        self._grid_prog = self.app.ctx.program(vertex_shader=GRID_VERT, fragment_shader=GRID_FRAG)
        self._grid_vbo = self.app.ctx.buffer(reserve=4)
        self._grid_vao = self.app.ctx.vertex_array(self._grid_prog, [(self._grid_vbo, "3f", "in_pos")])

    def _grid_segments_for_cells(self, cells, cs: float, cell_y: dict, y_fallback: float, y_offset: float = 0.02):
        segs = []
        for (ix, iz) in cells:
            x0 = ix * cs
            x1 = (ix + 1) * cs
            z0 = iz * cs
            z1 = (iz + 1) * cs
            y = float(cell_y.get((ix, iz), y_fallback)) + y_offset

            segs += [x0, y, z0,  x1, y, z0]
            segs += [x1, y, z0,  x1, y, z1]
            segs += [x1, y, z1,  x0, y, z1]
            segs += [x0, y, z1,  x0, y, z0]

        if not segs:
            return np.zeros((0, 3), dtype="f4")
        return np.array(segs, dtype="f4").reshape(-1, 3)

    def _draw_grid_segments(self, segs: np.ndarray, mvp: np.ndarray, color):
        if segs.size == 0:
            return
        self._grid_vbo.orphan(segs.nbytes)
        self._grid_vbo.write(segs.tobytes())
        self._grid_prog["mvp"].write(mvp.astype("f4").tobytes())
        self._grid_prog["color"].value = tuple(map(float, color))
        self._grid_vao.render(mode=mgl.LINES, vertices=len(segs))

    def render_debug_grid(self, mvp: np.ndarray,
                          color_all=(1.0, 1.0, 0.2),
                          color_occupied=(1.0, 0.2, 0.2)):
        if not self.debug_grid:
            return

        snap = self._debug_grid_snapshot
        cs = snap.get("cell_size", None)
        if not cs:
            return

        occupied = snap.get("occupied", set())
        cell_y = snap.get("cell_y", {})

        cam = self.app.camera.position
        ix0 = int(math.floor(cam.x / cs))
        iz0 = int(math.floor(cam.z / cs))
        R = int(self.debug_grid_radius_cells)

        # círculo de celdas alrededor de la cámara
        radius_cells = set()
        r2 = R * R
        for dx in range(-R, R + 1):
            for dz in range(-R, R + 1):
                if dx * dx + dz * dz <= r2:
                    radius_cells.add((ix0 + dx, iz0 + dz))

        # altura de referencia para celdas vacías: mediana de ocupadas cercanas
        occ_near = [cell_y[c] for c in (occupied & radius_cells) if c in cell_y]
        if occ_near:
            y_ref = float(np.median(np.array(occ_near, dtype="f4")))
        else:
            y_ref = 0.1

        self._ensure_grid_renderer()

        # Siempre visible
        self.app.ctx.disable(mgl.DEPTH_TEST)

        # 1) Todas en amarillo (radio)
        segs_all = self._grid_segments_for_cells(radius_cells, cs, cell_y, y_fallback=y_ref)
        self._draw_grid_segments(segs_all, mvp, color_all)

        # 2) Ocupadas en rojo (encima)
        segs_occ = self._grid_segments_for_cells(occupied & radius_cells, cs, cell_y, y_fallback=y_ref, y_offset=0.03)
        self._draw_grid_segments(segs_occ, mvp, color_occupied)

        self.app.ctx.enable(mgl.DEPTH_TEST)


class Rastro:
    def __init__(self, rad, persona: Person, infection_rate: float, evolution_rate: int, tick_duration: float = 0.2,
                 particles_per_step: int = 4, infection_distance: float = None, color=None, position_override=None, sala_name: str = None):
        self.O_radius = rad
        self.radius = rad
        self.infection_rate = infection_rate
        if position_override is not None:
            self.position = glm.vec3(position_override)
        else:
            self.position = glm.vec3(persona.position)
        self.expired = False
        self.sala_name = sala_name
        self.evolution = [1 - (1 / evolution_rate) * i for i in range(evolution_rate + 1)]
        self.tick_duration = tick_duration
        steps = max(1, len(self.evolution) - 1)
        self.particle_lifetime = max(0.6, self.tick_duration * steps * 1.2)
        self.particles_per_step = particles_per_step

        # Infection visual parameters
        self.infection_distance = infection_distance if infection_distance is not None else rad
        self.color = color if color is not None else (1.0, 0.5, 0.0)

        # Use Particles generator from ring.py
        self.particles = Particles(persona.ctx, persona.camera, min_alpha=0.0)
        # Use Particles generator from ring.py (solid mode for clearer contagion visualization)
        #self.particles = Particles(persona.ctx, persona.camera, default_solid=True)
        # Emit initial burst using infection_distance and ring color
        self.particles.emit(self.position, num=self.particles_per_step, color=self.color, radius=self.infection_distance, lifetime=self.particle_lifetime, fade_profile="late_fade")
        # Using solid mode
        #self.particles.emit(self.position, num=self.particles_per_step, color=self.color, radius=self.infection_distance, solid=True)

        # Limit concurrent particles per rastro to reduce load
        self.max_particles = max(4, self.particles_per_step * 3)

    def evolve(self):
        if not self.expired:
            # emit particles at current position (bounded by max_particles)
            # to_emit = min(self.particles_per_step, max(0, self.max_particles - len(self.particles.particles)))
            # if to_emit > 0:
            #     self.particles.emit(self.position, num=to_emit, color=self.color, radius=self.infection_distance, lifetime=self.particle_lifetime, fade_profile="late_fade")

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
                self.expired = True
                self.infection_rate = 0.0
                return 0
        
        if len(self.particles.particles) == 0:
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


# STEP 5: grid espacial 2D (XZ) para acelerar búsquedas
class SpatialHashGrid2D:
    def __init__(self, cell_size: float):
        self.cell_size = max(1e-6, float(cell_size))
        self.cells = {}  # (ix, iz) -> [Person]
        self.cell_y = {}  # (ix, iz) -> y aproximada

    def _cell(self, pos: glm.vec3):
        return (int(math.floor(pos.x / self.cell_size)), int(math.floor(pos.z / self.cell_size)))

    def build(self, people):
        self.cells.clear()
        self.cell_y.clear()
        for p in people:
            key = self._cell(p.position)
            self.cell_y.setdefault(key, float(p.position.y))
            self.cells.setdefault(key, []).append(p)

    def neighbors(self, pos: glm.vec3):
        ix, iz = self._cell(pos)
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                for p in self.cells.get((ix + dx, iz + dz), []):
                    yield p