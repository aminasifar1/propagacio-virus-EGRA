import pygame as pg
import moderngl as mgl
import numpy as np
import glm
import sys
import time
import math
import random


class PathfindingSystem:
    """Sistema simple de navegación con waypoints."""

    def __init__(self):
        self.waypoints = []

    def add_waypoint(self, position):
        """Añade un waypoint."""
        wp = Waypoint(glm.vec3(position))
        self.waypoints.append(wp)
        return wp

    def connect(self, wp1, wp2):
        """Conecta dos waypoints bidireccionalemente."""
        if wp2 not in wp1.connections:
            wp1.connections.append(wp2)
        if wp1 not in wp2.connections:
            wp2.connections.append(wp1)

    def get_nearest_waypoint(self, position):
        """Encuentra el waypoint más cercano a una posición."""
        if not self.waypoints:
            return None
        min_dist = float('inf')
        nearest = None
        for wp in self.waypoints:
            dist = glm.length(wp.position - position)
            if dist < min_dist:
                min_dist = dist
                nearest = wp
        return nearest
