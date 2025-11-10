# ================================
# classgrid.py (versió amb diagonals)
# ================================
import glm
import numpy as np
import moderngl as mgl
from pathfinding import PathfindingSystem
from waypoint import WaypointVisualizer

class ClassGrid:
    """Graella completa amb línies de cada cel·la i diagonals."""

    def __init__(self, ctx, camera, min_corner, max_corner, spacing=1.6, color=(0.0, 1.0, 0.0)):
        self.ctx = ctx
        self.camera = camera
        self.min_corner = min_corner
        self.max_corner = max_corner
        self.spacing = spacing
        self.color = color

        self.pathfinding = PathfindingSystem()
        self.visualizer = WaypointVisualizer(ctx, camera)

        self.grid_lines = []
        self.vbo = None
        self.vao = None

        self._generate_grid()
        self._build_lines_with_diagonals()
        self.visualizer.build_from_system(self.pathfinding)

    # --------------------------------------------------------
    def _generate_grid(self):
        """Crea els waypoints i connexions bàsiques."""
        ground_y = self.min_corner.y
        x_vals = np.arange(self.min_corner.x, self.max_corner.x + 1e-3, self.spacing)
        z_vals = np.arange(self.min_corner.z, self.max_corner.z + 1e-3, self.spacing)

        self.waypoints_grid = {}
        for x in x_vals:
            for z in z_vals:
                pos = glm.vec3(x, ground_y, z)
                wp = self.pathfinding.add_waypoint(pos)
                self.waypoints_grid[(x, z)] = wp

        # Connexions bàsiques en 4 direccions
        for x, z in self.waypoints_grid.keys():
            wp = self.waypoints_grid[(x, z)]
            for dx, dz in [(self.spacing, 0), (0, self.spacing)]:
                neighbor = self.waypoints_grid.get((x + dx, z + dz))
                if neighbor:
                    self.pathfinding.connect(wp, neighbor)

    # --------------------------------------------------------
    def _build_lines_with_diagonals(self):
        """Genera totes les línies horitzontals, verticals i diagonals."""
        ground_y = self.min_corner.y
        x_vals = np.arange(self.min_corner.x, self.max_corner.x + 1e-3, self.spacing)
        z_vals = np.arange(self.min_corner.z, self.max_corner.z + 1e-3, self.spacing)

        for x in x_vals:
            for z in z_vals:
                # Vèrtexs del subquadrat
                p0 = glm.vec3(x, ground_y, z)
                p1 = glm.vec3(x + self.spacing, ground_y, z)
                p2 = glm.vec3(x, ground_y, z + self.spacing)
                p3 = glm.vec3(x + self.spacing, ground_y, z + self.spacing)

                # Comprovem límits abans d’afegir línies
                if x + self.spacing <= self.max_corner.x:
                    self.grid_lines.append((p0, p1))
                if z + self.spacing <= self.max_corner.z:
                    self.grid_lines.append((p0, p2))
                if x + self.spacing <= self.max_corner.x and z + self.spacing <= self.max_corner.z:
                    # Línies oposades per tancar la cel·la
                    self.grid_lines.append((p1, p3))
                    self.grid_lines.append((p2, p3))
                    # Diagonals internes (X)
                    self.grid_lines.append((p0, p3))
                    self.grid_lines.append((p1, p2))

        # Convertim les línies a VBO
        vertices = []
        for p0, p1 in self.grid_lines:
            vertices.extend(p0)
            vertices.extend(p1)

        self.vbo = self.ctx.buffer(np.array(vertices, dtype='f4'))
        shader = self._get_shader()
        self.vao = self.ctx.vertex_array(shader, [(self.vbo, '3f', 'in_position')])
        self.vertex_count = len(vertices) // 3

    # --------------------------------------------------------
    def _get_shader(self):
        return self.ctx.program(
            vertex_shader='''
                #version 330
                in vec3 in_position;
                uniform mat4 m_proj;
                uniform mat4 m_view;
                uniform mat4 m_model;
                void main() {
                    gl_Position = m_proj * m_view * m_model * vec4(in_position,1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                uniform vec3 grid_color;
                out vec4 fragColor;
                void main() {
                    fragColor = vec4(grid_color,1.0);
                }
            '''
        )

    # --------------------------------------------------------
    def render(self):
        """Dibuixa la graella completa amb diagonals."""
        if self.vao:
            m_model = glm.mat4(1.0)
            self.vao.program['m_proj'].write(self.camera.m_proj)
            self.vao.program['m_view'].write(self.camera.m_view)
            self.vao.program['m_model'].write(m_model)
            self.vao.program['grid_color'].value = self.color
            self.ctx.line_width = 1.8
            self.vao.render(mode=mgl.LINES)

        # També dibuixa els waypoints connectats (opcional)
        self.visualizer.render()
