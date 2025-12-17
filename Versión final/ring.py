import pygame as pg
import moderngl as mgl
import numpy as np
import glm
import math
import random

class Ring:
    """Anell amb volum, Toon Shading i Outline (Cel Shading)."""

    def __init__(self, ctx, camera, radius=0.9, thickness=0.15, height=0.1, segments=24, color=(1.0, 0.2, 0.2),
                 position=glm.vec3(0, 0, 0), altura= 1):
        self.ctx = ctx
        self.camera = camera
        self.color = color
        self.position = position
        self.m_model = glm.translate(glm.mat4(), self.position)
        self.contagion_radius = radius + thickness
        self.altura = altura

        # --- Generació de la geometria 3D (Posició, Normal, Normal Suave) ---
        # Format del buffer: 3f (pos) + 3f (norm) + 3f (smooth_norm)
        vertex_data = []

        r_outer = radius + thickness / 2
        r_inner = radius - thickness / 2

        for i in range(segments):
            theta1 = (i / segments) * 2 * math.pi
            theta2 = ((i + 1) / segments) * 2 * math.pi

            c1, s1 = math.cos(theta1), math.sin(theta1)
            c2, s2 = math.cos(theta2), math.sin(theta2)

            # --- POSICIONES ---
            # 1 = actual, 2 = següent
            # ob = Outer-Bottom, ot = Outer-Top, ib = Inner-Bottom, it = Inner-Top
            
            # Segment actual (1)
            p_ob1 = (c1 * r_outer, 0, s1 * r_outer)
            p_ot1 = (c1 * r_outer, height, s1 * r_outer)
            p_ib1 = (c1 * r_inner, 0, s1 * r_inner)
            p_it1 = (c1 * r_inner, height, s1 * r_inner)

            # Segment següent (2)
            p_ob2 = (c2 * r_outer, 0, s2 * r_outer)
            p_ot2 = (c2 * r_outer, height, s2 * r_outer)
            p_ib2 = (c2 * r_inner, 0, s2 * r_inner)
            p_it2 = (c2 * r_inner, height, s2 * r_inner)

            # --- NORMALES PLANAS (Hard Shading) ---
            n_up   = (0, 1, 0)
            n_down = (0, -1, 0)
            n_out1 = (c1, 0, s1)
            n_out2 = (c2, 0, s2)
            n_in1  = (-c1, 0, -s1)
            n_in2  = (-c2, 0, -s2)

            # --- NORMALES SUAVES (Smooth Shading para Outline) ---
            # Promig de les cares adjacents a cada vèrtex per inflar en diagonal
            sn_ot1 = glm.normalize(glm.vec3(c1, 1, s1))   # Out + Up
            sn_ob1 = glm.normalize(glm.vec3(c1, -1, s1))  # Out + Down
            sn_it1 = glm.normalize(glm.vec3(-c1, 1, -s1)) # In + Up
            sn_ib1 = glm.normalize(glm.vec3(-c1, -1, -s1))# In + Down
            
            sn_ot2 = glm.normalize(glm.vec3(c2, 1, s2))
            sn_ob2 = glm.normalize(glm.vec3(c2, -1, s2))
            sn_it2 = glm.normalize(glm.vec3(-c2, 1, -s2))
            sn_ib2 = glm.normalize(glm.vec3(-c2, -1, -s2))

            # Helper per afegir vèrtex al buffer: Pos + Norm + SmoothNorm
            def add_vert(pos, norm, smooth_norm):
                vertex_data.extend(pos)
                vertex_data.extend(norm)
                vertex_data.extend((smooth_norm.x, smooth_norm.y, smooth_norm.z))

            # --- TRIANGULACIÓ (2 triangles per cara) ---

            # 1. Cara Superior (Top) - Normal UP
            add_vert(p_it1, n_up, sn_it1); add_vert(p_ot2, n_up, sn_ot2); add_vert(p_ot1, n_up, sn_ot1)
            add_vert(p_it1, n_up, sn_it1); add_vert(p_it2, n_up, sn_it2); add_vert(p_ot2, n_up, sn_ot2)

            # 2. Cara Inferior (Bottom) - Normal DOWN
            add_vert(p_ib1, n_down, sn_ib1); add_vert(p_ob1, n_down, sn_ob1); add_vert(p_ob2, n_down, sn_ob2)
            add_vert(p_ib1, n_down, sn_ib1); add_vert(p_ob2, n_down, sn_ob2); add_vert(p_ib2, n_down, sn_ib2)

            # 3. Cara Exterior (Outer) - Normals OUT
            add_vert(p_ob1, n_out1, sn_ob1); add_vert(p_ot1, n_out1, sn_ot1); add_vert(p_ot2, n_out2, sn_ot2)
            add_vert(p_ob1, n_out1, sn_ob1); add_vert(p_ot2, n_out2, sn_ot2); add_vert(p_ob2, n_out2, sn_ob2)

            # 4. Cara Interior (Inner) - Normals IN
            add_vert(p_ib1, n_in1, sn_ib1); add_vert(p_it2, n_in2, sn_it2); add_vert(p_it1, n_in1, sn_it1)
            add_vert(p_ib1, n_in1, sn_ib1); add_vert(p_ib2, n_in2, sn_ib2); add_vert(p_it2, n_in2, sn_it2)

        self.vbo = self.ctx.buffer(np.array(vertex_data, dtype='f4').flatten())
        
        self.shader = self.get_shader()
        self.vao = self.ctx.vertex_array(
            self.shader,
            # 3f pos, 3f normal, 3f smooth_normal
            [(self.vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_smooth_normal')]
        )
        self.update_uniforms()

    def get_shader(self):
        return self.ctx.program(
            vertex_shader='''
                #version 330
                in vec3 in_position;
                in vec3 in_normal;
                in vec3 in_smooth_normal; // Normal per inflar (outline)

                uniform mat4 m_proj;
                uniform mat4 m_view;
                uniform mat4 m_model;
                uniform float outline_width; // Grosor del borde

                out vec3 v_normal;
                out vec3 v_frag_pos;

                void main() {
                    vec3 pos = in_position;

                    // INFLAT PER AL BORDE NEGRE
                    if (outline_width > 0.0) {
                        pos += in_smooth_normal * outline_width;
                    }

                    vec4 world_pos = m_model * vec4(pos, 1.0);
                    v_frag_pos = world_pos.xyz;

                    // Normal matrix per il·luminació (sense inflar)
                    mat3 normal_matrix = mat3(transpose(inverse(m_model)));
                    v_normal = normalize(normal_matrix * in_normal);

                    gl_Position = m_proj * m_view * world_pos;
                }
            ''',
            fragment_shader='''
                #version 330
                in vec3 v_normal;
                in vec3 v_frag_pos;

                uniform vec3 light_pos;
                uniform vec3 view_pos;
                uniform vec3 ring_color;
                uniform float outline_width;

                out vec4 fragColor;

                void main() {
                    // --- PASADA 1: CONTORNO ---
                    if (outline_width > 0.0) {
                        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
                        return;
                    }

                    // --- PASADA 2: TOON SHADING ---
                    
                    vec3 norm = normalize(v_normal);
                    vec3 light_dir = normalize(light_pos - v_frag_pos);
                    vec3 view_dir = normalize(view_pos - v_frag_pos);

                    // 1. Intensitat Base (Lambert)
                    float intensity = max(dot(norm, light_dir), 0.05);
                    intensity = pow(intensity, 0.7); // No lineal

                    float light_level;

                    // 2. Escalera de Nivells
                    if (intensity > 0.95)      light_level = 1.0;
                    else if (intensity > 0.7)  light_level = 0.8;
                    else if (intensity > 0.3)  light_level = 0.6;
                    else if (intensity > 0.1)  light_level = 0.3;
                    else                       light_level = 0.1; // Sombra oscura

                    // 3. Especular Toon (Brillo)
                    vec3 halfwayDir = normalize(light_dir + view_dir);
                    float NdotH = max(dot(norm, halfwayDir), 0.0);
                    float spec = (pow(NdotH, 64.0) > 0.9) ? 0.5 : 0.0;

                    // 4. Combinar
                    vec3 result = ring_color * light_level + vec3(spec);

                    // 5. Correcció Gamma
                    result = pow(result, vec3(1.0 / 2.2));

                    fragColor = vec4(result, 1.0);
                }
            '''
        )

    def update_uniforms(self):
        self.shader['m_proj'].write(self.camera.m_proj)
        self.shader['m_view'].write(self.camera.m_view)
        self.shader['m_model'].write(self.m_model)
        self.shader['ring_color'].value = self.color
        self.shader['view_pos'].value = tuple(self.camera.position)

    def update(self, position):
        position.y = self.altura
        self.position = position

    def destroy(self):
        self.vbo.release()
        self.shader.release()
        self.vao.release()

    def render(self, light_pos):
        self.m_model = glm.translate(glm.mat4(), self.position)

        # Actualitzem uniforms
        self.shader['m_proj'].write(self.camera.m_proj)
        self.shader['m_view'].write(self.camera.m_view)
        self.shader['m_model'].write(self.m_model)
        self.shader['light_pos'].value = light_pos
        self.shader['view_pos'].value = tuple(self.camera.position)
        
        # Activar Culling
        self.ctx.enable(mgl.CULL_FACE)

        # --- PASADA 1: CONTORNO ---
        self.ctx.cull_face = 'front' # Dibuixar cares posteriors
        if 'outline_width' in self.shader:
            # Borde una mica més prim per l'anell
            self.shader['outline_width'].value = 0.02 
        self.vao.render(mode=mgl.TRIANGLES)

        # --- PASADA 2: OBJECTE ---
        self.ctx.cull_face = 'back' # Dibuixar cares anteriors
        if 'outline_width' in self.shader:
            self.shader['outline_width'].value = 0.0
        self.vao.render(mode=mgl.TRIANGLES)


class Particle:
    """Simple puff particle rendered as a small billboarded quad.

    Supports both the default soft circular masked particles and an optional
    solid-color mode (no mask). The shader exposes uniforms to switch between
    the modes, so you can toggle per-particle or per-emission.
    """
    def __init__(self, ctx, camera, position, color=(1.0, 0.5, 0.0), lifetime=0.6, max_scale=0.25, min_alpha=0.25, solid=False, mask_inner=0.35, mask_outer=0.5, fade_profile='smooth'):
        self.ctx = ctx
        self.camera = camera
        self.position = glm.vec3(position)
        self.color = color
        self.lifetime = lifetime
        self.age = 0.0
        self.scale = 0.05
        self.max_scale = max_scale
        self.min_alpha = min_alpha
        self.solid = bool(solid)
        self.mask_inner = float(mask_inner)
        self.mask_outer = float(mask_outer)
        self.velocity = glm.vec3(
            random.uniform(-0.3, 0.3),
            random.uniform(0.2, 0.8),
            random.uniform(-0.3, 0.3)
        )
        self.is_dead = False
        self._init_shared()
        self.fade_profile = fade_profile

    def _init_shared(self):
        """Initialize shared quad geometry and shader for all particles."""
        if hasattr(Particle, '_shared') and Particle._shared is not None:
            self.shader = Particle._shared['shader']
            self.vao = Particle._shared['vao']
            return

        # Quad (two triangles) with UVs in 0..1
        verts = [
            # x,y,z, u,v
            -0.5, -0.5, 0.0, 0.0, 0.0,
             0.5, -0.5, 0.0, 1.0, 0.0,
             0.5,  0.5, 0.0, 1.0, 1.0,
            -0.5, -0.5, 0.0, 0.0, 0.0,
             0.5,  0.5, 0.0, 1.0, 1.0,
            -0.5,  0.5, 0.0, 0.0, 1.0,
        ]
        vbo = self.ctx.buffer(np.array(verts, dtype='f4').flatten())

        shader = self._get_shader_quad()
        vao = self.ctx.vertex_array(shader, [(vbo, '3f 2f', 'in_position', 'in_uv')])

        Particle._shared = {'vbo': vbo, 'shader': shader, 'vao': vao}
        self.shader = shader
        self.vao = vao

    def _get_shader(self):
        return self.ctx.program(
            vertex_shader='''
                #version 330
                in vec3 in_position;
                uniform mat4 m_proj;
                uniform mat4 m_view;
                uniform mat4 m_model;
                void main() {
                    gl_Position = m_proj * m_view * m_model * vec4(in_position, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                uniform vec3 particle_color;
                uniform float alpha;
                out vec4 fragColor;
                void main() {
                    fragColor = vec4(particle_color, alpha);
                }
            '''
        )

    def _get_shader_quad(self):
        return self.ctx.program(
            vertex_shader='''
                #version 330
                in vec3 in_position; // xy = quad coords (-0.5..0.5)
                in vec2 in_uv;
                out vec2 v_uv;
                uniform mat4 m_proj;
                uniform mat4 m_view;
                uniform vec3 center;
                uniform vec3 cam_right;
                uniform vec3 cam_up;
                uniform float scale;
                void main() {
                    v_uv = in_uv;
                    vec3 world_pos = center + cam_right * in_position.x * scale + cam_up * in_position.y * scale;
                    gl_Position = m_proj * m_view * vec4(world_pos, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                in vec2 v_uv;
                uniform vec3 particle_color;
                uniform float alpha;
                // Solid mode: if u_solid==1, render flat color inside a circular discard.
                uniform int u_solid;
                // Mask parameters for the soft circular mask
                uniform float mask_inner;
                uniform float mask_outer;
                out vec4 fragColor;
                void main() {
                    // compute circular mask for both solid and soft modes
                    vec2 uv = v_uv - vec2(0.5);
                    float d = length(uv);
                    float mask = 1.0 - smoothstep(mask_inner, mask_outer, d);
                    // discard outside circle to get a round particle
                    if (mask <= 0.005) discard;

                    if (u_solid == 1) {
                        // Solid: inside the circle render flat color (respect alpha)
                        fragColor = vec4(particle_color, alpha);
                        return;
                    }

                    // Soft: multiply alpha by mask for smooth falloff
                    fragColor = vec4(particle_color, alpha * mask);
                }
            '''
        )

    def update(self, delta_time):
        self.age += delta_time
        if self.age >= self.lifetime:
            self.is_dead = True
            return
        # Para rastros (late_fade): no mover ni aplicar gravedad (evita que caigan al infinito)
        if getattr(self, "fade_profile", "smooth") != "late_fade":
            self.position += self.velocity * delta_time
            self.velocity.y -= 1.5 * delta_time

        progress = self.age / self.lifetime

        if getattr(self, "fade_profile", "smooth") == "late_fade":
            # aparición rápida (independiente del lifetime largo del rastro)
            grow_time = 0.12  # prueba 0.08–0.18
            t = min(1.0, self.age / grow_time)
            self.scale = self.max_scale * t
        else:
            # comportamiento original para partículas normales (puff)
            if progress < 0.3:
                self.scale = self.max_scale * (progress / 0.3)
            else:
                self.scale = self.max_scale

    def render(self, alpha_mul=1.0):
        if self.is_dead:
            return

        progress = self.age / self.lifetime

        def smoothstep01(x: float) -> float:
            x = max(0.0, min(1.0, x))
            return x * x * (3.0 - 2.0 * x)
        
        if getattr(self, "fade_profile", "smooth") == "late_fade":
            start_fade = 0.85  # solo se desvanece al final
            max_alpha = 0.60   # <- menos opaco

            if progress <= start_fade:
                alpha = max_alpha
            else:
                t = (progress - start_fade) / (1.0 - start_fade)
                alpha = max_alpha * max(0.0, 1.0 - t)

            alpha *= float(alpha_mul)
            alpha = max(0.0, min(1.0, alpha))
        else:
            # Fade in/out en segundos (ajusta si quieres)
            fade_in_time = 0.08
            fade_out_time = 0.12

            fade_in = smoothstep01(self.age / fade_in_time) if fade_in_time > 0 else 1.0
            remaining = self.lifetime - self.age
            fade_out = smoothstep01(remaining / fade_out_time) if fade_out_time > 0 else 1.0

            alpha = (1.0 - progress) * fade_in * fade_out
            alpha *= float(alpha_mul)
            alpha = max(0.0, min(1.0, alpha))

        # Billboard facing camera: pass center and camera axes
        self.shader['m_proj'].write(self.camera.m_proj)
        self.shader['m_view'].write(self.camera.m_view)
        self.shader['center'].value = tuple(self.position)
        self.shader['cam_right'].value = tuple(self.camera.right)
        self.shader['cam_up'].value = tuple(self.camera.up)
        self.shader['scale'].value = self.scale
        self.shader['particle_color'].value = self.color
        self.shader['alpha'].value = alpha
        self.shader['u_solid'].value = 1 if self.solid else 0
        self.shader['mask_inner'].value = self.mask_inner
        self.shader['mask_outer'].value = self.mask_outer
        self.vao.render(mode=mgl.TRIANGLES)


class Particles:
    """Particle generator managing multiple `Particle` instances.

    API:
      - emit(position, num=6, color=None, lifetime=1.0, radius=None, solid=None, mask_inner=None, mask_outer=None)
      - update(delta_time)
      - render()

    Examples:
      # Default soft, circular particles (the current configuration)
      particles = Particles(ctx, camera)
      particles.emit(pos, num=8, color=(1,0,0), radius=1.0)

      # Solid color particles (no soft mask)
      particles_solid = Particles(ctx, camera, default_solid=True)
      particles_solid.emit(pos, num=8, color=(0,1,0), radius=1.0)

    You can also toggle per-emission using the `solid` argument to `emit()`.
    """
    def __init__(self, ctx, camera, default_solid=False, mask_inner=0.35, mask_outer=0.5, min_alpha=0.25):
        self.ctx = ctx
        self.camera = camera
        self.particles = []
        # default visual parameters for emissions
        self.default_solid = bool(default_solid)
        self.mask_inner = float(mask_inner)
        self.mask_outer = float(mask_outer)
        self.min_alpha = float(min_alpha)

    def emit(self, position, num=6, color=None, lifetime=1.0, radius=None, solid=None, mask_inner=None, mask_outer=None, min_alpha=None, fade_profile=None):
        """Emit `num` particles around `position`.

        If `radius` is provided, particles spawn uniformly within a sphere of that radius
        (centered at position + vec3(0,1.0,0)). Color and lifetime are forwarded to
        `Particle` instances.
        """
        color = color or (1.0, 0.5, 0.0)
        r = radius if radius is not None else 0.25

        profile = fade_profile if fade_profile is not None else "smooth"

        def random_point_in_sphere(rval):
            if rval <= 0.0:
                return glm.vec3(0.0, 0.0, 0.0)
            while True:
                x = random.uniform(-1.0, 1.0)
                y = random.uniform(-1.0, 1.0)
                z = random.uniform(-1.0, 1.0)
                if x * x + y * y + z * z <= 1.0:
                    return glm.vec3(x, y, z) * rval

        # Choose a particle scale proportional to the emission radius so larger
        # infection zones don't produce unreasonably large particles.
        particle_max_scale = max(0.05, min(0.90, r * 0.2))
        # Resolve per-emission visual overrides or fall back to system defaults
        solid_mode = self.default_solid if solid is None else bool(solid)
        inner = mask_inner if mask_inner is not None else self.mask_inner
        outer = mask_outer if mask_outer is not None else self.mask_outer
        min_a = min_alpha if min_alpha is not None else self.min_alpha

        for _ in range(num):
            jitter = random_point_in_sphere(r)
            p = Particle(
                self.ctx, self.camera,
                position + glm.vec3(0, 1.0, 0) + jitter,
                color=color, lifetime=lifetime,
                max_scale=particle_max_scale,
                min_alpha=min_a,
                solid=solid_mode,
                mask_inner=inner,
                mask_outer=outer,
                fade_profile=profile
            )
            self.particles.append(p)

    def update(self, delta_time):
        for p in self.particles[:]:
            p.update(delta_time)
            if p.is_dead:
                # shared GL resources are not released per-particle
                self.particles.remove(p)

    def render(self, alpha_mul: float = 1.0, blend_additive: bool = False):
        self.ctx.enable(mgl.BLEND)
        if blend_additive:
            self.ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE
        else:
            self.ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA
        try:
            prev_depth_mask = getattr(self.ctx, 'depth_mask', True)
            self.ctx.depth_mask = False
        except Exception:
            prev_depth_mask = None

        if len(self.particles) < 200:
            sorted_particles = sorted(
                self.particles,
                key=lambda p: glm.length2(p.position - self.camera.position),
                reverse=True
            )
        else:
            sorted_particles = self.particles

        for p in sorted_particles:
            p.render(alpha_mul=alpha_mul)

        try:
            if prev_depth_mask is not None:
                self.ctx.depth_mask = prev_depth_mask
        except Exception:
            pass

        self.ctx.disable(mgl.BLEND)