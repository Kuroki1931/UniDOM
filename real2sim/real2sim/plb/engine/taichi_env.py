import os
import sys
import numpy as np
import cv2
import taichi as ti

# TODO: run on GPU, fast_math will cause error on float64's sqrt; removing it cuases compile error..
ti.init(arch=ti.gpu, debug=False, fast_math=True)

@ti.data_oriented
class TaichiEnv:
    def __init__(self, cfg, nn=False, loss=True):
        """
        A taichi env builds scene according the configuration and the set of manipulators
        """
        # primitives are environment specific parameters ..
        # move it inside can improve speed; don't know why..
        from .mpm_simulator import MPMSimulator
        from .primitive import Primitives
        from .renderer import Renderer
        from .shapes import Shapes
        from .losses import Loss
        from .nn.mlp import MLP
        from ..algorithms.solve import get_args

        self.args = get_args()
        idx = self.args.env_name.find('-')
        self.args.task_name = self.args.env_name[:idx]
        self.args.task_version = self.args.env_name[(idx+1):]
        sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
        if self.args.task_name in ['Move', 'Table']:
            from object.cloth import Cloth
            self.real2sim = Cloth()

        hand_position = self.real2sim.get_hand_position()
        obj_particle = self.real2sim.get_obj_particle()
        surface_index = self.real2sim.get_surface_index()
        self.cfg = cfg.ENV
        self.primitives = Primitives(cfg.PRIMITIVES, hand_position)
        self.shapes = Shapes(cfg.SHAPES, obj_particle)
        self.init_particles, self.particle_colors = self.shapes.get()

        cfg.SIMULATOR.defrost()
        self.n_particles = cfg.SIMULATOR.n_particles = len(self.init_particles)

        self.simulator = MPMSimulator(cfg.SIMULATOR, self.primitives, surface_index)
        self.renderer = Renderer(cfg.RENDERER, self.primitives)

        if nn:
            self.nn = MLP(self.simulator, self.primitives, (256, 256))

        if loss:
            self.loss = Loss(cfg.ENV.loss, self.simulator)
        else:
            self.loss = None
        self._is_copy = True

    def set_copy(self, is_copy: bool):
        self._is_copy= is_copy

    def initialize(self):
        # initialize all taichi variable according to configurations..
        self.primitives.initialize()
        self.simulator.initialize()
        self.renderer.initialize()
        if self.loss:
            self.loss.initialize()
            self.renderer.set_target_density(self.loss.target_density.to_numpy()/self.simulator.p_mass)

        # call set_state instead of reset..
        self.simulator.reset(self.init_particles)
        if self.loss:
            self.loss.clear()
    
    def initialize_update_target(self, target_density_path):
        # initialize all taichi variable according to configurations..
        self.primitives.initialize()
        self.simulator.initialize()
        self.renderer.initialize()
        if self.loss:
            self.loss.initialize(target_density_path)
            self.renderer.set_target_density(np.load(target_density_path)/self.simulator.p_mass)

        # call set_state instead of reset..
        self.simulator.reset(self.init_particles)
        if self.loss:
            self.loss.clear()

    def render(self, mode='human', **kwargs):
        assert self._is_copy, "The environment must be in the copy mode for render ..."
        if self.n_particles > 0:
            x = self.simulator.get_x(0)
            self.renderer.set_particles(x, self.particle_colors)
        img = self.renderer.render_frame(shape=1, primitive=1, **kwargs)
        img = np.uint8(img.clip(0, 1) * 255)

        if mode == 'human':
            cv2.imshow('x', img[..., ::-1])
            cv2.waitKey(1)
        elif mode == 'plt':
            import matplotlib.pyplot as plt
            plt.imshow(img)
            plt.show()
        else:
            return img

    def step(self, action=None):
        if action is not None:
            action = np.array(action)
        self.simulator.step(is_copy=self._is_copy, action=action)

    def compute_loss(self):
        assert self.loss is not None
        if self._is_copy:
            self.loss.clear()
            return self.loss.compute_loss(0)
        else:
            return self.loss.compute_loss(self.simulator.cur)
    
    def get_grid_mass(self, t):
        x = self.simulator.get_grid_mass(t)
        return x

    def get_state(self):
        assert self.simulator.cur == 0
        return {
            'state': self.simulator.get_state(0),
            'softness': self.primitives.get_softness(),
            'is_copy': self._is_copy
        }

    def set_state(self, state, softness, is_copy):
        self.simulator.cur = 0
        self.simulator.set_state(0, state)
        self.primitives.set_softness(softness)
        self._is_copy = is_copy
        if self.loss:
            self.loss.reset()
            self.loss.clear()

    def set_parameter(self, ground_friction):
        self.simulator.set_parameter_kernel(ground_friction)
        # self.simulator.set_parameter(ground_friction)
        # self.simulator.ground_friction = ground_friction