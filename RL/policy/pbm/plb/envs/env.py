import gym
from gym.spaces import Box
import os
import yaml
import numpy as np
from ..config import load
from yacs.config import CfgNode
from .utils import merge_lists

PATH = os.path.dirname(os.path.abspath(__file__))

class PlasticineEnv(gym.Env):
    def __init__(self, cfg_path, version, nn=False):
        from ..engine.taichi_env import TaichiEnv
        self.cfg_path = cfg_path
        cfg = self.load_varaints(cfg_path, version)
        self.taichi_env = TaichiEnv(cfg, nn) # build taichi environment
        self.taichi_env.initialize()
        self.cfg = cfg.ENV
        self.taichi_env.set_copy(True)
        self._init_state = self.taichi_env.get_state()
        self._n_observed_particles = self.cfg.n_observed_particles

        obs = self.reset()
        self.observation_space = Box(-np.inf, np.inf, obs.shape)
        self.action_space = Box(-1, 1, (self.taichi_env.primitives.action_dim,))
        
        self.primitive_pc = None
        self.primitive_center = None

    def reset(self):
        self.taichi_env.set_state(**self._init_state)
        self._recorded_actions = []
        return self._get_obs()

    def _get_obs(self, t=0):
        x = self.taichi_env.simulator.get_x(t)
        outs = []
        for i in self.taichi_env.primitives:
            outs.append(i.get_state(t))
        primitive_pos = np.concatenate(outs)[:3]
        step_size = len(x) // self._n_observed_particles
        mu = self.taichi_env.simulator.mu_value
        lam = self.taichi_env.simulator.lam_value
        yield_stress = self.taichi_env.simulator.yield_stress_value
        parameters = np.array([mu, lam, yield_stress])
        return np.concatenate([x[::step_size].reshape(-1), primitive_pos, parameters])
    
    def get_obs(self, t=0, time=0):
        plasticine_pc = self.taichi_env.simulator.get_x(t)
        if time == 0 or time == 1:
            self.primitive_pc = self.taichi_env.simulator.get_point_cloud(t)
            self.primitive_center = self.taichi_env.primitives[0].get_state(t)[:3]
        else:
            primitive_center = self.taichi_env.primitives[0].get_state(t)[:3]
            self.primitive_pc = self.primitive_pc + primitive_center - self.primitive_center
            self.primitive_center = primitive_center
        return plasticine_pc, self.primitive_pc

    def step(self, action):
        self.taichi_env.step(action)
        loss_info = self.taichi_env.compute_loss()

        self._recorded_actions.append(action)
        obs = self._get_obs()

        x = self.taichi_env.simulator.get_x(0)
        r = x.max(axis=0)[0] - x.min(axis=0)[0]

        loss_info['reward'] = r

        if np.isnan(obs).any() or np.isnan(r):
            if np.isnan(r):
                print('nan in r')
            import pickle, datetime
            with open(f'{self.cfg_path}_nan_action_{str(datetime.datetime.now())}', 'wb') as f:
                pickle.dump(self._recorded_actions, f)
            raise Exception("NaN..")
        return obs, r, False, loss_info

    def render(self, mode='human'):
        return self.taichi_env.render(mode)

    @classmethod
    def load_varaints(self, cfg_path, version):
        assert version >= 1
        cfg_path = os.path.join(PATH, cfg_path)
        cfg = load(cfg_path)
        variants = cfg.VARIANTS[version - 1]

        new_cfg = CfgNode(new_allowed=True)
        new_cfg = new_cfg._load_cfg_from_yaml_str(yaml.safe_dump(variants))
        new_cfg.defrost()
        if 'PRIMITIVES' in new_cfg:
            new_cfg.PRIMITIVES = merge_lists(cfg.PRIMITIVES, new_cfg.PRIMITIVES)
        if 'SHAPES' in new_cfg:
            new_cfg.SHAPES = merge_lists(cfg.SHAPES, new_cfg.SHAPES)
        cfg.merge_from_other_cfg(new_cfg)

        cfg.defrost()
        # set target path id according to version
        name = list(cfg.ENV.loss.target_path)
        
        cfg.ENV.loss.target_path = os.path.join(PATH, '../', ''.join(name))
        cfg.VARIANTS = None
        cfg.freeze()

        return cfg