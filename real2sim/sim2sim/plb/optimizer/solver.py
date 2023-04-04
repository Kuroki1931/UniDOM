import datetime
import taichi as ti
import numpy as np
import pickle
from yacs.config import CfgNode as CN
from datetime import datetime

from .optim import Optimizer, Adam, Momentum
from ..engine.taichi_env import TaichiEnv
from ..config.utils import make_cls_config

OPTIMS = {
    'Adam': Adam,
    'Momentum': Momentum
}


class Solver:
    def __init__(self, env: TaichiEnv, logger=None, cfg=None, **kwargs):
        self.cfg = make_cls_config(self, cfg, **kwargs)
        self.optim_cfg = self.cfg.optim
        self.env = env
        self.logger = logger

    def solve(self, init_parameters, actions, callbacks=()):
        env = self.env
        optim = OPTIMS[self.optim_cfg.type](init_parameters, self.optim_cfg)
        # set softness ..
        env_state = env.get_state()
        self.total_steps = 0

        def forward(sim_state, parameter, action):
            if self.logger is not None:
                self.logger.reset()

            # set parameter
            env.set_parameter(parameter[0])
            env.set_state(sim_state, self.cfg.softness, False)
            with ti.Tape(loss=env.loss.loss):
                for i in range(len(action)):
                    loss_info = env.compute_loss()

                    env.step(action[i])
                    self.total_steps += 1

                    if self.logger is not None:
                        self.logger.step(None, None, loss_info['reward'], None, i==len(action)-1, loss_info)
            loss = env.loss.loss[None]
            return loss, env.simulator.get_parameter_grad()

        best_parameters = None
        best_loss = 1e10
        parameters_list = []

        parameters = init_parameters
        for iter in range(self.cfg.n_iters):
            loss, grad = forward(env_state['state'], parameters, actions)
            if loss < best_loss:
                best_loss = loss
                best_parameters = parameters
            parameters = optim.step(grad)
            parameters_list.append(parameters.tolist())
            print(parameters)
            for callback in callbacks:
                callback(self, optim, loss, grad)

        env.set_state(**env_state)
        return best_parameters, parameters_list


    @staticmethod
    def init_actions(env, cfg):
        action_dim = env.primitives.action_dim
        horizon = cfg.horizon
        if cfg.init_sampler == 'uniform':
            return np.random.uniform(-cfg.init_range, cfg.init_range, size=(horizon, action_dim))
        else:
            raise NotImplementedError

    @classmethod
    def default_config(cls):
        cfg = CN()
        cfg.optim = Optimizer.default_config()
        cfg.n_iters = 100
        cfg.softness = 666.
        cfg.horizon = 50

        cfg.init_range = 0.
        cfg.init_sampler = 'uniform'
        return cfg


def solve_action(env, path, logger, args):
    import datetime, os, cv2
    import matplotlib.pyplot as plt
    from PIL import Image
    now = datetime.datetime.now()
    output_path = f'{path}/{env.spec.id}/{now}'
    os.makedirs(output_path, exist_ok=True)

    env.reset()
    img = env.render(mode='rgb_array')
    cv2.imwrite(f"{output_path}/init.png", img[..., ::-1])
    taichi_env: TaichiEnv = env.unwrapped.taichi_env

    actions = np.load('/root/real2sim/sim2sim/test/12:11:54.npy')[:, :3]
    T = actions.shape[0]
    args.num_steps = T * 50
    target_grids = np.load('/root/real2sim/sim2sim/test/fric_1.5/expert_0.0383_12:11:54_grid_mass.npy') 
    taichi_env.loss.update_target_density(target_grids)
    init_parameters = np.array([0.1])

    solver = Solver(taichi_env, logger, None,
                    n_iters=(args.num_steps + T-1)//T, softness=args.softness, horizon=T,
                    **{"optim.lr": args.lr, "optim.type": args.optim, "init_range": 0.0001})
    best_parameters, parameters_list = solver.solve(init_parameters, actions)
    np.save(f"{output_path}/parameters.npy", np.array(parameters_list))
    print(parameters_list[-1][0])
    env.taichi_env.set_parameter(parameters_list[-1][0])
    
    frames = []
    for idx, act in enumerate(actions):
        start_time = datetime.datetime.now()
        env.step(act)
        if idx % 5 == 0:
            img = env.render(mode='rgb_array')
            pimg = Image.fromarray(img)
            frames.append(pimg)
        end_time = datetime.datetime.now()
        take_time = end_time - start_time
        take_time = take_time.total_seconds()
        print('take time', take_time)
    print(output_path)
    frames[0].save(f'{output_path}/demo.gif', save_all=True, append_images=frames[1:], loop=0)
    
    return
