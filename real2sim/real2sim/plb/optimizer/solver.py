import datetime
import taichi as ti
import numpy as np
import pickle
import shutil
from yacs.config import CfgNode as CN
from datetime import datetime

from scipy.spatial.distance import cdist
from .optim import Optimizer, Adam, Momentum
from ..engine.taichi_env import TaichiEnv
from ..config.utils import make_cls_config

OPTIMS = {
    'Adam': Adam,
    'Momentum': Momentum
}

YIELD_STRESS = 200


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
            env.set_parameter(parameter[0], parameter[1], YIELD_STRESS) # mu, lam, yield_stress
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
            print('--------')
            parameters = optim.step(grad)
            print('loss:', loss, 'E:', parameters[0], 'Poisson:', parameters[1], 'yield_stress:', parameters[2])
            parameters[2] = YIELD_STRESS
            parameters[1] = np.clip(parameters[1], 0.20, 0.4)
            parameters[0] = np.clip(parameters[0], 500, 10700)
            parameters = np.clip(parameters, 0.01, 9999999999999999)
            parameters_list.append(parameters.tolist())
            print('loss:', loss, 'E:', parameters[0], 'Poisson:', parameters[1], 'yield_stress:', parameters[2])
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
    for t in range(2004, 2010):
        rope_type = args.rope_type
        input_path = f'/root/real2sim/real2sim/real_points/{rope_type}'
        output_path = f'/root/real2sim/real2sim/real_points/{rope_type}/{now}/{t}'
        os.makedirs(output_path, exist_ok=True)

        env.reset()
        img = env.render(mode='rgb_array')
        cv2.imwrite(f"{output_path}/init.png", img[..., ::-1])
        taichi_env: TaichiEnv = env.unwrapped.taichi_env

        actions = np.array([[0, 0.6, 0]]*150)
        target_grids = np.load(f'{input_path}/real_densities.npy')
        target_grids = np.repeat(target_grids, env.taichi_env.simulator.substeps, axis=0)
        T = actions.shape[0]
        args.num_steps = T * 100
        taichi_env.loss.update_target_density(target_grids)
        E_bottom, E_upper = 2000, 8000
        Poisson_bottom, Poisson_upper = 0.2, 0.4
        yield_stress_bottom, yield_stress_upper = 200, 200

        np.random.seed(t)
        E = np.random.uniform(E_bottom, E_upper)
        Poisson = np.random.uniform(Poisson_bottom, Poisson_upper)
        yield_stress = np.random.uniform(yield_stress_bottom, yield_stress_upper)
        init_parameters = np.array([E, Poisson, yield_stress])

        # save initial gif
        env.taichi_env.set_parameter(init_parameters[0], init_parameters[1], init_parameters[2])
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
        frames[0].save(f'{output_path}/initial_E{init_parameters[0]}_lam{init_parameters[1]}_yield{init_parameters[2]}.gif',
                    save_all=True, append_images=frames[1:], loop=0)
        env.reset()

        # optimize
        solver = Solver(taichi_env, logger, None,
                        n_iters=(args.num_steps + T-1)//T, softness=args.softness, horizon=T,
                        **{"optim.lr": args.lr, "optim.type": args.optim, "init_range": 0.0001})
        best_parameters, parameters_list = solver.solve(init_parameters, actions)
        np.save(f"{output_path}/parameters.npy", np.array(parameters_list))
        print(parameters_list[-1])

        # save optimized gif
        env.taichi_env.set_parameter(parameters_list[-1][0], parameters_list[-1][1], parameters_list[-1][2])
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
        pred_last_state = env.taichi_env.simulator.get_x(0)

        def chamfer_distance(A, B):
            # compute distance matrix between A and B
            dist_matrix = cdist(A, B)
            # for each point in A, compute minimum distance to any point in B
            dist_A = np.min(dist_matrix, axis=1)
            # for each point in B, compute minimum distance to any point in A
            dist_B = np.min(dist_matrix, axis=0)
            # compute Chamfer distance
            chamfer_dist = np.mean(dist_A) + np.mean(dist_B)
            return chamfer_dist

        last_state = np.load(f'{input_path}/real_pcds_modify.npy', allow_pickle=True)[-1]
        chamfer_dist = chamfer_distance(last_state, pred_last_state)

        frames[0].save(f'{output_path}/optimized_E{parameters_list[-1][0]}_Poisson{parameters_list[-1][1]}_yield{parameters_list[-1][2]}.gif',
            save_all=True, append_images=frames[1:], loop=0)
        with open(f'{output_path}/setting.txt', 'w') as f:
            f.write(f'{chamfer_dist}, {E}, {Poisson}, {yield_stress}, {parameters_list[-1][0]}, {parameters_list[-1][1]}, {parameters_list[-1][2]}')
