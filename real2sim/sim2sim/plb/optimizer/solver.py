import datetime
import taichi as ti
import numpy as np
import pickle
from yacs.config import CfgNode as CN
from datetime import datetime

from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw
from scipy.spatial.distance import cdist

from .optim import Optimizer, Adam, Momentum
from ..engine.taichi_env import TaichiEnv
from ..config.utils import make_cls_config

OPTIMS = {
    'Adam': Adam,
    'Momentum': Momentum
}

YIELD_STRESS = 500

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
            env.set_parameter(parameter[0], parameter[1], YIELD_STRESS)
            env.set_state(sim_state, self.cfg.softness, False)
            with ti.Tape(loss=env.loss.loss):
                for i in range(len(action)):
                    loss_info = env.compute_loss()

                    env.step(action[i])
                    self.total_steps += 1

                    if self.logger is not None:
                        self.logger.step(None, None, loss_info['reward'], None, i==len(action)-1, loss_info)
            loss = env.loss.loss[None]
            return loss, loss_info['reward'], env.simulator.get_parameter_grad()

        best_parameters = None
        best_loss = 1e10
        parameters_list = []
        reward_list = []

        parameters = init_parameters
        for iter in range(self.cfg.n_iters):
            loss, reward, grad = forward(env_state['state'], parameters, actions)
            if loss < best_loss:
                best_loss = loss
                best_parameters = parameters
            parameters = optim.step(grad)
            parameters[2] = YIELD_STRESS
            parameters_list.append(parameters.tolist())
            reward_list.append(reward)
            print('loss', loss, 'reward', reward, parameters, grad)
            for callback in callbacks:
                callback(self, optim, loss, grad)

        env.set_state(**env_state)
        return best_parameters, parameters_list, reward_list


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

    BASE_TASK = 'Move_1000_8000_1000_8000_500_2000'
    base_path = f'/root/real2sim/sim2sim/output/{BASE_TASK}'
    output_path = f'{base_path}/{now}'
    os.makedirs(output_path, exist_ok=True)

    parameter_list = BASE_TASK.split('_')[1:]
    parameter_list = [int(parameter) for parameter in parameter_list]
    mu_bottom, mu_upper = parameter_list[0], parameter_list[1]
    lam_bottom, lam_upper = parameter_list[2], parameter_list[3]
    yield_stress_bottom, yield_stress_upper = parameter_list[4], parameter_list[5]

    action = np.array([[0, 0.6, 0]]*150)

    for t in tqdm(range(1000, 1005)):
        # collect ground truth
        np.random.seed(t)
        mu = np.random.uniform(mu_bottom, mu_upper)
        lam = np.random.uniform(lam_bottom, lam_upper)
        yield_stress = np.random.uniform(yield_stress_bottom, yield_stress_upper)
        env.taichi_env.set_parameter(mu, lam, yield_stress)
        env.reset()
        import pdb; pdb.set_trace()
        frames = []
        target_grids = []
        for idx, act in enumerate(action):
            env.step(act)
            if idx % 5 == 0:
                target_grids.append(env.taichi_env.get_grid_mass())
                img = env.render(mode='rgb_array')
                pimg = Image.fromarray(img)
                I1 = ImageDraw.Draw(pimg)
                I1.text((5, 5), f'mu{mu:.2f},lam{lam:.2f},yield_stress{yield_stress:.2f}', fill=(255, 0, 0))
                frames.append(pimg)
        frames[0].save(f'{output_path}/ground_truth_demo.gif', save_all=True, append_images=frames[1:], loop=0)
        last_state = env.taichi_env.simulator.get_x(0)
        import pdb; pdb.set_trace()

        target_grids = np.repeat(np.array(target_grids), env.taichi_env.simulator.substeps, axis=0)
        T = action.shape[0]
        args.num_steps = T * 30

        for i in tqdm(range(5)):
            # set test
            env.reset()
            taichi_env: TaichiEnv = env.unwrapped.taichi_env
            taichi_env.loss.update_target_density(target_grids)

            np.random.seed(i)
            initial_mu = np.random.uniform(mu_bottom, mu_upper)
            initial_lam = np.random.uniform(lam_bottom, lam_upper)
            initial_yield_stress = np.random.uniform(yield_stress_bottom, yield_stress_upper)
            init_parameters = np.array([initial_mu, initial_lam, initial_yield_stress])
            env.taichi_env.set_parameter(initial_mu, initial_lam, initial_yield_stress)

            solver = Solver(taichi_env, logger, None,
                        n_iters=(args.num_steps + T-1)//T, softness=args.softness, horizon=T,
                        **{"optim.lr": args.lr, "optim.type": args.optim, "init_range": 0.0001})
            best_parameters, parameters_list, reward_list = solver.solve(init_parameters, action)
            np.save(f"{output_path}/{i}_parameters.npy", np.array(parameters_list))
            np.save(f"{output_path}/{i}_rewards.npy", np.array(reward_list))

            optimized_mu = parameters_list[-1][0]
            optimized_lam = parameters_list[-1][1]
            optimized_yield_stress = parameters_list[-1][2]
            print(optimized_mu, optimized_lam, optimized_yield_stress)
            env.taichi_env.set_parameter(optimized_mu, optimized_lam, optimized_yield_stress)

            frames = []
            for idx, act in enumerate(action):
                env.step(act)
                if idx % 5 == 0:
                    img = env.render(mode='rgb_array')
                    pimg = Image.fromarray(img)
                    I1 = ImageDraw.Draw(pimg)
                    I1.text((5, 5), f'mu{optimized_mu:.2f},lam{optimized_lam:.2f},yield_stress{optimized_yield_stress:.2f}', fill=(255, 0, 0))
                    frames.append(pimg)
            frames[0].save(f'{output_path}/{i}_pred_demo.gif', save_all=True, append_images=frames[1:], loop=0)
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
            chamfer_dist = chamfer_distance(last_state, pred_last_state)

            with open(f'{output_path}/{i}.txt', 'w') as f:
                f.write(f'{chamfer_dist}, {mu}, {lam}, {yield_stress}, {initial_mu}, {initial_lam}, {initial_yield_stress}, {optimized_mu},{optimized_lam},{optimized_yield_stress}')
