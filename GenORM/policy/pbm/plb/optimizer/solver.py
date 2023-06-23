import datetime
import taichi as ti
import numpy as np
import pickle
import json
from yacs.config import CfgNode as CN
from datetime import datetime

import datetime, os, cv2
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
import random

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

    def solve(self, init_actions=None, callbacks=()):
        env = self.env
        if init_actions is None:
            init_actions = self.init_actions(env, self.cfg)
        # initialize ...
        optim = OPTIMS[self.optim_cfg.type](init_actions, self.optim_cfg)
        # set softness ..
        env_state = env.get_state()
        self.total_steps = 0

        def forward(sim_state, action):
            if self.logger is not None:
                self.logger.reset()

            env.set_state(sim_state, self.cfg.softness, False)
            with ti.Tape(loss=env.loss.loss):
                for i in range(len(action)):
                    env.step(action[i])
                    self.total_steps += 1
                    loss_info = env.compute_loss()
                    if self.logger is not None:
                        self.logger.step(None, None, loss_info['reward'], None, i==len(action)-1, loss_info)
            loss = env.loss.loss[None]
            return loss, env.primitives.get_grad(len(action))

        best_action = None
        best_loss = 1e10

        actions = init_actions
        for iter in range(self.cfg.n_iters):
            self.params = actions.copy()
            loss, grad = forward(env_state['state'], actions)
            if loss < best_loss:
                best_loss = loss
                best_action = actions.copy()
            actions = optim.step(grad)
            for callback in callbacks:
                callback(self, optim, loss, grad)

        env.set_state(**env_state)
        return actions
        # return best_action


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


def tell_rope_break(image):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define pink color range in HSV
    lower_pink = np.array([140, 50, 50])
    upper_pink = np.array([170, 255, 255])

    # Create a mask for pink color
    mask = cv2.inRange(hsv, lower_pink, upper_pink)

    # Find contours in the masked image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Count the number of contours
    num_pink_objects = len(contours)
    return num_pink_objects > 1


def rope_action(env, output_path, flag=None, T=12, step_num=50):
    # first step: 10 time step same action (0, 1]
    for action_value in np.linspace(0.1, 1, 10):
        env.reset()
        first_action = np.array([[action_value, 0, 0]]*T)
        if flag:
            dummy_action = np.zeros((T, 4))
            first_action = np.concatenate([first_action, dummy_action], axis=1)
        # frames = []
        for idx, act in enumerate(first_action):
            env.step(act)
            if idx+1 == T:
                img = env.render(mode='rgb_array')
        #     img = env.render(mode='rgb_array')
        #     pimg = Image.fromarray(img)
        #     frames.append(pimg)
        # frames[0].save(f'{output_path}/first_{action_value}_demo.gif', save_all=True, append_images=frames[1:], loop=0)
        rope_state = env.taichi_env.simulator.get_x(0)
        rope_length = rope_state.max(axis=0)[0] - rope_state.min(axis=0)[0] 
        possible = tell_rope_break(img)
        if possible:
            action_value -= 0.1
            break

    # second step: 10 time step action [action_value - 0.1, action_value + 0.1]
    if flag:
        actions_list = np.random.normal(action_value, 0.05, (step_num, T, 7))
    else:
        actions_list = np.random.normal(action_value, 0.05, (step_num, T, 3))

    best_rope_length = 0
    best_action = None

    for step, second_action in enumerate(actions_list):
        print(step,'/', step_num)
        second_action = np.concatenate([second_action])
        env.reset()
        # frames = []
        for idx, act in enumerate(second_action):
            env.step(act)
            if idx+1 == T:
                img = env.render(mode='rgb_array')
        #     img = env.render(mode='rgb_array')
        #     pimg = Image.fromarray(img)
        #     frames.append(pimg)
        # frames[0].save(f'{output_path}/second_{action_value}_demo.gif', save_all=True, append_images=frames[1:], loop=0)
        rope_state = env.taichi_env.simulator.get_x(0)
        rope_length = rope_state.max(axis=0)[0] - rope_state.min(axis=0)[0] 
        possible = tell_rope_break(img)
        if rope_length > best_rope_length and not possible:
            best_rope_length = rope_length
            best_action = second_action
    return best_action

def solve_action(env, path, logger, args):
    repeat_time = 1
    for i in range(repeat_time):
        idx = args.env_name.find('-')
        args.task_name = args.env_name[:idx]
        args.task_version = args.env_name[(idx+1):]
        now = datetime.datetime.now()
        # E_bottom, E_upper = 1779.38, 1779.38
        # Poisson_bottom, Poisson_upper = 0.35, 0.35
        # E_bottom, E_upper = 3276.12, 3276.12
        # Poisson_bottom, Poisson_upper = 0.346, 0.346
        E_bottom, E_upper = 1500, 1500
        Poisson_bottom, Poisson_upper = 0.36, 0.36
        yield_stress_bottom, yield_stress_upper = 200, 200
        output_path = f'{path}/{args.task_name}_{E_bottom}_{E_upper}_{Poisson_bottom}_{Poisson_upper}_{yield_stress_bottom}_{yield_stress_upper}/{env.spec.id}/{now}'
        os.makedirs(output_path, exist_ok=True)
        env.reset()
        img = env.render(mode='rgb_array')
        cv2.imwrite(f"{output_path}/init.png", img[..., ::-1])
        taichi_env: TaichiEnv = env.unwrapped.taichi_env
        T = env._max_episode_steps

        # set randam parameter
        np.random.seed(int(args.task_version[1:])*repeat_time+i)
        E = np.random.uniform(E_bottom, E_upper)
        Poisson = np.random.uniform(Poisson_bottom, Poisson_upper)
        yield_stress = np.random.uniform(yield_stress_bottom, yield_stress_upper)
        print('parameter', E, Poisson, yield_stress)
        env.taichi_env.set_parameter(E, Poisson, yield_stress)
        
        experts_output_dir = f'/root/ExPCP/policy/pbm/experts/{args.task_name}_{E_bottom}_{E_upper}_{Poisson_bottom}_{Poisson_upper}_{yield_stress_bottom}_{yield_stress_upper}/{env.spec.id}'
        if not os.path.exists(experts_output_dir):
            os.makedirs(experts_output_dir, exist_ok=True)

        if args.task_name in ['Move']:
            action = np.array([[0, 0.6, 0]]*150)
        elif args.task_name in ['Torus']:
            random.seed(int(args.task_version[1:])*repeat_time+i)
            ranges = [(0.1, 0.3), (0.1, 0.6), (0.5, 0.5)]
            samples = []
            for r in ranges:
                samples.append(random.uniform(*r))
            # start_pos = np.array(samples)
            # start_pos = np.array([0.1, 0.5, 0.5])
            # start_pos = np.array([0.2, 0.4, 0.5])
            start_pos = np.array([0.2, 0.3, 0.5])
            initial_primitive_pos = env.taichi_env.primitives[0].get_state(0)[:3]
            init_actions = np.linspace(initial_primitive_pos, start_pos, 200)
            init_actions = np.diff(init_actions, n=1, axis=0)
            init_actions = np.vstack([init_actions, init_actions[0][None, :]])
            action = np.concatenate([init_actions, np.array([[0, 0, 0]]*400)])
            
            initial_state = env.taichi_env.simulator.get_x(0)
            rope_bottom_index = np.argmin(initial_state[:, 1])

            try:
                best_max_x = 0
                env.taichi_env.primitives.set_softness()
                frames = []
                for idx, act in enumerate(action):
                    env.step(act)
                    
                    if idx == 300:
                        release_point = env.taichi_env.primitives[0].get_state(0)[:3]
                        env.taichi_env.primitives.set_softness1(0)
                    if idx % 5 == 0:
                        img = env.render(mode='rgb_array')
                        cv2.imwrite(f"{output_path}/{idx}.png", img[..., ::-1])
                        pimg = Image.fromarray(img)
                        I1 = ImageDraw.Draw(pimg)
                        I1.text((5, 5), f'E{E:.2f},Poisson{Poisson:.2f},yield_stress{yield_stress:.2f}', fill=(255, 0, 0))
                        frames.append(pimg)

                    state = env.taichi_env.simulator.get_x(0)
                    max_x = state[rope_bottom_index][0]
                    if max_x > best_max_x:
                        best_max_x = max_x
                frames[0].save(f'{output_path}/best_x{best_max_x:.2f}_x{start_pos[0]:.2f}_y{start_pos[1]:.2f}_E{E:.2f},Poisson{Poisson:.2f},yield_stress{yield_stress:.2f}_.gif', save_all=True, append_images=frames[1:], loop=0)

                bc_data = {
                    'release_point': release_point,
                    'max_x': best_max_x,
                    'action': action,
                    'E': E,
                    'Poisson': Poisson,
                    'yield_stress': yield_stress,
                    'env_name': args.env_name,
                }

                now = datetime.datetime.now()
                current_time = now.strftime("%H:%M:%S")
                with open(f'{experts_output_dir}/expert_best_x{best_max_x:.2f}_x{start_pos[0]:.2f}_y{start_pos[1]:.2f}_{current_time}.pickle', 'wb') as f:
                    pickle.dump(bc_data, f)
            except:
                print('Nan error')
                pass
            continue
        else:
            solver = Solver(taichi_env, logger, None,
                            n_iters=(args.num_steps + T-1)//T, softness=args.softness, horizon=T,
                            **{"optim.lr": args.lr, "optim.type": args.optim, "init_range": 0.0001})
            action = solver.solve()

        np.save(f"{output_path}/action.npy", action)
        print(action)

        env.reset()
        try:
            frames = []
            env.taichi_env.primitives.set_softness()
            for idx, act in enumerate(action):
                env.step(act)
                if idx == 200:
                    env.taichi_env.primitives.set_softness1(0)
                if idx % 5 == 0:
                    img = env.render(mode='rgb_array')
                    pimg = Image.fromarray(img)
                    I1 = ImageDraw.Draw(pimg)
                    I1.text((5, 5), f'E{E:.2f},Poisson{Poisson:.2f},yield_stress{yield_stress:.2f}', fill=(255, 0, 0))
                    frames.append(pimg)
            frames[0].save(f'{output_path}/demo.gif', save_all=True, append_images=frames[1:], loop=0)

            # create dataset for bc
            env.reset()
            action_list = []
            plasticine_pc_list = []
            primitive_pc_list = []
            reward_list = []
            loss_info_list = []

            for i in range(len(action)):
                action_list.append(action[i])

                plasticine_pc = env.taichi_env.simulator.get_x(0)
                primtiive_pc = env.taichi_env.primitives[0].get_state(0)[:3]
                plasticine_pc_list.append(plasticine_pc.tolist())
                primitive_pc_list.append(primtiive_pc.tolist())

                obs, r, done, loss_info = env.step(action[i])
                last_iou = loss_info['incremental_iou']
                reward_list.append(r)
                loss_info_list.append(loss_info)

            print('length', i, 'r', r, 'last_iou', last_iou)
            bc_data = {
                'action': np.array(action_list),
                'E': E,
                'Poisson': Poisson,
                'yield_stress': yield_stress,
                'rewards': np.array(reward_list),
                'env_name': args.env_name,
                'plasticine_pc': np.array(plasticine_pc_list),
                'primitive_pc': np.array(primitive_pc_list),
                'last_iou': last_iou
            }

            now = datetime.datetime.now()
            current_time = now.strftime("%H:%M:%S")
            with open(f'{experts_output_dir}/expert_{last_iou:.4f}_{current_time}.pickle', 'wb') as f:
                pickle.dump(bc_data, f)
            with open(f'{output_path}/iou_{last_iou}.txt', 'w') as f:
                f.write(str(last_iou))
        except:
            print('Nan error')
            pass