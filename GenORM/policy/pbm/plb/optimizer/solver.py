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


def solve_action(env, path, logger, args):
    repeat_time = 1000
    for i in range(repeat_time):
        idx = args.env_name.find('-')
        args.task_name = args.env_name[:idx]
        args.task_version = args.env_name[(idx+1):]
        now = datetime.datetime.now()
        # E_bottom, E_upper = 1779.38, 1779.38
        # Poisson_bottom, Poisson_upper = 0.35, 0.35
        # E_bottom, E_upper = 3276.12, 3276.12
        # Poisson_bottom, Poisson_upper = 0.346, 0.346
        E_bottom, E_upper = 500, 10500
        Poisson_bottom, Poisson_upper = 0.2, 0.4
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
        
        experts_output_dir = f'/root/GenORM/policy/pbm/experts/{args.task_name}_{E_bottom}_{E_upper}_{Poisson_bottom}_{Poisson_upper}_{yield_stress_bottom}_{yield_stress_upper}/{env.spec.id}'
        if not os.path.exists(experts_output_dir):
            os.makedirs(experts_output_dir, exist_ok=True)

        if args.task_name in ['Table', 'Move']:
            action = np.array([[0, 0.6, 0]]*150)
        if args.task_name in ['Rope']:
            height = 0.32
            window = 30
            steps = 16
            scale= 0.005
            # up
            action_up = np.array([[0, (height / window / steps) / scale, 0]]*window*steps)
            
            # down
            mask_sum_list = []
            acxel_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
            for acxel_value in acxel_list:
                # sum 16. 
                unit_action = height / window / steps
                acxel_rate_list = [
                    1-acxel_value*4, 1-acxel_value*3.5, 1-acxel_value*3, 1-acxel_value*2.5, 1-acxel_value*2, 1-acxel_value*1.5, 1-acxel_value*1, 1-acxel_value*0.5,
                    1+acxel_value*0.5, 1+acxel_value*1, 1+acxel_value*1.5, 1+acxel_value*2, 1+acxel_value*2.5, 1+acxel_value*3, 1+acxel_value*3.5, 1+acxel_value*4
                ]
                action_list = np.array(acxel_rate_list) * unit_action
                action_list = np.clip(action_list, 0, 1000000)
                action_list = np.repeat(action_list, window)
                # scale
                action_list = action_list / scale
                action_down = np.array([[0, -i, 0] for i in action_list[::-1]])
                
                action = np.concatenate([action_up, action_down])
                action = np.concatenate([action, np.array([[0, 0, 0]]*100)])
           
                env.reset()
                env.taichi_env.primitives.set_softness()
                frames = []
                for idx, act in enumerate(action):
                    env.step(act)
                    if idx == window * steps - 1:
                        state = env.taichi_env.simulator.get_x(0)
                    # if idx % 5 == 0:
                    #     img = env.render(mode='rgb_array')
                    #     pimg = Image.fromarray(img)
                    #     I1 = ImageDraw.Draw(pimg)
                    #     I1.text((5, 5), f'axcel{acxel_value:.3f},E{E:.2f},Poisson{Poisson:.2f},yield_stress{yield_stress:.2f}', fill=(255, 0, 0))
                    #     frames.append(pimg)
                img = env.render(mode='rgb_array')
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                # Define blue color range in HSV
                lower_blue = np.array([120, 150, 150])
                upper_blue = np.array([140, 255, 255])

                # Create a mask for pink color
                mask = cv2.inRange(hsv, lower_blue, upper_blue)
                mask_sum = mask.sum()
                mask_sum_list.append(mask_sum)
                # frames[0].save(f'{output_path}/axcel{acxel_value:.3f}_mask_sum{mask_sum}_E{E:.2f},Poisson{Poisson:.2f},yield_stress{yield_stress:.2f}_.gif', save_all=True, append_images=frames[1:], loop=0)
                
                if mask_sum > 9000000:
                    bc_data = {
                    'max_acxel': acxel_value,
                    'bf_state': state,
                    'E': E,
                    'Poisson': Poisson,
                    'yield_stress': yield_stress,
                    'env_name': args.env_name,
                    'height': height,
                    'window': window,
                    'steps': steps,
                    'scale': scale
                }
                    now = datetime.datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    with open(f'{experts_output_dir}/expert_acxel{acxel_value:.3f}_E{E:.2f},Poisson{Poisson:.2f},yield_stress{yield_stress:.2f}_{current_time}.pickle', 'wb') as f:
                        pickle.dump(bc_data, f)
            
            max_index = np.argmax(mask_sum_list)
            max_acxel = acxel_list[max_index]
            
            bc_data = {
                   'max_acxel': max_acxel,
                   'bf_state': state,
                   'E': E,
                   'Poisson': Poisson,
                   'yield_stress': yield_stress,
                   'env_name': args.env_name,
                   'height': height,
                   'window': window,
                   'steps': steps,
                   'scale': scale
               }
            now = datetime.datetime.now()
            current_time = now.strftime("%H:%M:%S")
            with open(f'{experts_output_dir}/expert_acxel{max_acxel:.3f}_E{E:.2f},Poisson{Poisson:.2f},yield_stress{yield_stress:.2f}_{current_time}.pickle', 'wb') as f:
                pickle.dump(bc_data, f)
            continue

        elif args.task_name in ['Pinch']:
           T = 5
           action_value = np.random.uniform(0.01, 0.02)
           action = np.concatenate([np.array([[action_value, 0, 0]]*T), np.array([[0, 0, 0]]*50)])
          
           initial_state = env.taichi_env.simulator.get_x(0)
           rope_bottom_index = np.argmin(initial_state[:, 1])

           try:
               best_max_y = 0
               best_max_x = 0
               env.taichi_env.primitives.set_softness()
               frames = []
               for idx, act in enumerate(action):
                   env.step(act)
                   # if idx % 1 == 0:
                   #     img = env.render(mode='rgb_array')
                   #     pimg = Image.fromarray(img)
                   #     I1 = ImageDraw.Draw(pimg)
                   #     I1.text((5, 5), f'E{E:.2f},Poisson{Poisson:.2f},yield_stress{yield_stress:.2f}', fill=(255, 0, 0))
                   #     frames.append(pimg)
                   state = env.taichi_env.simulator.get_x(0)
                   max_x = state[rope_bottom_index][0]
                   max_y = state[rope_bottom_index][1]
                   if max_y > best_max_y:
                       best_max_y = max_y
                       best_max_x = max_x
               # frames[0].save(f'{output_path}/action{action_value:.3f}_best_x{best_max_x:.2f}_best_y{best_max_y:.2f}_E{E:.2f},Poisson{Poisson:.2f},yield_stress{yield_stress:.2f}_.gif', save_all=True, append_images=frames[1:], loop=0)

               bc_data = {
                   'max_x': best_max_x,
                   'max_y': best_max_y,
                   'action': action,
                   'E': E,
                   'Poisson': Poisson,
                   'yield_stress': yield_stress,
                   'env_name': args.env_name,
               }
               now = datetime.datetime.now()
               current_time = now.strftime("%H:%M:%S")
               with open(f'{experts_output_dir}/expert_action{action_value:.3f}_best_x{best_max_x:.2f}_best_y{best_max_y:.2f}_{current_time}.pickle', 'wb') as f:
                   pickle.dump(bc_data, f)
           except:
               print('Nan error', E, Poisson, yield_stress)
               pass
           continue
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

            env.taichi_env.primitives.set_softness()
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