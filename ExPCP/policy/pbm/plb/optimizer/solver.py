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
        return best_action


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
    T = env._max_episode_steps
    solver = Solver(taichi_env, logger, None,
                    n_iters=(args.num_steps + T-1)//T, softness=args.softness, horizon=T,
                    **{"optim.lr": args.lr, "optim.type": args.optim, "init_range": 0.0001})
    action = solver.solve()
    np.save(f"{output_path}/action.npy", action)
    print(action)
    
    frames = []
    for idx, act in enumerate(action):
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
    frames[0].save(f'{output_path}/demo.gif', save_all=True, append_images=frames[1:], loop=0)

    # create dataset for bc
    env.reset()
    action_list = []
    plasticine_pc_list = []
    primitive_pc_list = []
    reward_list = []
    loss_info_list = []
    last_iou_list = []

    for i in range(len(action)):
        action_list.append(action[i])

        plasticine_pc, primtiive_pc = env.get_obs(0)
        plasticine_pc_list.append(plasticine_pc.tolist())
        primitive_pc_list.append(primtiive_pc.tolist())

        obs, r, done, loss_info = env.step(action[i])
        last_iou = loss_info['incremental_iou']
        reward_list.append(r)
        loss_info_list.append(loss_info)

    experts_output_dir = f'/root/ExPCP/policy/pbm/experts/{args.env_name}'
    if not os.path.exists(experts_output_dir):
        os.makedirs(experts_output_dir, exist_ok=True)

    print('length', i, 'r', r, 'last_iou', last_iou)
    bc_data = {
        'action': np.array(action_list),
        'rewards': np.array(reward_list),
        'env_name': args.env_name,
        'plasticine_pc': np.array(plasticine_pc_list),
        'primitive_pc': np.array(primitive_pc_list),
        'loss_info_list': loss_info_list
    }
    
    print(action.shape, np.array(reward_list).shape, np.array(plasticine_pc_list).shape, np.array(primitive_pc_list).shape)
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    with open(f'{experts_output_dir}/expert_{last_iou:.4f}_{current_time}.pickle', 'wb') as f:
        pickle.dump(bc_data, f)
    with open(f'{output_path}/iou_{last_iou}.txt', 'w') as f:
        f.write(str(last_iou))
