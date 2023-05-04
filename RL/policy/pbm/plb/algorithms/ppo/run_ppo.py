import copy
import glob
import os
import time
from collections import deque

import cv2
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from plb.algorithms.ppo.ppo import algo, utils
from plb.algorithms.ppo.ppo.algo import gail
from plb.algorithms.ppo.ppo.arguments import get_args
from plb.algorithms.ppo.ppo.envs import make_vec_envs
from plb.algorithms.ppo.ppo.model import Policy
from plb.algorithms.ppo.ppo.storage import RolloutStorage
from plb.algorithms.ppo.evaluation import evaluate

PARAMETER_SIXE = 3


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

def train_ppo(env, path, logger, old_args):
    num_steps = old_args.num_steps
    args = get_args()
    args.num_steps = env._max_episode_steps
    args.num_env_steps = num_steps

    log_dir = args.log_dir = path
    args.save_dir = path

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    eval_log_dir = log_dir

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    from plb import envs
    envs = make_vec_envs(env, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    if not old_args.para:
        obs_size = envs.observation_space.shape[0] - PARAMETER_SIXE
        actor_critic = Policy(
            (obs_size,),
            envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy})
    else:
        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo2':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo2_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))

        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    episodes = 0
    episodes_step = 0
    print('start training')

    ep_reward = 0

    total_steps = 0
    logger.reset()
    for j in range(num_updates):
        # set randam parameter: mu, lam, yield_stress
        np.random.seed(int(j))
        mu = np.random.uniform(10, 500)
        lam = np.random.uniform(10, 500)
        yield_stress = np.random.uniform(10, 500)
        parameters = torch.tensor([mu, lam, yield_stress])[None, :]
        print('parameter', mu, lam, yield_stress)
        env.taichi_env.set_parameter(mu, lam, yield_stress)
        envs = make_vec_envs(env, args.seed, args.num_processes,
                        args.gamma, args.log_dir, device, False)

        obs = envs.reset()
        if not old_args.para:
            obs = obs[:, :-PARAMETER_SIXE]
            obs_size = envs.observation_space.shape[0] - PARAMETER_SIXE
            rollouts = RolloutStorage(args.num_steps, args.num_processes,
                            (obs_size,), envs.action_space,
                            actor_critic.recurrent_hidden_state_size)
        else:
            rollouts = RolloutStorage(args.num_steps, args.num_processes,
                            envs.observation_space.shape, envs.action_space,
                            actor_critic.recurrent_hidden_state_size)
        rollouts.obs[0].copy_(obs)
        rollouts.to(device)

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        bf_length = 0.2
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            if not old_args.para:
                obs = obs[:, :-PARAMETER_SIXE]
            total_steps += 1
            diff_length = infos[0]['reward'] - bf_length
            bf_length = infos[0]['reward']
            ep_reward += diff_length

            if done[0]:
                img = envs.render(mode='rgb_array')
                if tell_rope_break(img):
                     ep_reward -= 1
                episodes += 1
                episode_rewards.append(ep_reward)
                logger.reset()
                logger.step(None, None, ep_reward, None, done[0], infos[0])
                logger.reset()
                reward = torch.tensor([[ep_reward]])
                #output = f"Episode: {episodes}, step: {step} reward: {ep_reward},  iou: {ep_iou},  last_iou: {ep_last_iou}"
                #print(output)
                #logger.write(output+'\n')
                #logger.flush()
                ep_reward = 0
                if episodes % 200 == 0:
                    # if episodes >= (test_times + 1) * 200:
                    ob_rms = utils.get_vec_normalize(envs).ob_rms
                    total_reward, total_iou, total_last_iou = evaluate(actor_critic, ob_rms, envs, args.seed,
                             args.num_processes, eval_log_dir, device)
                    output = f"Test Episode: {episodes}, step: {step} reward: {total_reward},  iou: {total_iou},  last_iou: {total_last_iou}"
                    print(output)
                episodes_step = 0
            else:
                episodes_step += 1
                logger.step(None, None, ep_reward, None, done[0], infos[0])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        print('interval', j)
        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
            or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                    .format(j-1, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), dist_entropy, value_loss,
                            action_loss))
