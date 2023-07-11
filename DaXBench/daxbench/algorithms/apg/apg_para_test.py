import argparse
import os
import time
import pickle
from typing import Optional

import flax
import imageio
import jax
import jax.numpy as jnp
import optax
import numpy as np
from absl import logging
from brax import envs
from brax.training import distribution, pmap
from brax.training import networks
from brax.training import normalization
from brax.training.types import PRNGKey
from brax.training.types import Params
from flax import linen
from tensorboardX import SummaryWriter
from daxbench.core.envs.basic.mpm_env import MPMEnv
from daxbench.core.envs.shape_rope_env import ShapeRopeEnv
from daxbench.core.envs.registration import env_functions

logging.set_verbosity(logging.INFO)
best_reward = 0


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    key: PRNGKey
    normalizer_params: Params
    optimizer_state: optax.OptState
    policy_params: Params


def train(
    environment_fn,
    episode_length: int,
    action_repeat: int = 1,
    num_envs: int = 1,
    num_eval_envs: int = 20,
    max_gradient_norm: float = 1e9,
    learning_rate=1e-4,
    normalize_observations=False,
    seed=0,
    log_frequency=10,
    truncation_length: Optional[int] = None,
):
    xt = time.time()
    # args.logdir = (
    #     f"logs/apg_para/{args.env}/{args.env}_ep_len{args.ep_len}_num_envs{args.num_envs}_lr{args.lr}"
    #     f"_max_it{args.max_it}_max_grad_norm{args.max_grad_norm}/seed{args.seed}"
    # )
    # writer = SummaryWriter(args.logdir)

    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count

    local_devices_to_use = min(local_devices_to_use, args.gpus)
    logging.info(
        "Device count: %d, process count: %d (id %d), local device count: %d, "
        "devices to be used count: %d",
        jax.device_count(),
        process_count,
        process_id,
        local_device_count,
        local_devices_to_use,
    )
    logging.info("Available devices %s", jax.devices())

    # seeds
    key = jax.random.PRNGKey(seed)
    key, key_models, key_env = jax.random.split(key, 3)
    key_env = jax.random.split(key_env, process_count)[process_id]
    key = jax.random.split(key, process_count)[process_id]
    key_eval = jax.random.PRNGKey(seed + 666)

    # envs
    core_env = environment_fn(
        batch_size=num_envs // local_devices_to_use, seed=seed, aux_reward=True
    )

    step_fn = core_env.step_diff
    reset_fn = core_env.reset

    if isinstance(core_env, MPMEnv) and not isinstance(core_env, ShapeRopeEnv):
        auto_reset = jax.pmap(core_env.auto_reset)

    eval_env = environment_fn(batch_size=num_eval_envs, seed=seed + 666)
    eval_step_fn = eval_env.step_diff
    eval_reset_fn = eval_env.reset

    # initialize policy
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=core_env.action_size
    )
    policy_model = make_direct_optimization_model(
        parametric_action_distribution, core_env.observation_size
    )

    # init optimizer
    params_path = '/root/DaXBench/logs/apg_para/fold_cloth1_para/fold_cloth1_para_ep_len3_num_envs2_lr0.0001_max_it2000_max_grad_norm0.3/seed0/apg_fold_cloth1_para_2000_890.340257286776.pkl'
    with open(params_path, "rb") as f:
        policy_params = pickle.load(f)
    args.lodir = '/'.join(params_path.split('/')[:-1])
    policy_params = policy_model.init(key_models)
    optimizer = optax.adam(learning_rate=learning_rate)
    optimizer_state = optimizer.init(policy_params)

    # state normalizer
    (
        normalizer_params,
        normalizer_update_fn,
        normalizer_apply_fn,
    ) = normalization.create_observation_normalizer(
        core_env.observation_size,
        normalize_observations=True,
        num_leading_batch_dims=2,
        pmap_to_devices=local_devices_to_use,
    )

    """
    Evaluation functions
    """
    # prepare training
   
    t = time.time()
    test_reward_dict = {}
    for it in range(args.max_it + 1):
        # radomize parameters
        np.random.seed(it)
        core_env_stiffness = np.random.uniform(100, 1500)
        eval_env_stiffness = np.random.uniform(100, 1500)
        
        # recreate env function
        core_env = environment_fn(
            batch_size=num_envs // local_devices_to_use, seed=seed, aux_reward=True, stiffness=core_env_stiffness
        )
        step_fn = core_env.step_diff
        reset_fn = core_env.reset
        if isinstance(core_env, MPMEnv) and not isinstance(core_env, ShapeRopeEnv):
            auto_reset = jax.pmap(core_env.auto_reset)
        eval_env = environment_fn(batch_size=num_eval_envs, seed=seed + 666, stiffness=eval_env_stiffness)
        eval_step_fn = eval_env.step_diff
        eval_reset_fn = eval_env.reset

        # reset function
        _, first_state = reset_fn(key_env)
        if not isinstance(core_env, MPMEnv) or isinstance(core_env, ShapeRopeEnv):
            reset_fn = jax.vmap(reset_fn)
        first_state = jax.tree_util.tree_map(
            lambda x: jnp.stack([x] * local_devices_to_use), first_state
        )
        _, eval_first_state = eval_reset_fn(key_eval)

        if not isinstance(core_env, MPMEnv) or isinstance(core_env, ShapeRopeEnv):
            key_debug, key_eval = jax.random.split(key_eval)
            key_envs = jax.random.split(key_env, local_devices_to_use)
            _, train_first_state = reset_fn(key_envs)
        else:
            key_env = jax.random.split(key_env, 1)[0]
            key_envs = jax.random.split(key_env, args.num_envs)
            key_envs = key_envs.reshape(
                (local_devices_to_use, args.num_envs // local_devices_to_use, -1)
            )
            train_first_state = auto_reset(first_state, first_state, key_envs)
        
        # first state parameter modify
        

        actor_lr = (1e-5 - args.lr) * float(it / args.max_it) + args.lr
        optimizer = optax.adam(learning_rate=actor_lr)
        print("actor_lr: ", actor_lr)

        logging.info(
            "starting iteration %s %s %s", it, time.time() - xt, time.time() - t
        )
        t = time.time()

        if it % args.eval_freq == 0:
            _, reward_list = eval_policy(
                it, training_state, eval_first_state, key_debug
            )
            test_reward = jnp.mean(reward_list.sum(0))
            test_reward_dict[it] = [test_reward._value.max(), eval_env.simulator.stiffness]
            logging.info("Test reward %s", test_reward)
            writer.add_scalar("test_reward", test_reward, it)
            writer.add_scalar("last_reward", reward_list[-1].mean(), it)
            file_to_save = open(f"{args.logdir}/apg_{args.env}_{it}_{eval_env.simulator.stiffness}.pkl", "wb")
            single_param = jax.tree_util.tree_map(
                lambda x: x[0], training_state.policy_params
            )
            pickle.dump(single_param, file_to_save)
            file_to_save.close()
        # optimization
        t = time.time()
        training_state, metrics, state_list, action_list, synchro = minimize(
            training_state, train_first_state
        )

        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
        logging.info("Training reward %s", jnp.mean(metrics["reward"].sum(0)))
        writer.add_scalar("grad_norm", metrics["grad_norm"].mean(), it)
        sps = (episode_length * num_envs) / (time.time() - t)
        training_walltime += time.time() - t

    params = jax.tree_util.tree_map(lambda x: x[0], training_state.policy_params)
    normalizer_params = jax.tree_util.tree_map(
        lambda x: x[0], training_state.normalizer_params
    )
    params = normalizer_params, params
    inference = make_inference_fn(
        core_env.observation_size, core_env.action_size, normalize_observations
    )
    
    with open(f"{args.logdir}/apg_reward_dict.pkl", "wb") as f:
        pickle.dump(test_reward_dict, f)

def make_direct_optimization_model(parametric_action_distribution, obs_size):
    return networks.make_model(
        [512, 256, parametric_action_distribution.param_size],
        obs_size,
        activation=linen.swish,
    )


def make_inference_fn(observation_size, action_size, normalize_observations):
    """Creates params and inference function for the direct optimization agent."""
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    _, obs_normalizer_apply_fn = normalization.make_data_and_apply_fn(
        observation_size, normalize_observations
    )
    policy_model = make_direct_optimization_model(
        parametric_action_distribution, observation_size
    )

    def inference_fn(params, obs, key):
        normalizer_params, params = params
        obs = obs_normalizer_apply_fn(normalizer_params, obs)
        action = parametric_action_distribution.sample(
            policy_model.apply(params, obs), key
        )
        return action

    return inference_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env",
        default="fold_cloth1",
        help="name of the environment (default: %(default)s)",
    )
    parser.add_argument(
        "--ep_len",
        default=10,
        type=int,
        help="length of each episode (default: %(default)s)",
    )
    parser.add_argument(
        "--num_envs",
        default=4,
        type=int,
        help="number of environments used in training (default: %(default)s)",
    )
    parser.add_argument(
        "--lr", default=1e-4, type=float, help="learning rate (default: %(default)s)"
    )
    parser.add_argument(
        "--max_it",
        default=2000,
        type=int,
        help="maximum number of iterations (default: %(default)s)",
    )
    parser.add_argument(
        "--max_grad_norm",
        default=0.3,
        type=float,
        help="maximum norm to perform gradient clip (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", default=1, type=int, help="random seed (default: %(default)s)"
    )
    parser.add_argument(
        "--gpus", default=1, type=int, help="number of GPUs (default: %(default)s)"
    )
    parser.add_argument(
        "--eval_freq",
        default=20,
        type=int,
        help="number of iterations for each evaluation (default: %(default)s)",
    )

    args = parser.parse_args()

    train(
        environment_fn=env_functions[args.env],
        episode_length=args.ep_len,
        num_envs=args.num_envs,
        learning_rate=args.lr,
        normalize_observations=True,
        log_frequency=args.max_it,
        max_gradient_norm=args.max_grad_norm,
        seed=args.seed,
    )
