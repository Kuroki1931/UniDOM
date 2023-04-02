import os
import sys
import random
import datetime

sys.path.insert(0, './')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import torch
import pickle
import argparse
import logging
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from clip import tokenize

from tqdm import tqdm
from util.clip import build_model, load_clip
from models.cls_ssg_model import CLS_SSG_Model
from PIL import Image
from util.lang_goal import LANG_GOAL

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'pbm'))
sys.path.append('/root/fairseq/examples/MMPT')

from plb.envs import make
from plb.algorithms.logger import Logger
from plb.algorithms.discor.run_sac import train as train_sac
from plb.algorithms.ppo.run_ppo import train_ppo
from plb.algorithms.TD3.run_td3 import train_td3
from plb.optimizer.solver import solve_action
from plb.optimizer.solver_nn import solve_nn
from util.preprocess import sample_pc
# from videoclip import pooled_text

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size in training')
    parser.add_argument('--epoch', default=10000, type=int, help='number of epoch in training')
    parser.add_argument('--save_epoch', default=300, type=int, help='save epoch')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_plasticine_point', type=int, default=3072, help='Point Number of Plasticine')
    parser.add_argument('--num_primitive_point', type=int, default=1024, help='Point Number of Primitive')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--clip_type', type=int, default=1, help='0 -> video clip, 1 -> clip')
    parser.add_argument('--command_type', type=int, default=0, help='0 ->whole, 1 -> separate')
    parser.add_argument('--command_num', type=int, default=4, help='command num')
    
    parser.add_argument("--algo", type=str, default='action')
    parser.add_argument("--env_name", type=str, default="Rollingpin-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sdf_loss", type=float, default=500)
    parser.add_argument("--density_loss", type=float, default=500)
    parser.add_argument("--contact_loss", type=float, default=1)
    parser.add_argument("--soft_contact_loss", action='store_true')
    parser.add_argument("--num_steps", type=int, default=100)
    # differentiable physics parameters
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--softness", type=float, default=6666.)
    parser.add_argument("--optim", type=str, default='Adam', choices=['Adam', 'Momentum'])
    return parser.parse_args()

tf.random.set_seed(1234)
CHECK_POINT_PATH = '/root/dmlc/policy/log/2023-03-17_10-49/2023-03-17_10-54/model/weights.ckpt'
# CHECK_POINT_PATH = '/root/dmlc/policy/log/2023-03-17_10-40/2023-03-17_10-48/model_4800/weights.ckpt'


def test(args):
    '''LOG'''
    args = parse_args()
    
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experts_dir = CHECK_POINT_PATH.split('/')[-4]

    '''CLIP'''
    clip_model, _ = load_clip("RN50", jit=False)
    clip_rn50 = build_model(clip_model.state_dict())
    clip_rn50 = clip_rn50.float()
    clip_rn50.eval()
    def clip_emb(lang):
        tokens = tokenize([lang]).numpy()
        token_tensor = torch.from_numpy(tokens)
        lang_goal_emb, lang_token_embs = clip_rn50.encode_text_with_embeddings(token_tensor)
        return lang_goal_emb[0].to('cpu').detach().numpy().copy()
    
    action_size = 3
    num_point = args.num_primitive_point + args.num_plasticine_point

    model = CLS_SSG_Model(args.batch_size, action_size)
   
    model.build([(args.batch_size, num_point, 3), (args.batch_size, num_point, 5), (args.batch_size, 1024)])
    print(model.summary())
    model.compile(
		optimizer=keras.optimizers.Adam(args.lr, clipnorm=0.1),
		loss='mean_squared_error',
		metrics='mean_squared_error',
		weighted_metrics='mean_squared_error'
	)
    model.load_weights(CHECK_POINT_PATH).expect_partial()
    
    '''env'''
    set_random_seed(args.seed)
    env1 = make('Rollingpin-v1', nn=(args.algo=='nn'), sdf_loss=args.sdf_loss,
								density_loss=args.density_loss, contact_loss=args.contact_loss,
								soft_contact_loss=args.soft_contact_loss)
    env1.seed(args.seed)
    env2 = make('Rollingpin-v2', nn=(args.algo=='nn'), sdf_loss=args.sdf_loss,
								density_loss=args.density_loss, contact_loss=args.contact_loss,
								soft_contact_loss=args.soft_contact_loss)
    env2.seed(args.seed)
    env3 = make('Rollingpin-v3', nn=(args.algo=='nn'), sdf_loss=args.sdf_loss,
								density_loss=args.density_loss, contact_loss=args.contact_loss,
								soft_contact_loss=args.soft_contact_loss)
    env3.seed(args.seed)
    env4 = make('Rollingpin-v4', nn=(args.algo=='nn'), sdf_loss=args.sdf_loss,
								density_loss=args.density_loss, contact_loss=args.contact_loss,
								soft_contact_loss=args.soft_contact_loss)
    env4.seed(args.seed)
    env_dict = {'Rollingpin-v1': env1, 'Rollingpin-v2': env2, 'Rollingpin-v3': env3, 'Rollingpin-v4': env4}
    # env2 = make('Rope-v1', nn=(args.algo=='nn'), sdf_loss=args.sdf_loss,
	# 							density_loss=args.density_loss, contact_loss=args.contact_loss,
	# 							soft_contact_loss=args.soft_contact_loss)
    # env2.seed(args.seed)
    # env3 = make('Torus-v1', nn=(args.algo=='nn'), sdf_loss=args.sdf_loss,
	# 							density_loss=args.density_loss, contact_loss=args.contact_loss,
	# 							soft_contact_loss=args.soft_contact_loss)
    # env3.seed(args.seed)
    # env_dict = {'Rollingpin': env1, 'Rope': env2, 'Torus': env3}

    env_lang_list = []
    for env_name in LANG_GOAL.keys():
        for index in range(len(LANG_GOAL[env_name].keys())-1):
            env_lang = env_name + '&&' + str(index) + '&&' + '&&'.join(LANG_GOAL[env_name][index].values())
            print(env_lang)
            env_lang_list.append(env_lang)
    
    env_lang_list = [
        'Rollingpin-v3&&0&&Push the right side of the object&&object&&push&&right side',
        'Rollingpin-v3&&1&&Compress right flank of the soft body&&soft body&&compress&&right flank',
        'Rollingpin-v3&&2&&crush right side of the box&&box&&crush&&right side',
        'Rollingpin-v3&&3&&flatten right-hand side of the plasticine&&plasticine&&flatten&&right-hand side',
        'Rollingpin-v2&&0&&Apply force to left surface of the box&&box&&apply force&&left surface',
        'Rollingpin-v2&&1&&Flatten left portion of the box&&box&&flatten&&left portion',
        'Rollingpin-v2&&2&&Push left flank of the plasticine&&plasticine&&push&&left flank',
        'Rollingpin-v2&&3&&Crush the left side of the soft object&&soft object&&crush&&left side',

        # 'Rollingpin-v1&&0&&Push the entire section of the object&&object&&push&&entire section',
        # 'Rollingpin-v1&&1&&Crush the whole surface of the box&&box&&crush&&whole surface',
        # 'Rollingpin-v1&&2&&Flatten whole section of the plasticine&&plasticine&&flatten&&whole section',
        # 'Rollingpin-v1&&3&&Apply force to whole surface of the soft body&&soft body&&apply force&&whole surface',
        # 'Rollingpin-v2&&0&&Push the left side of the object&&object&&push&&left side',
        # 'Rollingpin-v2&&1&&Compress left flank of the soft bod&&soft body&&compress&&left flank',
        # 'Rollingpin-v2&&2&&crush left side of the box&&box&&crush&&left side',
        # 'Rollingpin-v2&&3&&flatten left-hand side of the plasiticne&&plasticine&&flatten&&left-hand side',
        # 'Rollingpin-v3&&0&&Apply force to right surface of the box&&box&&apply force&&right surface',
        # 'Rollingpin-v3&&1&&Flatten right portion of the box&&box&&flatten&&right portion',
        # 'Rollingpin-v3&&2&&Push right flank of the plasticine&&plasticine&&push&&right flank',
        # 'Rollingpin-v3&&3&&Crush the right side of the soft object&&soft object&&crush&&right side',
        # 'Rollingpin-v4&&0&&Apply pressure to the center of the plasticine&&plasticine&&apply pressure&&center',
        # 'Rollingpin-v4&&1&&push center surface of the box&&box&&push&&center surface',
        # 'Rollingpin-v4&&2&&Compress center area of the soft object&&soft object&&compress&&center area',
        # 'Rollingpin-v4&&3&&Crush the center of the object&&object&&crush&&center',
    ]

    row_count = 0
    for env_lang in tqdm(list(env_lang_list)):
        # test_env_name, test_lang = env_lang.split('&&')
        test_env_name, index, test_lang, test_object, test_manipulation, test_location = env_lang.split('&&')
        test_env = test_env_name.split('-')[0]
        # env = env_dict[test_env]
        env = env_dict[test_env_name]
        env.reset()
        
        if False and row_count > 0:
            env.taichi_env.initialize()
            env.taichi_env.simulator.reset(test_plasticine_pc)
            state = env.taichi_env.get_state()
            env.taichi_env.set_state(**state)

        num_steps = LANG_GOAL[test_env_name]['num_steps']
        output_dir = f"{'/'.join(CHECK_POINT_PATH.split('/')[:-1])}/evaluation/{timestr}/{test_env_name}/{index}_{test_lang}"
        os.makedirs(output_dir, exist_ok=True)
  
        if args.clip_type == 0:
            test_lang_goal_emb = pooled_text(test_lang)
        else:
            if args.command_type:
                object_lang_goal_emb = clip_emb(test_object)
                manipulation_lang_goal_emb = clip_emb(test_manipulation)
                location_lang_goal_emb = clip_emb(test_location)
                test_lang_goal_emb = np.concatenate([object_lang_goal_emb, manipulation_lang_goal_emb, location_lang_goal_emb])
            else:
                test_lang_goal_emb = clip_emb(test_lang)

        pc_encode = np.zeros((args.num_plasticine_point + args.num_primitive_point, 2))
        pc_encode[:args.num_plasticine_point, 0] = 1
        pc_encode[args.num_plasticine_point:, 1] = 1

        imgs = []
        for t in range(num_steps):
            print(t, '/', num_steps)

            test_plasticine_pc, test_primtiive_pc = env.get_obs(0, t)
            if test_primtiive_pc.shape[0] == 0 or test_plasticine_pc.shape[0] == 0:
                env.step(np.array([0, 0, 0])) # plasticinelab bug?
                continue

            test_points = sample_pc(test_plasticine_pc, test_primtiive_pc, args.num_plasticine_point, args.num_primitive_point)
            vector = test_points - np.mean(test_primtiive_pc, axis=0)
            vector_encode = np.hstack([vector, pc_encode])
            act = model.forward_pass([
                tf.cast(tf.convert_to_tensor(test_points[None]), tf.float32),
                tf.cast(tf.convert_to_tensor(vector_encode[None]), tf.float32),
                tf.cast(tf.convert_to_tensor(test_lang_goal_emb[None]), tf.float32)
            ], False, 1)
            act = act.numpy()[0]
            print(act)

            if 'Rollingpin' in test_env_name:
                act[0] *= -1
                act = act[[0, 2, 1]]
            _, _, _, loss_info = env.step(act)

            last_iou = loss_info['incremental_iou']
            
            if t % 5 == 0:
                print(f"Saving gif at {t} steps")
                imgs.append(Image.fromarray(env.render(mode='rgb_array')))
        
        imgs[0].save(f"{output_dir}/{last_iou:.4f}_{t}.gif", save_all=True, append_images=imgs[1:], loop=0)
        with open(f'{output_dir}/last_iou_{t}.txt', 'w') as f:
            f.write(str(last_iou))
        
        row_count += 1


if __name__ == '__main__':
	args = parse_args()
	test(args)
