import os
import sys
import random
import datetime

import glob
import pickle
import json

sys.path.insert(0, './')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.append('/root/fairseq/examples/MMPT')

import numpy as np
import torch
import argparse
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from clip import tokenize

from util.clip import build_model, load_clip
from util import tf_utils
from util.preprocess import sample_pc
from util.lang_goal import LANG_GOAL
from videoclip import pooled_text


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--num_plasticine_point', type=int, default=3072, help='Point Number of Plasticine')
    parser.add_argument('--num_primitive_point', type=int, default=1024, help='Point Number of Primitive')
    parser.add_argument('--num_expert', type=int, default=5, help='sample time for each expert')
    parser.add_argument('--sample_time', type=int, default=10, help='sample time for each expert')
    parser.add_argument('--clip_type', type=int, default=1, help='0 -> video clip, 1 -> clip')
    parser.add_argument('--command_type', type=int, default=0, help='0 ->whole, 1 -> separate')
    parser.add_argument('--command_num', type=int, default=4, help='command num')
    
    return parser.parse_args()


def create_example(points, vector_encode, lang, action):

    feature = {
		'points' : tf_utils.float_list_feature(points.reshape(-1, 1)),
        'vector_encode': tf_utils.float_list_feature(vector_encode.reshape(-1, 1)),
		'lang' : tf_utils.float_list_feature(lang.reshape(-1, 1)),
		'action' : tf_utils.float_list_feature(action.reshape(-1, 1))
	}

    return tf.train.Example(features=tf.train.Features(feature=feature))


def main(args):
    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./data/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(timestr)
    exp_dir.mkdir(exist_ok=True)
    
    '''CLIP'''
    model, _ = load_clip("RN50", jit=False)
    clip_rn50 = build_model(model.state_dict())
    clip_rn50 = clip_rn50.float()
    clip_rn50.eval()
    def clip_emb(lang):
        tokens = tokenize([lang]).numpy()
        token_tensor = torch.from_numpy(tokens)
        lang_goal_emb, lang_token_embs = clip_rn50.encode_text_with_embeddings(token_tensor)
        return lang_goal_emb[0].to('cpu').detach().numpy().copy()

    '''DATA LOADING'''
    file_list = []
    env_count = {}
    env_lang_list = []
    files = glob.glob('/root/dmlc/policy/pbm/experts/*/expert*.pickle')
    with tf.io.TFRecordWriter(f'{exp_dir}/train_experts.tfrecord') as train_writer, tf.io.TFRecordWriter(f'{exp_dir}/validation_experts.tfrecord') as validation_writer:
        for path in files:
            if 'Rollingpin-v5' in path:
                continue
            with open(path, 'rb') as f: 
                data = pickle.load(f)           
            env_name = data['env_name']
            # data.env_name = env_name
            # with open(path, 'wb') as f:
            #     pickle.dump(bc_data, f)

            if env_name not in env_count:
                env_count[env_name] = 0

            if env_count[env_name] < args.num_expert:
                print(path)
                env_count[env_name] += 1

                action = data['action']
                plasticine_pc = data['plasticine_pc']
                primitive_pc = data['primitive_pc']
                pc_encode = np.zeros((args.num_plasticine_point + args.num_primitive_point, 2))
                pc_encode[:args.num_plasticine_point, 0] = 1
                pc_encode[args.num_plasticine_point:, 1] = 1

                for sample_i in range(args.sample_time):
                    # index = sample_i % args.command_num
                    index = 0
                    command = LANG_GOAL[env_name][index]['command']
                    env_lang = env_name + '&&' + str(index) + '&&' + command

                    if args.clip_type == 0:
                        lang_goal_emb = pooled_text(command)
                        env_lang = env_lang + '&&' + '' + '&&' + '' + '&&' + ''
                    else:
                        if args.command_type:
                            command_object = command = LANG_GOAL[env_name][index]['object']
                            object_lang_goal_emb = clip_emb(command_object)
                            command_manipulation = command = LANG_GOAL[env_name][index]['manipulation']
                            manipulation_lang_goal_emb = clip_emb(command_manipulation)
                            command_location = command = LANG_GOAL[env_name][index]['location']
                            location_lang_goal_emb = clip_emb(command_location)
                            lang_goal_emb = np.concatenate([object_lang_goal_emb, manipulation_lang_goal_emb, location_lang_goal_emb])
                            env_lang = env_lang + '&&' + command_object + '&&' + command_manipulation + '&&' + command_location
                        else:
                            lang_goal_emb = clip_emb(command)
                            env_lang = env_lang + '&&' + '' + '&&' + '' + '&&' + ''
                    
                    file_list.append(path + '&&' + env_lang)
                    env_lang_list.append(env_lang)

                    for i in range(action.shape[0]):
                        plasticine_pc_i = plasticine_pc[i]
                        primitive_pc_i = primitive_pc[i]
                        if len(primitive_pc_i) == 0:
                            primitive_pc_i = primitive_pc[i+1]
                        primitive_center_i = np.mean(primitive_pc_i, axis=0)
                        
                        points = sample_pc(plasticine_pc_i, primitive_pc_i, args.num_plasticine_point, args.num_primitive_point)
                        vector = points - primitive_center_i
                        vector_encode = np.hstack([vector, pc_encode])
                        act = action[i]

                        if 'Rollingpin' in env_name:
                            act = act[[0, 2, 1]]
                            act[2] = 0
                            act[0] *= -1
                        tf_example = create_example(points, vector_encode, lang_goal_emb, act)
                        if random.randint(1, 10) == 1:
                            validation_writer.write(tf_example.SerializeToString())
                        else:
                            train_writer.write(tf_example.SerializeToString())
    unique_env_lang = set(env_lang_list)
        
    with open(f'{exp_dir}/file.txt', 'w') as fp:
        for item in file_list:
            # write each item on a new line
            fp.write("%s\n" % item)
    with open(f'{exp_dir}/env_count.txt', mode="w") as f:
        json.dump(env_count, f, indent=4)
    with open(f'{exp_dir}/unique_env_lang.pickle', mode="wb") as f:
        pickle.dump(unique_env_lang, f)
    with open(f'{exp_dir}/args.txt', mode="w") as f:
        json.dump(args.__dict__, f, indent=4)


if __name__ == '__main__':
    args = parse_args()
    main(args)