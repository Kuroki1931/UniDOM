#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
import pickle
from cv_bridge import CvBridge

from std_msgs.msg import Int8, Float32
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

import argparse
import random
import numpy as np
import os
import torch
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from plb.envs import make
from plb.algorithms.logger import Logger

from plb.algorithms.discor.run_sac import train as train_sac
from plb.algorithms.ppo.run_ppo import train_ppo
from plb.algorithms.TD3.run_td3 import train_td3
from plb.optimizer.solver import solve_action
from plb.optimizer.solver_nn import solve_nn
from util.lang_goal import ACTION

os.environ['TI_USE_UNIFIED_MEMORY'] = '0'
os.environ['TI_DEVICE_MEMORY_FRACTION'] = '0.9'
os.environ['TI_DEVICE_MEMORY_GB'] = '4'
os.environ['TI_ENABLE_CUDA'] = '0'
os.environ['TI_ENABLE_OPENGL'] = '0'

RL_ALGOS = ['sac', 'td3', 'ppo']
DIFF_ALGOS = ['action', 'nn']

import datetime, os, cv2
import matplotlib.pyplot as plt
from PIL import Image

class TeleopCommand:
    START = 1
    STOP = 2
    REMOVE = 3
    ERROR = 4


class LoggerCommand:
    STOP = 0
    START = 1
    ERROR = 2


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default='action')
    parser.add_argument("--env_name", type=str, default="Move-v1")
    parser.add_argument("--path", type=str, default='./output')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sdf_loss", type=float, default=500)
    parser.add_argument("--density_loss", type=float, default=500)
    parser.add_argument("--contact_loss", type=float, default=1)
    parser.add_argument("--soft_contact_loss", action='store_true')

    parser.add_argument("--num_steps", type=int, default=None)

    # differentiable physics parameters
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--softness", type=float, default=6666.)
    parser.add_argument("--optim", type=str, default='Adam', choices=['Adam', 'Momentum'])
    parser.add_argument("--create_grid_mass", action='store_true')

    args=parser.parse_args()

    return args

class Camera():
    def __init__(self, env, im, name):
        rospy.Subscriber(
            '/teleop/command_c1', Int8, self.command_c1_cb)
        rospy.Subscriber(
            '/teleop/command_c2', Int8, self.command_c2_cb)
        rospy.Subscriber(
            '/controller/vel_c1', Twist, self.vel_c1_cb)
        rospy.Subscriber(
            '/controller/vel_c2', Twist, self.vel_c2_cb)
        rospy.Subscriber(
            '/controller/trigger_c1', Float32, self.trigger_c1_cb)
        # self.pub_rgb = rospy.Publisher('pub_rgb_image', Image)
        self.rgb_image = None
        self.running = False
        self.reset = False
        self.remove = False
        self.bridge = CvBridge()

        self.env = env
        self.im = im
        self.action_list = []
        self.scale_x = 6
        self.scale_y = 6
        self.scale_z = 6

        self.name = name

        output_dir = f"../experts/{args.env_name}/action"
        os.makedirs(output_dir, exist_ok=True)

        while not rospy.is_shutdown():
            if self.running:
                pos_act = self.vel_c1
                pos_act = np.clip(pos_act,-1, 1)

                env_name = args.env_name.split('-')[0]
                order = ACTION[env_name]['order']
                direction = ACTION[env_name]['direction']
                pos_act = pos_act[order]
                pos_act = pos_act * direction
                print(pos_act)

                if self.trigger_c1 > 0.9:
                    gripper_act = 1
                elif self.trigger_c1 > 0:
                    gripper_act = -1
                else:
                    gripper_act = 0
                act = np.append(pos_act, gripper_act)
                print(act)

                if 'Chopsticks' in self.name:
                    self.action_list.append(act)
                else:
                    act[-1] = 0
                    self.action_list.append(act)

                if 'Chopsticks' in self.name:
                    act = np.insert(act, 3, [0]*3)
                    print(f"chopsticks: {act}")
                else:
                    act = act[:3]

                self.env.step(act)
                self.render_env()

                print(len(self.action_list))
            if self.remove:
                self.action_list = []
                self.env.reset()
                self.running = False
                self.reset = False
                self.remove = False
                self.render_env()
                print("### Teleop Stop ###")

            if self.reset:
                now = datetime.datetime.now()
                current_time = now.strftime("%H:%M:%S")
                np.save(f'{output_dir}/{current_time}.npy', np.array(self.action_list))
                self.action_list = []
                self.env.reset()
                self.running = False
                self.reset = False
                self.remove = False
                self.render_env()
                print("### Teleop Stop ###")
    
    def render_env(self):
        img = self.env.render(mode='rgb_array')
        self.im.set_data(img)
        plt.pause(0.005)

    def command_c1_cb(self, msg):
        if (msg.data == TeleopCommand.START and self.running == False) or (msg.data == TeleopCommand.STOP and self.running == False):
            self.running = True
            print("### Teleop Start ###")
        elif (msg.data == TeleopCommand.STOP and self.running == True) or (msg.data == TeleopCommand.START and self.running == True):
            self.running = False
            print("### Teleop Stop ###")
        elif msg.data == TeleopCommand.REMOVE:
            self.remove = True
            print("### Remove ###")

    def command_c2_cb(self, msg):
        if msg.data == TeleopCommand.START:
            self.reset = True
            print("### Reset ###")
        elif msg.data == TeleopCommand.STOP:
            self.reset = True
            print("### Reset ###")
        elif msg.data == TeleopCommand.REMOVE:
            self.remove = True
            print("### Remove ###")

    def vel_c1_cb(self, msg):
        vel_x = - self.scale_x * msg.linear.x
        vel_y = self.scale_y * msg.linear.y
        vel_z = self.scale_z * msg.linear.z
        self.vel_c1 = np.array([vel_x, vel_y, vel_z])

    def vel_c2_cb(self, msg):
        vel_x = - self.scale_x * msg.linear.x
        vel_y = self.scale_y * msg.linear.y
        vel_z = self.scale_z * msg.linear.z
        self.vel_c2 = np.array([vel_x, vel_y, vel_z])

    def trigger_c1_cb(self, msg):
        self.trigger_c1 = np.array(msg.data)

if __name__ == '__main__':
    rospy.init_node('special_node', log_level=rospy.DEBUG)

    args = get_args()
    set_random_seed(args.seed)
    env = make(args.env_name, nn=(args.algo=='nn'), sdf_loss=args.sdf_loss,
                            density_loss=args.density_loss, contact_loss=args.contact_loss,
                            soft_contact_loss=args.soft_contact_loss)
    env.seed(args.seed)
    env.reset()

    name = args.env_name.split('-')[0]
    
    if 'Chopsticks' in name:
        dummy_act = np.array([0]*7)
    else:
        dummy_act = np.array([0]*3)

    env.step(dummy_act)
    dummy_img = env.render(mode='rgb_array')
    im = plt.imshow(dummy_img)
    
    cam = Camera(env, im, name)

    while not rospy.is_shutdown():

        
        rospy.sleep(1)