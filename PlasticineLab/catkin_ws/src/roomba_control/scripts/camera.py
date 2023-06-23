#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
import pickle
from cv_bridge import CvBridge

from std_msgs.msg import Int8
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

import argparse
import random
import numpy as np
import os
import torch
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from plb.envs import make
from plb.algorithms.logger import Logger

from plb.algorithms.discor.run_sac import train as train_sac
from plb.algorithms.ppo.run_ppo import train_ppo
from plb.algorithms.TD3.run_td3 import train_td3
from plb.optimizer.solver import solve_action
from plb.optimizer.solver_nn import solve_nn

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
now = datetime.datetime.now()

class TeleopCommand:
    CARIBRATION = 1
    START = 2
    STOP = 3
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
    parser.add_argument("--env_name", type=str, default="Rollingpin-v1")
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
    def __init__(self, env, im):
        rospy.Subscriber('/camera/color/image_raw', Image, self.rgb_callback)
        rospy.Subscriber(
            '/teleop/command_c1', Int8, self.command_c1_cb)
        rospy.Subscriber(
            '/teleop/command_c2', Int8, self.command_c2_cb)
        rospy.Subscriber(
            '/controller/vel_c1', Twist, self.vel_c1_cb)
        rospy.Subscriber(
            '/controller/vel_c2', Twist, self.vel_c2_cb)
        self.pub_rgb = rospy.Publisher('pub_rgb_image', Image)
        self.rgb_image = None
        self.running = False
        self.bridge = CvBridge()

        self.env = env
        self.im = im
        self.action_list = []

    def rgb_callback(self, msg):
        if self.running:
            act = self.vel_c1
            self.action_list.append(act)

            self.env.step(act)
            img = self.env.render(mode='rgb_array')
            self.im.set_data(img)
            plt.pause(0.005)

            print(len(self.action_list))
            np.save('/root/dmlc/policy/pbm/tele/action.npy', np.array(self.action_list))
        
    def command_c1_cb(self, msg):
        if msg.data == TeleopCommand.START:
            self.running = True
        elif msg.data == TeleopCommand.STOP:
            self.running = False

    def command_c2_cb(self, msg):
        if msg.data == TeleopCommand.START:
            self.running = True
        elif msg.data == TeleopCommand.STOP:
            self.running = False

    def vel_c1_cb(self, msg):
        self.vel_c1 = np.array([msg.linear.x, msg.linear.y, msg.linear.z])

    def vel_c2_cb(self, msg):
        self.vel_c2 = np.array([msg.linear.x, msg.linear.y, msg.linear.z])

if __name__ == '__main__':
    rospy.init_node('special_node', log_level=rospy.DEBUG)

    args = get_args()
    set_random_seed(args.seed)
    env = make(args.env_name, nn=(args.algo=='nn'), sdf_loss=args.sdf_loss,
                            density_loss=args.density_loss, contact_loss=args.contact_loss,
                            soft_contact_loss=args.soft_contact_loss)
    env.seed(args.seed)
    env.reset()
    
    dummy_act = np.array([0, 0, 0])
    env.step(dummy_act)
    dummy_img = env.render(mode='rgb_array')
    
    im = plt.imshow(dummy_img)
    
    cam = Camera(env, im)

    while not rospy.is_shutdown():

        
        rospy.sleep(1)