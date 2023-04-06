import cv2
import time
import numpy as np
from scipy.spatial.transform import Rotation
import os


from cv2 import aruco

class Cloth:
    def __init__(self):
        self.obj_pcds = np.load('/root/real2sim/real2sim/points/test/object_pcd.npy')
        self.hand_position = np.load('/root/real2sim/real2sim/points/test/pritmive_position.npy')
        
    def get_hand_position(self):
        return [str(tuple(self.hand_position))]

    def get_obj_particle(self):
        return self.obj_pcds

    
    
    
    
    




