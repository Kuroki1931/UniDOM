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
    
    def get_surface_index(self):
        surface_index = []
        threshold = 0.01
        width = (0.3/2, 0.07/2, 0.1/2)
        init_pos = (0.5, 0.05, 0.5)
        for pcd in self.obj_pcds:
            if (init_pos[0] - width[0] + threshold >= pcd[0]) | (init_pos[0] + width[0] - threshold <= pcd[0]):
                surface_index.append(1)
            # don't use bottom side
            elif (init_pos[1] + width[1] - threshold <= pcd[1]):
                surface_index.append(1)
            elif (init_pos[2] - width[2] + threshold >= pcd[2]) | (init_pos[2] + width[2] - threshold <= pcd[2]):
                surface_index.append(1)
            else:
                surface_index.append(0)
        return surface_index

    
    
    
    
    




