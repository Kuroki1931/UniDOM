import cv2
import time
import numpy as np
from scipy.spatial.transform import Rotation
import os


from cv2 import aruco

class Cloth:
    def __init__(self, create_grid_mass=None, task_ver=None, picture_name=None, read_pic=None):
        self.create_grid_mass = create_grid_mass
        self.task_ver = task_ver
        if self.create_grid_mass:
            self.frame = cv2.imread(f'/root/roomba_hack/pbm/tasks/multi_bc/imgs/task1/{self.task_ver}/goal.jpg')
        else:
            self.frame = cv2.imread(f'/root/roomba_hack/pbm/tasks/multi_bc/imgs/task1/{self.task_ver}/{picture_name}.jpg')

        dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters_create()
        gray = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dict_aruco, parameters=parameters)
        # corner
        cornerUL = np.array([0., 0.])
        self.origin  = np.array([-corners[np.where(ids == 0)[0][0]][0][0][1], corners[np.where(ids == 0)[0][0]][0][0][0]])
        cornerUR = np.array([-corners[np.where(ids == 1)[0][0]][0][1][1], corners[np.where(ids == 1)[0][0]][0][1][0]]) - self.origin
        cornerBR = np.array([-corners[np.where(ids == 2)[0][0]][0][2][1], corners[np.where(ids == 2)[0][0]][0][2][0]]) - self.origin
        cornerBL = np.array([-corners[np.where(ids == 3)[0][0]][0][3][1], corners[np.where(ids == 3)[0][0]][0][3][0]]) - self.origin

        ## convert
        self.sim_unit_aruco2sim = 1 / abs(cornerUL[1] - cornerBL[1])
        
    def hand_position(self):
        
        return None

    def get_obj_particle(self, hight=0.02):
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        lower = np.array([40, 35, 126])
        upper = np.array([98, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(self.frame, self.frame, mask=mask)

        h, s, v1 = cv2.split(result)
        erosion = cv2.erode(v1, np.ones((2,2),np.uint8), iterations = 1)
        closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))
        binary = np.where(closing>0, 1, closing)

        base_p_list = []
        for i in range(binary.shape[0]):
            for j in range(binary.shape[1]):
                if binary[int(i)][int(j)]==1:
                    y = (-i - self.origin[0]) * self.sim_unit_aruco2sim
                    x = (j - self.origin[1]) * self.sim_unit_aruco2sim
                    base_p_list.append([y, x])
        base_p = np.array(base_p_list)

        def rand_ints_nodup(a, b, k):
            ns = []
            while len(ns) < k:
                n = np.random.randint(a, b)
                if not n in ns:
                    ns.append(n)
            return ns

        if base_p.shape[0] > 5000:
            random_index = rand_ints_nodup(1, base_p.shape[0], 5000)
            base_p = base_p[random_index, :]
      
        add_p_num = 10000 - len(base_p)
        each_p_add_num = add_p_num // len(base_p) + 2
        added_p_list = []
        for p in base_p:
            added_p_list.append(p)
            random_p = np.random.uniform(low=-0.003, high=0.003, size=(each_p_add_num, 2)) + p
            for j in range(each_p_add_num):
                added_p_list.append(random_p[j])
        added_p = np.array(added_p_list)

        p_hight = (np.random.random((len(added_p), 1)) * 2 - 1) * 0.5 * hight + hight/2
        particle = np.insert(added_p, [1], p_hight, axis=1)
        if self.create_grid_mass:
            np.save(f'/root/roomba_hack/pbm/tasks/multi_bc/imgs/task1/{self.task_ver}/goal_state.npy', particle)
        return particle

    
    
    
    
    




