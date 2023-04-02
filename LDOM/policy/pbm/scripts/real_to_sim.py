import cv2
from cv2 import aruco
import numpy as np
import time


area_hight = 195
roomba_hight = 8
roomba_radian = 17
box_hight = 12
box_length = 23
box_depth = 15

dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()

frame = cv2.imread('start.jpg')
gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
num_id = 0
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dict_aruco, parameters=parameters)


# 左上(原点)
cornerUL = np.array([0., 0.])
origin  = np.array([-corners[np.where(ids == 0)[0][0]][0][0][1], corners[np.where(ids == 0)[0][0]][0][0][0]])
# 右上
cornerUR = np.array([-corners[np.where(ids == 1)[0][0]][0][1][1], corners[np.where(ids == 1)[0][0]][0][1][0]]) - origin
# 右下
cornerBR = np.array([-corners[np.where(ids == 2)[0][0]][0][2][1], corners[np.where(ids == 2)[0][0]][0][2][0]]) - origin
# 左下
cornerBL = np.array([-corners[np.where(ids == 3)[0][0]][0][3][1], corners[np.where(ids == 3)[0][0]][0][3][0]]) - origin
# objectの左上
objectUL = np.array([-corners[np.where(ids == 4)[0][0]][0][0][1], corners[np.where(ids == 4)[0][0]][0][0][0]]) - origin
# ルンバ中心
roombaUL = np.array([-corners[np.where(ids == 5)[0][0]][0][0][1], corners[np.where(ids == 5)[0][0]][0][0][0]]) - origin
rommbaUR = np.array([-corners[np.where(ids == 5)[0][0]][0][1][1], corners[np.where(ids == 5)[0][0]][0][1][0]]) - origin
roombaC = (roombaUL + rommbaUR)/2


## start

sim_unit_cm2sim = 1 / area_hight
sim_unit_aruco2sim = 1 / abs(cornerUL[1] - cornerBL[1])
sim_roomba_init = roombaC * sim_unit_aruco2sim
sim_roomba_hight = roomba_hight * sim_unit_cm2sim
sim_roomba_radian = roomba_radian * sim_unit_cm2sim

sim_box_hight = box_hight * sim_unit_cm2sim
sim_box_length = box_length * sim_unit_cm2sim
sim_box_depth = box_depth * sim_unit_cm2sim
sim_box_init = objectUL*sim_unit_aruco2sim + np.array([sim_box_length/2, sim_box_depth/2])

output = {
    'roomba_init': sim_roomba_init,
    'roomba_hight': sim_roomba_hight,
    'roomba_radian': sim_roomba_radian,
    'box_init': sim_box_init,
    'box_length': sim_box_length,
    'box_hight': sim_box_hight,
    'box_depth': sim_box_depth,
}

f = open('start.txt', 'w') # 書き込みモードで開く
for key,value in sorted(output.items()):
    f.write(f'{key} {value}\n')
f.close()


## goal

frame = cv2.imread('goal.jpg')
gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
num_id = 0
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dict_aruco, parameters=parameters)


# 左上(原点)
cornerUL = np.array([0., 0.])
origin  = np.array([-corners[np.where(ids == 0)[0][0]][0][0][1], corners[np.where(ids == 0)[0][0]][0][0][0]])
# 右上
cornerUR = np.array([-corners[np.where(ids == 1)[0][0]][0][1][1], corners[np.where(ids == 1)[0][0]][0][1][0]]) - origin
# 右下
cornerBR = np.array([-corners[np.where(ids == 2)[0][0]][0][2][1], corners[np.where(ids == 2)[0][0]][0][2][0]]) - origin
# 左下
cornerBL = np.array([-corners[np.where(ids == 3)[0][0]][0][3][1], corners[np.where(ids == 3)[0][0]][0][3][0]]) - origin
# objectの左上
objectUL = np.array([-corners[np.where(ids == 4)[0][0]][0][0][1], corners[np.where(ids == 4)[0][0]][0][0][0]]) - origin
# ルンバ中心
roombaUL = np.array([-corners[np.where(ids == 5)[0][0]][0][0][1], corners[np.where(ids == 5)[0][0]][0][0][0]]) - origin
rommbaUR = np.array([-corners[np.where(ids == 5)[0][0]][0][1][1], corners[np.where(ids == 5)[0][0]][0][1][0]]) - origin
roombaC = (roombaUL + rommbaUR)/2

goal_unit_cm2sim = 64 / area_hight
goal_unit_aruco2sim = 64 / abs(cornerUL[1] - cornerBL[1])
goal_roomba_init = roombaC * goal_unit_aruco2sim
goal_roomba_hight = roomba_hight * goal_unit_cm2sim
goal_roomba_radian = roomba_radian * goal_unit_cm2sim

goal_box_hight = box_hight * goal_unit_cm2sim
goal_box_length = box_length * goal_unit_cm2sim
goal_box_depth = box_depth * goal_unit_cm2sim
goal_box_init = objectUL*goal_unit_aruco2sim + np.array([goal_box_length/2, goal_box_depth/2])

output = {
    'roomba_init': goal_roomba_init,
    'roomba_hight': goal_roomba_hight,
    'roomba_radian': goal_roomba_radian,
    'box_init': goal_box_init,
    'box_length': goal_box_length,
    'box_hight': goal_box_hight,
    'box_depth': goal_box_depth,
}

f = open('goal.txt', 'w') # 書き込みモードで開く
for key,value in sorted(output.items()):
    f.write(f'{key} {value}\n')
f.close()