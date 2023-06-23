#!/usr/bin/env python3
import rospy
from setup import Setup
import math

EEF_GRIP = 0.15


def main():
    s = Setup()
    
    s.reset()
    # yoko x 
    # tate y
    gripper_x, gripper_z = 0.25, 0.25
    delta_theta = 45
    radians = math.radians(delta_theta)
    offset_x, offset_z = EEF_GRIP*math.cos(radians), EEF_GRIP*math.sin(radians)
    eef_x, eef_z = gripper_x + offset_x, gripper_z + offset_z
    print(eef_x, eef_z)
    s.grab_and_release(delta_x=eef_x, delta_y=eef_z, delta_theta=delta_theta)
    
    
if __name__ == '__main__':
    rospy.init_node('grab_and_release')
    
    main()