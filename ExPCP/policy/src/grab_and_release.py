#!/usr/bin/env python3
import rospy
from setup import Setup
import math

def main():
    s = Setup()
    
    s.reset()
    # yoko x 
    # tate y
    s.grab_and_release(delta_x=0.1, delta_y=0.2, delta_theta=45)
    
    
if __name__ == '__main__':
    rospy.init_node('grab_and_release')
    
    main()