#!/usr/bin/env python3

import rospy

from dynamic_reconfigure.server import Server
from xarm_vcd.cfg import leaf_sizeConfig

def callback(config, level):
    rospy.loginfo("""Reconfigure Request: {leaf_size},\ 
          """.format(**config))
    # print("aaaaaaaaaaaaaaaa")
    rospy.set_param('/leaf_size', config['leaf_size'])
    return config

if __name__ == "__main__":
    rospy.init_node("leaf_size", anonymous = False)

    srv = Server(leaf_sizeConfig, callback)
    rospy.spin()