#! /usr/bin/env python3

import rospy
import cv2
from cv_bridge import CvBridge
import copy
import message_filters
import numpy as np
from sensor_msgs.msg import Image, CameraInfo

class Process:
    def __init__(self):
        rospy.Subscriber('/camera/color/image_raw', Image, self.color_cb)
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_cb)
        rospy.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo, self.camera_cb)
        self.rgb_sub, self.depth_sub, self.camera_info_sub = None, None, None
        
        self.debug_pub = rospy.Publisher('/debug_mask', Image, queue_size=10)
        self.depth_pub = rospy.Publisher('/masked_depth', Image, queue_size=10)
        self.camera_info_pub = rospy.Publisher('/masked_depth/camera_info', CameraInfo, queue_size=10)
        self.rotate_img_pub = rospy.Publisher('/rotated_img', Image, queue_size=10)

    def color_cb(self, image):
        self.rgb_sub = image

    def depth_cb(self, depth):
        self.depth_sub = depth
    
    def camera_cb(self, info):
        self.camera_info_sub = info
    

    def make_masked_depth(self):
        current_depth = self.depth_sub
        current_camera_info = self.camera_info_sub

        bgr_img = CvBridge().imgmsg_to_cv2(self.rgb_sub, "bgr8")
        hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        gray_hsv_img = cv2.cvtColor(hsv_img, cv2.COLOR_BGR2GRAY)

        #180度回転させた画像をpublish
        img_rotate_180_clockwise = cv2.rotate(bgr_img, cv2.ROTATE_180)
        tmp_rot = CvBridge().cv2_to_imgmsg(img_rotate_180_clockwise, "bgr8")
        self.rotate_img_pub.publish(tmp_rot)

        #大津のアルゴリズムを用いて二値化
        ret_hsv, img_thresh = cv2.threshold(gray_hsv_img, 0, 255, cv2.THRESH_OTSU)

        mask_hsv = CvBridge().cv2_to_imgmsg(img_thresh, encoding="passthrough")
        self.debug_pub.publish(mask_hsv)

        #実際に使うmaskを以下で計算
        depth_img = CvBridge().imgmsg_to_cv2(current_depth, "passthrough")
        masked_depth_img = copy.deepcopy(depth_img)

        #ガウシアンフィルタでノイズを除去
        img_thresh = cv2.GaussianBlur(img_thresh, (5, 5), sigmaX=1.3)

        masked_depth_img = masked_depth_img * (img_thresh//255)

        #一定の深さ以下のものは黒で塗りつぶすことで、地面の影響を抑える（heuristic）
        masked_depth_img[masked_depth_img > 670] = 0

        masked_depth_img = CvBridge().cv2_to_imgmsg(masked_depth_img, encoding="passthrough")
        
        #マスクして得られた布のdepth画像とカメラ情報を出力 
        t = rospy.Time.now()
        masked_depth_img.header.stamp = t
        masked_depth_img.header.frame_id = "eye_on_hand_camera_color_optical_frame"
        current_camera_info.header.stamp = t

        self.depth_pub.publish(masked_depth_img)
        self.camera_info_pub.publish(current_camera_info)

    def main(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.rgb_sub is None or self.depth_sub is None or self.camera_info_sub is None:
                # print("hoge")
                continue
            self.make_masked_depth()
            rate.sleep()

if __name__ == '__main__':
    rospy.init_node('process')

    pro = Process()
    pro.main()