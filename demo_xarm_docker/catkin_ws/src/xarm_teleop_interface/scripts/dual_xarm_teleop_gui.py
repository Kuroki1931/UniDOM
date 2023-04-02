#!/usr/bin/env python3
#coding: utf-8
import rospy
import numpy as np
import copy
import sys
import time
import os
from std_msgs.msg import Float32, Int8, Bool, Empty
from xarm_msgs.msg import RobotMsg
from xarm_msgs.srv import SetInt16, ClearErr
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon


class TeleopCommand:
    CARIBRATION = 1
    START = 2
    STOP = 3
    ERROR = 4


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__()

        # windowの設定
        # self.setGeometry(1300,300,1000,850) # windowの(位置x,位置y,大きさx,大きさy)
        self.setWindowTitle('XArm Teleop GUI')   # windowのタイトル表示
        # self.setWindowIcon(QIcon("hogehoge.png"))

        # Central Widget
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)

        mainLayout = QHBoxLayout()
        centralWidget.setLayout(mainLayout)

        leftLayout = QVBoxLayout()
        rightLayout = QVBoxLayout()
        mainLayout.addLayout(leftLayout)
        mainLayout.addLayout(rightLayout)

        resetButton = QPushButton('RESET')
        goHomeButton = QPushButton('GO HOME')
        self.manualModeButton = QPushButton('MANUAL MODE')
        self.manualModeButton.setCheckable(True)
        self.manualModeButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        leftLayout.addWidget(resetButton)
        leftLayout.addWidget(goHomeButton)
        rightLayout.addWidget(self.manualModeButton)

        # click event
        resetButton.clicked.connect(self.on_reset)
        goHomeButton.clicked.connect(self.on_gohome)
        self.manualModeButton.clicked.connect(self.on_manual_toggled)

        self.command_pub_ = rospy.Publisher('/teleop/command', Int8, queue_size=1)
        self.movehome_pub_ = rospy.Publisher('/teleop/movehome', Empty, queue_size=1)
        self.reset_pub_ = rospy.Publisher('/teleop/reset', Empty, queue_size=1)
        self.enable_pub_ = rospy.Publisher('/teleop/enable', Bool, queue_size=1)
        self.command_pub_ = rospy.Publisher('/teleop/command', Int8, queue_size=1)
        self.trigger_pub_ = rospy.Publisher('/controller/trigger', Float32, queue_size=1)
        self.rosbag_pub_ = rospy.Publisher('/logging', Int8, queue_size=1)

        self.command_pub_r = rospy.Publisher('/teleop_r/command', Int8, queue_size=1)
        self.movehome_pub_r = rospy.Publisher('/teleop_r/movehome', Empty, queue_size=1)
        self.reset_pub_r = rospy.Publisher('/teleop_r/reset', Empty, queue_size=1)
        self.enable_pub_r = rospy.Publisher('/teleop_r/enable', Bool, queue_size=1)
        self.command_pub_r = rospy.Publisher('/teleop_r/command', Int8, queue_size=1)
        self.trigger_pub_r = rospy.Publisher('/controller_r/trigger', Float32, queue_size=1)
        self.rosbag_pub_r = rospy.Publisher('/logging_r', Int8, queue_size=1)

        self.set_manual_mode(False)

        self.states_pre = RobotMsg()
        self.states_pre_r = RobotMsg()
        _xx = rospy.Subscriber('/L_xarm7/xarm_states', RobotMsg, self.states_cb)
        # _xx = rospy.Subscriber('/joint_states', RobotMsg, self.states_cb)
        _tr = rospy.Subscriber('/teleop/reset', Empty, self.reset_cb)
        _tg = rospy.Subscriber('/teleop/gohome', Empty, self.gohome_cb)

        _xx_r = rospy.Subscriber('/R_xarm7/xarm_states', RobotMsg, self.states_cb_r)
        # _xx = rospy.Subscriber('/joint_states', RobotMsg, self.states_cb)
        _tr_r = rospy.Subscriber('/teleop_r/reset', Empty, self.reset_cb_r)
        _tg_r = rospy.Subscriber('/teleop_r/gohome', Empty, self.gohome_cb_r)

    def states_cb(self, states):
        # print(states)
        if self.states_pre.err == 0 and states.err != 0:
            self.reset_pub_.publish()
        self.states_pre = states

    def gohome_cb(self, data):
        self.on_gohome()

    def reset_cb(self, empty):
        command = Int8()
        command.data = TeleopCommand.ERROR
        self.command_pub_.publish(command)

        data = Bool()
        data = False
        self.enable_pub_.publish(data)

        self.clear_err()
        self.set_manual_mode(True)
        # ret = QMessageBox.warning(None, "Warning", "Move the robot to a safe place and then press the Yes button.", QMessageBox.Yes)
        # if ret == QMessageBox.Yes:
        #     self.set_manual_mode(False)
        #     time.sleep(0.5)
        #     self.on_gohome()


    
    def states_cb_r(self, states):
        # print(states)
        if self.states_pre_r.err == 0 and states.err != 0:
            self.reset_pub_r.publish()
        self.states_pre_r = states

    def gohome_cb_r(self, data):
        self.on_gohome_r()

    def reset_cb_r(self, empty):
        command = Int8()
        command.data = TeleopCommand.ERROR
        self.command_pub_r.publish(command)

        data = Bool()
        data = False
        self.enable_pub_r.publish(data)

        self.clear_err_r()
        self.set_manual_mode_r(True)
        # ret = QMessageBox.warning(None, "Warning", "Move the robot to a safe place and then press the Yes button.", QMessageBox.Yes)
        # if ret == QMessageBox.Yes:
        #     self.set_manual_mode(False)
        #     time.sleep(0.5)
        #     self.on_gohome()




    # reset
    def on_reset(self):
        self.reset_pub_.publish()
        print("reset")

    def clear_err(self):
        clear_err = rospy.ServiceProxy('/L_xarm7/clear_err', ClearErr)
        try:
            clear_err()
        except rospy.ServiceException as e:
            print("service call failed: %s" % e)

    # go home
    def on_gohome(self):
        self.set_manual_mode(False)
        self.movehome_pub_.publish()
        self.trigger_pub_.publish(0.0)
        print("go home completed")

    def set_manual_mode(self, on):
        clear_err = rospy.ServiceProxy('/L_xarm7/clear_err', ClearErr)
        set_mode = rospy.ServiceProxy('/L_xarm7/set_mode', SetInt16)
        set_state = rospy.ServiceProxy('/L_xarm7/set_state', SetInt16)
        data = Bool()
        if on:
            data = False
            self.enable_pub_.publish(data)
            try:
                clear_err()
                set_mode(2)
                set_state(0)
            except rospy.ServiceException as e:
                print("service call failed: %s" % e)
            self.manualModeButton.setChecked(True)
            self.manualModeButton.setStyleSheet("background-color: red")
            self.manualModeButton.setText("MANUAL MODE ON")
        else:
            data = True
            self.enable_pub_.publish(data)
            try:
                clear_err()
                set_mode(1)
                set_state(0)
            except rospy.ServiceException as e:
                print("service call failed: %s" % e)
            self.manualModeButton.setChecked(False)
            self.manualModeButton.setStyleSheet("background-color: white")
            self.manualModeButton.setText("MANUAL MODE OFF")

    # manual mode
    def on_manual_toggled(self, checked):
        self.set_manual_mode(checked)



    # reset
    def on_reset_r(self):
        self.reset_pub_r.publish()
        print("reset_r")

    def clear_err_r(self):
        clear_err = rospy.ServiceProxy('/R_xarm7/clear_err', ClearErr)
        try:
            clear_err()
        except rospy.ServiceException as e:
            print("service call failed on Rxarm: %s" % e)

    # go home
    def on_gohome_r(self):
        self.set_manual_mode_r(False)
        self.movehome_pub_r.publish()
        self.trigger_pub_r.publish(0.0)
        print("go home completed R")

    def set_manual_mode_r(self, on):
        clear_err = rospy.ServiceProxy('/R_xarm7/clear_err', ClearErr)
        set_mode = rospy.ServiceProxy('/R_xarm7/set_mode', SetInt16)
        set_state = rospy.ServiceProxy('/R_xarm7/set_state', SetInt16)
        data = Bool()
        if on:
            data = False
            self.enable_pub_r.publish(data)
            try:
                clear_err()
                set_mode(2)
                set_state(0)
            except rospy.ServiceException as e:
                print("service call failed Rxarm: %s" % e)
            self.manualModeButton.setChecked(True)
            self.manualModeButton.setStyleSheet("background-color: red")
            self.manualModeButton.setText("MANUAL MODE ON R")
        else:
            data = True
            self.enable_pub_r.publish(data)
            try:
                clear_err()
                set_mode(1)
                set_state(0)
            except rospy.ServiceException as e:
                print("service call failed: %s" % e)
            self.manualModeButton.setChecked(False)
            self.manualModeButton.setStyleSheet("background-color: white")
            self.manualModeButton.setText("MANUAL MODE OFF R")

    # manual mode
    def on_manual_toggled(self, checked):
        self.set_manual_mode(checked)


def main():
    # main process:
    rospy.init_node('dual_xarm_teleop_gui')
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
