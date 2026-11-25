# -*- coding: utf-8 -*-
import time
from Arm_Lib import Arm_Device

from params import *


def read_servo_degree(Arm):
    Arm.Arm_serial_set_torque(0)
    print("\n机械臂舵机力矩已关闭，可以掰动\n")
    input("\n当机械臂掰动到位后请输按回车\n")
    Arm.Arm_serial_set_torque(1)
    print("\n机械臂舵机力矩已启动\n")
    time.sleep(.2)
    for i in range(6):
        aa = Arm.Arm_serial_servo_read(i + 1)
        print("第" + str(i + 1) + "号舵机角度为：", aa)


def arm_wait(arm_stack_len):
    """
    等待执行完栈中所有的指令

    Args:
        arm_stack_len:      栈长度 (共享)
    """
    # 等待完成全部动作
    while arm_stack_len.value:
        time.sleep(.1)


def arm_standby(direction, claw, arm_target, arm_stack_len=None, wait=False):
    """
    待命位(左中右)

    Args:
        direction:          方向
        claw:               夹爪松紧 (默认松开)
        arm_target:         机械臂角度 (共享)
        arm_stack_len:      指令栈长度 (共享)
        wait:           是否需要等待完成全部指令
    """
    arm_target[:] = [0, direction, 180, 0, 0, 90, claw]
    # 角度写入完成
    arm_target[0] = 1
    time.sleep(.01)  # 等待指令被取走

    if wait:
        arm_wait(arm_stack_len)


def arm_scout(arm_target, direction=arm_middle, claw=loose):
    """
    目标检测斜向下位

    Args:
        arm_target:         机械臂角度 (共享)
        direction:          方向
        claw:               夹爪松紧 (默认松开)
    """

    arm_target[:] = [0, direction, 90, 90, 0, 90, claw]
    # 角度写入完成
    arm_target[0] = 1
    time.sleep(.01)  # 等待指令被取走


def arm_midway(arm_target):
    """
    中途防碰撞位

    Args:
        arm_target:         机械臂角度 (共享)
    """
    arm_target[:] = [0, 90, 90, 30, 30, 90, loose]
    # 角度写入完成
    arm_target[0] = 1
    time.sleep(.01)  # 等待指令被取走


def arm_grab(data_ik, arm_target, arm_stack_len):
    """
    抓取全流程

    Args:
        data_ik:            抓取时的舵机角度
        arm_target:         机械臂角度 (共享)
        arm_stack_len:      指令栈长度
    """
    # 居中待命
    arm_standby(arm_middle, loose, arm_target, arm_stack_len)

    # 中途防撞
    arm_midway(arm_target)

    # 张开,前出,预备
    arm_target[:] = [0,
                     data_ik[0],
                     data_ik[1],
                     data_ik[2],
                     data_ik[3] - 25,
                     90,
                     loose]
    # 角度写入完成
    arm_target[0] = 1
    time.sleep(.02)  # 等待指令被取走

    # 张开,前出,到位
    arm_target[0] = 0
    arm_target[4] = data_ik[3]
    # 角度写入完成
    arm_target[0] = 1
    # 等待完成全部动作
    arm_wait(arm_stack_len)

    # 抓紧
    arm_target[0] = 0
    arm_target[6] = tight
    # 角度写入完成
    arm_target[0] = 1
    # 等待完成全部动作
    arm_wait(arm_stack_len)
    time.sleep(.3)
#    # 摘下
#    arm_target[:] = [87, 75, 113, 152, 87, 29, loose]
    # 抓紧，摘下
#    arm_target[0] = 0
#    arm_target[4] = data_ik[3] - 15

    # 角度写入完成
    arm_target[0] = 1
    time.sleep(.02)  # 等待指令被取走

    # 回中
    arm_standby(arm_middle, tight, arm_target, arm_stack_len)

    # 回中
    arm_standby(arm_middle, tight, arm_target, arm_stack_len)

    # 等待完成全部动作
    arm_wait(arm_stack_len)


def put_temp(arm_target, arm_stack_len):
    """
    将抓取到的球放置到在指定承接位的格子中

    Args:
        arm_target:         机械臂角度 (共享)
        arm_stack_len:      指令栈长度 (共享)
    """
    # #####################################################################
    # TODO 添加放置到固定位置的动作  读取舵机角度来获得
    # TODO 注意 由于储物罐在机械臂后方 放置时机械臂需向后大角度翻转 需考虑分解放置动作  以下为参考
    arm_target[:] = [0, 90, 130, 20, 80, 90, tight]
    # 角度写入完成
    arm_target[0] = 1
    time.sleep(.01)  # 等待指令被取走
    arm_target[:] = [0,76, 90, 97, 88, 86, tight]
#    arm_target[:] = [0, 90, 90, 90, 90, 90, tight]
    # 角度写入完成
    arm_target[0] = 1
    time.sleep(.01)  # 等待指令被取走
    arm_target[:] = [0 ,87, 75, 113, 152, 90, tight]
#    arm_target[:] = [0, 90, 50, 155, 170, 90, tight]
    # 角度写入完成
    arm_target[0] = 1
    time.sleep(.01)  # 等待指令被取走
    arm_target[:] = [0 ,87, 75, 113, 152, 90, loose]
#    arm_target[:] = [0, 90, 50, 155, 170, 90, loose]
    # 角度写入完成
    arm_target[0] = 1
    time.sleep(.01)  # 等待指令被取走
    # #####################################################################

    # 回中
    arm_standby(arm_middle, loose, arm_target, arm_stack_len)

    # 摘下
#    arm_target[:] = [87, 75, 113, 152, 87, 29, loose]
    # 回中
    arm_standby(arm_middle, loose, arm_target, arm_stack_len)

    # 等待完成全部动作
    arm_wait(arm_stack_len)
