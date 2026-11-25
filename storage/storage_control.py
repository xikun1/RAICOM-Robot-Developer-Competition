# -*- coding: utf-8 -*-
import time
from threading import Thread
from multiprocessing import Process, Value, Array
from Arm_Lib import Arm_Device

from params import *

Arm = Arm_Device()
time.sleep(.01)


def storage_control(flag_task_s, storage_target_s, storage_stack_len_s):
    """
    机械臂控制

    Args:
        flag_task_s:            任务开启标志位（共享）
        storage_target_s:       储物罐指令（共享）
        storage_stack_len_s:    储物罐指令栈长度（共享）
    """
    my_print("这里是储物罐进程")
    my_print("储物罐舵机自检", "tip")
    Arm.Arm_PWM_servo_write(cross_id, 0)  # 十字拨盘左转限位
    time.sleep(.3)
    Arm.Arm_PWM_servo_write(cross_id, 180)  # 十字拨盘右转限位
    time.sleep(.3)
    Arm.Arm_PWM_servo_write(cross_id, 90)  # 十字拨盘居中
    time.sleep(.3)

    Arm.Arm_PWM_servo_write(left_diversion_id, 180)  # 左导流槽最大开度限位
    time.sleep(.3)
    Arm.Arm_PWM_servo_write(right_diversion_id, 0)  # 右导流槽最大开度限位  (左右舵机同向安装时角度相反)
    time.sleep(.3)
    Arm.Arm_PWM_servo_write(left_diversion_id, 0)  # 左导流槽关闭
    time.sleep(.3)
    Arm.Arm_PWM_servo_write(right_diversion_id, 180)  # 右导流槽关闭
    time.sleep(.3)

    my_print("储物罐自检完成", "tip")

    storage_stack = []

    def enable_storage_stack(flag_task_l, storage_target_l, storage_stack_len_l):
        """
        储物罐指令栈

        Args:
            flag_task_l:        任务开启标志位（共享）
            storage_target_l:   机械臂角度（共享）
            storage_stack_len_l:指令栈长度（共享）
        """
        while flag_task_l.value:
            if storage_target_l[0]:
                storage_stack.append(storage_target_l[1])
                storage_target_l[0] = 0
                storage_stack_len_l.value = len(storage_stack)

    t_s_stack = Thread(target=enable_storage_stack,
                       args=(flag_task_s, storage_target_s, storage_stack_len_s))
    t_s_stack.start()

    while flag_task_s.value:

        if storage_stack_len_s.value == 0:
            time.sleep(.1)
            continue
        cmd_str = str(storage_stack[0])  # 0 10 20 30 40 10 11 20 21 30 31 40 41
        storage_now = storage_dic.get(storage_stack[0])  # [cross left  right]

        # 十字拨盘
        Arm.Arm_PWM_servo_write(cross_id, int(storage_now[0] * 180 / 270))
        time.sleep(.5)

        # 导流槽
        if len(cmd_str) == 1:  # 待命 指令
            Arm.Arm_PWM_servo_write(left_diversion_id, 0)  # 左导流槽关
            Arm.Arm_PWM_servo_write(right_diversion_id, 180)  # 右导流槽关
            storage_stack.pop(0)
            storage_stack_len_s.value = len(storage_stack)
        else:
            if cmd_str[1] == '0':  # 承接放置 指令
                Arm.Arm_PWM_servo_write(left_diversion_id, 0)  # 左导流槽关
                Arm.Arm_PWM_servo_write(right_diversion_id, 180)  # 右导流槽关
            elif cmd_str[1] == '1':  # 左导流槽
                Arm.Arm_PWM_servo_write(left_diversion_id, storage_now[1])  # 左导流槽开
                time.sleep(1)  # 等待滚落
                Arm.Arm_PWM_servo_write(left_diversion_id, 0)  # 左导流槽关
            else:  # 右导流槽
                Arm.Arm_PWM_servo_write(right_diversion_id, storage_now[2])  # 右导流槽开
                time.sleep(1)  # 等待滚落
                Arm.Arm_PWM_servo_write(right_diversion_id, 180)  # 右导流槽关
            time.sleep(.5)  # 等待收起

            storage_stack.pop(0)
            storage_stack_len_s.value = len(storage_stack)  # 更新栈长度

    t_s_stack.join()
    my_print("储物罐进程结束")
