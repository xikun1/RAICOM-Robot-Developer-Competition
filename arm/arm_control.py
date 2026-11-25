# -*- coding: utf-8 -*-
"""
可以尝试使用multiprocessing中的Queue队列
来代替本代码中自定义的指令栈
但笔者测试时发现，当需要等待指令全部执行完时，不容易在队列中最后一个指令被取走并执行完成后再认为已经全部执行完
故搁置队列的方案
"""
from threading import Thread
from multiprocessing import Process, Value, Array
from Arm_Lib import Arm_Device

from arm.motions import *


Arm = Arm_Device()
time.sleep(.01)


def arm_control(flag_task_a, arm_target_a, arm_stack_len_a):
    """
    机械臂控制

    Args:
        flag_task_a:        任务开启标志位（共享）
        arm_target_a:       机械臂角度（共享）
        arm_stack_len_a:    机械臂指令栈长度
    """
    my_print("这里是机械臂进程")
    my_print("机械臂居中，开始初始化", "tip")
    Arm.Arm_serial_servo_write6(90 + delta_servo_1, 180 + delta_servo_2,
                                0 + delta_servo_3, 0 + delta_servo_4,
                                90 + delta_servo_5, loose, 800)
    time.sleep(.1)
    Arm.Arm_serial_servo_write6(90 + delta_servo_1, 180 + delta_servo_2,
                                0 + delta_servo_3, 0 + delta_servo_4,
                                90 + delta_servo_5, loose, 800)
    time.sleep(.1)
    my_print("机械臂完成初始化", "tip")

    arm_stack = []

    def enable_arm_stack(flag_task_l, arm_target_l, arm_stack_len_l):
        """
        机械臂指令栈

        Args:
            flag_task_l:    任务开启标志位（共享）
            arm_target_l:   机械臂角度（共享）
            arm_stack_len_l:指令栈长度（共享）
        """
        while flag_task_l.value:
            if arm_target_l[0]:
                arm_stack.append(arm_target_l[1:])
                arm_target_l[0] = 0
                arm_stack_len_l.value = len(arm_stack)

    t_a_stack = Thread(target=enable_arm_stack,
                       args=(flag_task_a, arm_target_a, arm_stack_len_a))
    t_a_stack.start()

    arm_last = [90, 180, 0, 0, 90, 90]

    while flag_task_a.value:

        if arm_stack_len_a.value == 0:
            time.sleep(.1)
            continue

        arm_now = arm_stack[0]

        delta_max = int(max([abs(arm_now[i] - arm_last[i]) for i in range(5)]))
        delta_t = max(delta_max * 5, 400)

        Arm.Arm_serial_servo_write6(arm_now[0] + delta_servo_1, arm_now[1] + delta_servo_2,
                                    arm_now[2] + delta_servo_3, arm_now[3] + delta_servo_4,
                                    arm_now[4] + delta_servo_5, arm_now[5], delta_t)
        time.sleep(delta_t / 800)

        arm_last = arm_now
        arm_stack.pop(0)
        arm_stack_len_a.value = len(arm_stack)  # 更新栈长度

    t_a_stack.join()
    my_print("机械臂进程结束")
