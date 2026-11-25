import time
from multiprocessing import Process

import numpy as np

from task.task_functions import *


def task(flag_task,
         img_mem, detect_frame, flag_yolo, flag_mask_type,
         grey_cmd, grey_stack_len,
         arm_target, arm_stack_len,
         storage_target, storage_stack_len):
    """
    任务进程

    Args:
        flag_task:          任务进行标志位 (共享)
        img_mem:            固定位相机捕获的帧 (共享)
        detect_frame:       接收yolo检测数据 (共享)
        flag_yolo:          目标检测开启标志位 (默认开启) (共享)
        flag_mask_type:     当前视野加蒙版(防止检查到隔壁树, 默认不加) (共享)
        grey_cmd:           底盘控制 (共享)
        grey_stack_len:     灰度指令栈长度 (共享)
        arm_target:         机械臂角度 (共享)
        arm_stack_len:      机械臂指令栈长度 (共享)
        storage_target:     机械臂角度 (共享)
        storage_stack_len:  机械臂指令栈长度 (共享)
    """
    img_mem = np.frombuffer(img_mem, dtype=np.uint8).reshape(img_size[1], img_size[0], 3)
    my_print("从出发区前往岔路口选择道路", "tip")
    go_choose_path(flag_task,
                   detect_frame, flag_yolo, flag_mask_type,
                   grey_cmd, grey_stack_len,
                   arm_target)

    # 抓取
    my_print("前往抓取", "tip")
    go_grab_12(flag_task,
               flag_yolo, flag_mask_type, img_mem, detect_frame,
               grey_cmd, grey_stack_len,
               arm_target, arm_stack_len,
               storage_target, storage_stack_len)

    # 去放置
    my_print("前往放置", "tip")
    go_put(flag_task,
           flag_mask_type, flag_yolo, detect_frame,
           grey_cmd, grey_stack_len,
           arm_target, arm_stack_len,
           storage_target, storage_stack_len)

    # 去终点
    flag_yolo.value = 0
    chassis_gray('8', grey_cmd, grey_stack_len, wait=True)
    my_print("到达泊车区  任务结束", "tip")

    time.sleep(1)
    flag_task.value = 0


def task_management(flag_task,
                    img_mem, detect_frame, flag_yolo, flag_mask_type,
                    grey_cmd, grey_stack_len,
                    arm_target, arm_stack_len,
                    storage_target, storage_stack_len):
    """
    任务时间管理进程

    Args:
        flag_task:          任务进行标志位 (共享)
        img_mem:            固定位相机捕获的帧 (共享)
        detect_frame:       接收yolo检测数据 (共享)
        flag_yolo:          目标检测开启标志位 (默认开启) (共享)
        flag_mask_type:     当前视野加蒙版(防止检查到隔壁树, 默认不加) (共享)
        grey_cmd:           底盘控制 (共享)
        grey_stack_len:     灰度指令栈长度 (共享)
        arm_target:         机械臂角度 (共享)
        arm_stack_len:      机械臂指令栈长度 (共享)
        storage_target:     机械臂角度 (共享)
        storage_stack_len:  机械臂指令栈长度 (共享)
    """
    t_start = time.time()

    p_task = Process(target=task,
                     args=(flag_task,
                           img_mem, detect_frame, flag_yolo, flag_mask_type,
                           grey_cmd, grey_stack_len,
                           arm_target, arm_stack_len,
                           storage_target, storage_stack_len))
    p_task.start()

    while flag_task.value and p_task.is_alive() and time.time() - t_start < task_time:
        p_task.join(timeout=0.2)
    flag_task.value = 0
    p_task.join(timeout=1)
    if p_task.is_alive():
        p_task.terminate()
        p_task.join()
    elif p_task.exitcode != 0:
        raise RuntimeError('The robot task process failed')
