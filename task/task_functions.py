import time

from obj_detection.decisions import most_common_label
from obj_detection.yolo_decoder import yolo_decoder
from arm.coord_2_claw import arm_coord_2_claw
from arm.socket_communication import socket_communicate
from arm.motions import *
from params import *


def chassis_gray(cmd, grey_cmd, grey_stack_len, wait=False):
    """
    底盘灰度指令巡线

    Args:
        cmd:                灰度指令
        grey_cmd:           底盘指令控制 (共享)
        grey_stack_len:     指令栈长度（共享）
        wait:               是否需要等待完成全部指令
    """
    time.sleep(.1)  # 等待指令栈取走

    grey_cmd[:] = [b'1', cmd.encode()]

    if wait:
        time.sleep(1)
        while grey_stack_len.value:
            time.sleep(.2)


def storage_response(fruit_type, diversion, storage_target, storage_stack_len, wait=False):
    """
    设置储物罐指令
    Args:
        fruit_type:         水果种类 0蓝莓 1红枣 2草莓 3无花果
        diversion:          方向 1左 2右
        storage_target:     储物罐动作编码 (共享)
        storage_stack_len:  储物罐指令栈长度 (共享)
        wait:               是否需要等待完成全部指令
    Returns:

    """
    storage_target[1] = fruit_type * 10 + diversion
    storage_target[0] = 1

    if wait:
        time.sleep(.2)
        while storage_stack_len.value:
            time.sleep(.2)


def go_choose_path(flag_task,
                   detect_frame, flag_yolo, flag_mask_type,
                   grey_cmd, grey_stack_len,
                   arm_target):
    """
    从出发区前往岔路口 识别转向标志牌 完成切换道路切换

    Args:
        flag_task:      任务进行标志位 (共享)
        detect_frame:   接收yolo检测数据 (共享)
        flag_yolo:      目标检测开启标志位 (默认开启) (共享)
        flag_mask_type: 当前视野加蒙版(防止检查到隔壁树, 默认不加) (共享)
        grey_cmd:       底盘控制 (共享)
        grey_stack_len: 灰度指令栈长度 (共享)
        arm_target:     机械臂角度 (共享)
    """
    # 机械臂瞭望
    arm_scout(arm_target, arm_middle, claw=loose)

    # 开启目标检测
    flag_mask_type.value = 0
    flag_yolo.value = 1

    # 前往1号节点
    chassis_gray('1', grey_cmd, grey_stack_len)

    last_body = []
    cnt_pr = 0
    while flag_task.value:
        time.sleep(0.02)
        # 等待检测信息写入完全
        f_decode = detect_frame[:].decode()  # 为防止判断完帧写完到解码之间主进程改写帧导致解码错乱，需先记录后再解码

        if f_decode[0] != '#':  # 帧头判断帧写完
            continue

        # 解码检测信息
        f_decode = detect_frame[:].decode()  # 为防止判断帧写完到解码之间主进程改写帧导致解码错乱，需先记录后再解码
        _, f_num, f_body = yolo_decoder(f_decode)  # 解码

        if f_num and f_body != last_body:  # 解码包中有检测到物体，且不等于上次解码的结果(防止主进程未更新)
            last_body = f_body  # 更新
            labels = [f_body[i]["label"] for i in range(f_num)]  # 记录本次检测得的物体的标签

            if not cnt_pr % 10:  # 每记录5次输出1次，减少输出
                #print("obj insight", [obj_list[labels[i]] for i in range(f_num)], "\n")
                cnt_pr += 1

            dir_mode = most_common_label(f_body, sign_code)  # 取目标检测结果众数为方向

            if dir_mode in [obj_dic.get("left"), obj_dic.get("ahead"), obj_dic.get("right")]:
                my_print(f"当前十字路口 {sign_code.get(dir_mode)}", "tip")
                if dir_mode == obj_dic.get("left"):
                    chassis_gray('a', grey_cmd, grey_stack_len)
                elif dir_mode == obj_dic.get("right"):
                    chassis_gray('d', grey_cmd, grey_stack_len)
                elif dir_mode == obj_dic.get("ahead"):
                    chassis_gray('2', grey_cmd, grey_stack_len)
                break

    # 关闭目标检测
    flag_yolo.value = 0
    time.sleep(.1)
    # 机械臂待命
    arm_standby(arm_middle, loose, arm_target)


def detect_type_loc(flag_task, detect_frame):
    """
    检测当前节点后的树上果实种、数量以及像素位置

    Args:
        flag_task:          任务进行标志位（共享）
        detect_frame:       接收yolo检测数据 (共享)

    Returns:
        种类, 数量, 位置
    """
    last_id = -1

    t0 = time.time()

    cnt_pr = 0

    while flag_task.value and time.time() - t0 < 10:
        time.sleep(0.02)
        # 等待检测信息写入完全
        f_decode = detect_frame[:].decode()  # 为防止判断完帧写完到解码之间主进程改写帧导致解码错乱，需先记录后再解码

        if f_decode[0] != '#':  # 帧头判断完帧写完
            continue

        # 解码检测信息
        f_id, f_num, f_body = yolo_decoder(f_decode)  # 解码

        if f_num and f_id > last_id:  # 解码包中有检测到物体，且id增加(防止主进程未更新)
            last_id = f_id  # 更新
            labels = [f_body[i]["label"] for i in range(f_num)]  # 记录本次检测得的物体的标签

            if not cnt_pr % 5:  # 每记录5次输出1次，减少输出
                #print("obj insight", [obj_list[labels[i]] for i in range(f_num)], "\n")
                cnt_pr += 1

            if obj_dic.get("blueberry") in labels \
                    or obj_dic.get("date") in labels \
                    or obj_dic.get("strawberry") in labels \
                    or obj_dic.get("fig") in labels:  # 存在果实种类
                fruit_type = most_common_label(f_body, fruit_code)
                fruit_num = labels.count(fruit_type)
                fruit_locations = []
                for i in range(len(labels)):
                    if labels[i] == fruit_type:
                        fruit_locations.append(f_body[i]['bbox'])
                my_print("fruit_type\tfruit_num\tfruit_locations", "tip",
                         f"   {fruit_type}\t\t{fruit_num}\t\t{fruit_locations}")
                return fruit_type, fruit_num, fruit_locations
            else:
                pass
    my_print("fruit_type\tfruit_num\tfruit_locations", "tip",
             "   0\t\t0\t\t0")
    return None, 0, None


def grab_1(fruit_type, fruit_pix,
           detect_frame,
           arm_target, arm_stack_len,
           storage_target, storage_stack_len):
    """
    完成抓取

    Args:
        fruit_type:         水果种类
        fruit_pix:          像素坐标系目标中心位置
        detect_frame:       接收yolo检测数据 (共享)
        arm_target:         机械臂角度 (共享)
        arm_stack_len:      机械臂指令栈长度 (共享)
        storage_target:     机械臂角度 (共享)
        storage_stack_len:  机械臂指令栈长度 (共享)

    Returns:
        是否抓取成功
    """
    # 解算各关节角度
    coord = arm_coord_2_claw(fruit_pix)
    data_ik = socket_communicate(coord)

    for i in range(grab_trial_times):  # 尝试 x 次
        # 抓取
        my_print(f"第 {i+1} 次抓取尝试")
        arm_grab(data_ik, arm_target, arm_stack_len)

        # 储物罐十字拨盘转动，将指定种类的格子转动到承接位置
        storage_response(fruit_type, 0, storage_target, storage_stack_len)
        # 放到储物罐
        put_temp(arm_target, arm_stack_len)
        return True

#        # 判断是否抓取到
#        last_id = -1
#        cnt_pr = 0

#        # 判断抓取成功
#        t0 = time.time()
#        while time.time() - t0 < 1.5:
#            # 等待检测信息写入完全
#            f_decode = detect_frame[:].decode()  # 为防止判断完帧写完到解码之间主进程改写帧导致解码错乱，需先记录后再解码

#            if f_decode[0] != '#':  # 帧头判断完帧写完
#                continue

#            # 解码检测信息
#            f_id, f_num, f_body = yolo_decoder(f_decode)  # 解码

#            if f_num and f_id > last_id:  # 解码包中有检测到物体，且id增加(防止主进程未更新)
#                last_id = f_id  # 更新
#                labels = [f_body[i]["label"] for i in range(f_num)]  # 记录本次检测得的物体的标签

#                if not cnt_pr % 5:  # 每记录5次输出1次，减少输出
#                    print("obj insight", [obj_list[labels[i]] for i in range(f_num)], "\n")
#                cnt_pr += 1

#                # 若视野中有球则获取球的边界框
#                if obj_dic.get('mature') in labels:
#                    bbox = f_body[labels.index(obj_dic.get('mature'))]["bbox"]
#                elif obj_dic.get('immature') in labels:
#                    bbox = f_body[labels.index(obj_dic.get('immature'))]["bbox"]
#                else:  # 若视野中无球则返回False
#                    return False
#                center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
#                # 判断球的中心是否在可认为是抓取到的区域
#                if grabbed_range[0][0] < center[0] < grabbed_range[0][1] \
#                        and grabbed_range[1][0] < center[1] < grabbed_range[1][1]:
#                    # 储物罐十字拨盘转动，将指定种类的格子转动到承接位置
#                    storage_response(fruit_type, 0, storage_target, storage_stack_len)
#                    # 放到储物罐
#                    put_temp(arm_target, arm_stack_len)
#                    return True
#                else:
#                    # TODO 由于存在抓取到但是果实标志被爪子部件遮挡而导致的无法识别到果实，本项目未对此处理，所以该抓取判断是不完备，
#                    #  建议在此条件分支中加入兴趣区域的颜色判断来增加系统健壮性（可参考rc2022项目中的颜色判断抓取成功）
#                    pass
    return False


def grab_3(flag_task,
           flag_yolo, flag_mask_type, detect_frame,
           arm_target, arm_stack_len,
           storage_target, storage_stack_len,
           time_out=45):
    """
    在位 检测种类 抓取当前树上全部球

    Args:
        flag_task:          任务进行标志位 (共享)
        flag_yolo:          目标检测开启标志位 (默认开启) (共享)
        flag_mask_type:     当前视野加蒙版(防止检查到隔壁树, 默认不加) (共享)
        detect_frame:       接收yolo检测数据 (共享)
        arm_target:         机械臂角度 (共享)
        arm_stack_len:      机械臂指令栈长度 (共享)
        storage_target:     机械臂角度 (共享)
        storage_stack_len:  机械臂指令栈长度 (共享)
        time_out:           尝试时间

    Returns:

    """
    flag_yolo.value = 1
    flag_mask_type.value = 1

    t0 = time.time()
    grabbed_cnt = 0
    while flag_task.value:
        # 超时退出
        task_time_grab = time.time() - t0
        if task_time_grab >= time_out:
            print("\n超时未完成抓取\n")
            break

#        arm_standby(arm_mfiddle, wide_open, arm_target, arm_stack_len, wait=True)
        fruit_type, fruit_num, fruit_locations = detect_type_loc(flag_task, detect_frame)
        if fruit_num:
            my_print(f"抓取 {fruit_code.get(fruit_type)}", "tip")
            fruit_pix = [(fruit_locations[0][0] + fruit_locations[0][2]) / 2,
                         (fruit_locations[0][1] + fruit_locations[0][3]) / 2]
            ret = grab_1(fruit_type, fruit_pix,
                         detect_frame,
                         arm_target, arm_stack_len,
                         storage_target, storage_stack_len)
            if ret:
                grabbed_cnt += 1
                my_print(f"当前在这棵树上抓取到 {grabbed_cnt} 个 {fruit_code.get(fruit_type)}", "tip")
                if grabbed_cnt == 3:
                    break
        else:
            break


def go_grab_12(flag_task,
               flag_yolo, flag_mask_type, img_mem, detect_frame,
               grey_cmd, grey_stack_len,
               arm_target, arm_stack_len,
               storage_target, storage_stack_len):
    """
    抓取4树12个球

    Args:
        flag_task:          任务进行标志位 (共享)
        flag_yolo:          目标检测开启标志位 (默认开启) (共享)
        flag_mask_type:     当前视野加蒙版(防止检查到隔壁树, 默认不加) (共享)
        img_mem:            固定位相机捕获的帧 (共享)
        detect_frame:       接收yolo检测数据 (共享)
        grey_cmd:           底盘控制 (共享)
        grey_stack_len:     灰度指令栈长度 (共享)
        arm_target:         机械臂角度 (共享)
        arm_stack_len:      机械臂指令栈长度 (共享)
        storage_target:     机械臂角度 (共享)
        storage_stack_len:  机械臂指令栈长度 (共享)

    Returns:

    """
    for i in range(1, 5):  # 1 2 -> 4   3 4 -> 5
        # 前往抓取位
        chassis_gray("4" if i < 3 else "5", grey_cmd, grey_stack_len)  # 前往对应节点
        if i & 1:  # 奇数树号左转grab_3
            chassis_gray(turn_left, grey_cmd, grey_stack_len)
        else:
            chassis_gray(turn_right, grey_cmd, grey_stack_len)
        my_print(f"当前转到 第 {i + 1} 棵树", "tip")
        chassis_gray(drive_ahead, grey_cmd, grey_stack_len, wait=True)  # 前往停止线

        grab_3(flag_task,
               flag_yolo, flag_mask_type, detect_frame,
               arm_target, arm_stack_len,
               storage_target, storage_stack_len)

        # 回退
        chassis_gray(drive_back, grey_cmd, grey_stack_len)
        if i & 1:  # 奇数树号右转
            chassis_gray(turn_right, grey_cmd, grey_stack_len)
        else:
            chassis_gray(turn_left, grey_cmd, grey_stack_len)


def go_put(flag_task,
           flag_mask_type, flag_yolo, detect_frame,
           grey_cmd, grey_stack_len,
           arm_target, arm_stack_len,
           storage_target, storage_stack_len):
    """
    前往测序并放置

    Args:
        flag_task:          任务进行标志位 (共享)
        detect_frame:       接收yolo检测数据 (共享)
        flag_yolo:          目标检测开启标志位 (默认开启) (共享)
        flag_mask_type:     当前视野加蒙版(防止检查到隔壁树, 默认不加) (共享) (0 转向 无蒙版  1 抓取 有蒙版 含爪  2 放置 有蒙版 去爪)
        grey_cmd:           底盘控制 (共享)
        grey_stack_len:     灰度指令栈长度 (共享)
        arm_target:         机械臂角度 (共享)
        arm_stack_len:      机械臂指令栈长度 (共享)
        storage_target:     机械臂角度 (共享)
        storage_stack_len:  机械臂指令栈长度 (共享)
    """
    # 机械臂左转
    arm_standby(arm_left, loose, arm_target)
    # 前往6号节点
    chassis_gray('6', grey_cmd, grey_stack_len, wait=True)
    my_print("当前到达1/2号放置区")

    # 添加放置位的蒙版
    flag_mask_type.value = 2
    # 开始目标检测
    flag_yolo.value = 1
    fruit_type, fruit_num, _ = detect_type_loc(flag_task, detect_frame)
    flag_yolo.value = 0  # 关闭目标检测
    if fruit_num:
        my_print(f"当前放置 {fruit_code.get(fruit_type)} 到左侧", "tip")
        storage_response(fruit_type, 1, storage_target, storage_stack_len, wait=True)

    # 机械臂右转
    arm_standby(arm_right, loose, arm_target, arm_stack_len, wait=True)
    # 开始目标检测
    flag_yolo.value = 1
    fruit_type, fruit_num, _ = detect_type_loc(flag_task, detect_frame)
    flag_yolo.value = 0  # 关闭目标检测
    if fruit_num:
        my_print(f"当前放置 {fruit_code.get(fruit_type)} 到右侧", "tip")
        storage_response(fruit_type, 2, storage_target, storage_stack_len, wait=True)

    # 前往7号节点
    chassis_gray('7', grey_cmd, grey_stack_len, wait=True)
    my_print("当前到达3/4号放置区")
    # 开始目标检测
    flag_yolo.value = 1
    fruit_type, fruit_num, _ = detect_type_loc(flag_task, detect_frame)
    flag_yolo.value = 0  # 关闭目标检测
    if fruit_num:
        my_print(f"当前放置 {fruit_code.get(fruit_type)} 到右侧", "tip")
        storage_response(fruit_type, 2, storage_target, storage_stack_len, wait=True)

    # 机械臂左转
    arm_standby(arm_left, loose, arm_target, arm_stack_len, wait=True)
    # 开始目标检测
    flag_yolo.value = 1
    fruit_type, fruit_num, _ = detect_type_loc(flag_task, detect_frame)
    flag_yolo.value = 0  # 关闭目标检测
    if fruit_num:
        my_print(f"当前放置 {fruit_code.get(fruit_type)} 到左侧", "tip")
        storage_response(fruit_type, 1, storage_target, storage_stack_len, wait=True)
