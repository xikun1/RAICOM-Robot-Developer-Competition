"""Jetson entry point for the original camera, chassis, arm, and storage workers."""

import multiprocessing as mp
import os
from pathlib import Path
import sys
import time


def run_robot(args):
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Robot mode requires the Linux Jetson setup. Use replay or infer on a desktop.")
    for file in (args.engine, args.plugin):
        if not file.is_file():
            raise FileNotFoundError(file)
    if not args.ik_socket.is_socket():
        raise FileNotFoundError("Arm kinematics server socket: {}".format(args.ik_socket))
    if not Path(args.serial_port).exists():
        raise FileNotFoundError("Chassis serial port: {}".format(args.serial_port))

    import cv2
    import numpy as np

    from arm.arm_control import arm_control
    from camera.get_vid import get_vid
    from chassis.chassis_control import chassis_control
    from obj_detection.yolo_decoder import FRAME_BYTES, encode_yolo_frame
    from obj_detection.yolov5_trt_cam import load_model, plot_boxes
    from params import fruit_code, img_size, sign_code
    from storage.storage_control import storage_control
    from task.task_management import task_management

    os.environ["RAICOM_IK_SOCKET"] = str(args.ik_socket.resolve())
    ctx = mp.get_context("spawn")
    flag_task = ctx.Value("i", 1)
    flag_yolo = ctx.Value("i", 0)
    flag_mask_type = ctx.Value("i", 0)
    height, width = img_size[1], img_size[0]
    frame_store = ctx.RawArray("B", height * width * 3)
    frame_lock = ctx.Lock()
    frame_ready = ctx.Event()
    shared_image = np.frombuffer(frame_store, dtype=np.uint8).reshape(height, width, 3)
    detections = ctx.Array("c", b"*" * FRAME_BYTES)
    grey_cmd = ctx.Array("c", b"00")
    grey_stack_len = ctx.Value("i", 0)
    arm_target = ctx.Array("d", [1, 90, 180, 0, 0, 90, 90])
    arm_stack_len = ctx.Value("i", 0)
    storage_target = ctx.Array("i", [0, 0])
    storage_stack_len = ctx.Value("i", 0)

    camera = ctx.Process(name="camera", target=get_vid,
                         args=(flag_task, frame_store, frame_lock, frame_ready, args.camera))
    workers = (
        ctx.Process(name="arm", target=arm_control, args=(flag_task, arm_target, arm_stack_len)),
        ctx.Process(name="chassis", target=chassis_control,
                    args=(flag_task, grey_cmd, grey_stack_len, args.serial_port)),
        ctx.Process(name="storage", target=storage_control,
                    args=(flag_task, storage_target, storage_stack_len)),
        ctx.Process(name="task", target=task_management,
                    args=(flag_task, frame_store, detections, flag_yolo, flag_mask_type,
                          grey_cmd, grey_stack_len, arm_target, arm_stack_len,
                          storage_target, storage_stack_len)),
    )
    model = load_model(args.engine, args.plugin)
    started = []
    try:
        camera.start()
        started.append(camera)
        if not frame_ready.wait(8) or not flag_task.value:
            raise RuntimeError("The robot camera did not provide a frame")
        for worker in workers:
            worker.start()
            started.append(worker)

        frame_id = 1
        while flag_task.value:
            for worker in started:
                if worker.exitcode not in (None, 0):
                    raise RuntimeError("{} worker failed (exit {})".format(worker.name, worker.exitcode))
            if not flag_yolo.value:
                time.sleep(0.025)
                continue
            with frame_lock:
                image = shared_image.copy()
            image, boxes, scores, labels = model.infer(image)
            items = [{"label": label, "bbox": box} for box, label in zip(boxes, labels)]
            priority = fruit_code if flag_mask_type.value else sign_code
            items.sort(key=lambda item: item["label"] not in priority)
            detections[:] = encode_yolo_frame(frame_id, items)
            frame_id += 1
            if args.display:
                cv2.imshow("RAICOM detections", plot_boxes(image, boxes, scores, labels))
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        flag_task.value = 0
        for worker in started:
            worker.join(timeout=2)
            if worker.is_alive():
                worker.terminate()
                worker.join()
        if args.display:
            cv2.destroyAllWindows()
        model.destroy()

    failures = [worker.name for worker in started if worker.exitcode not in (0, None)]
    if failures:
        raise RuntimeError("Workers stopped with errors: {}".format(", ".join(failures)))
