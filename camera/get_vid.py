"""Copy USB camera frames into the shared 640 x 480 image buffer."""

import cv2
import numpy as np

from params import img_size


def get_vid(flag_task, frame_store, frame_lock, ready, camera_index):
    width, height = img_size
    shared = np.frombuffer(frame_store, dtype=np.uint8).reshape(height, width, 3)
    camera = cv2.VideoCapture(camera_index)
    try:
        if not camera.isOpened():
            raise RuntimeError("Cannot open camera {}".format(camera_index))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        camera.set(cv2.CAP_PROP_BRIGHTNESS, 16)
        camera.set(cv2.CAP_PROP_CONTRAST, 32)
        while flag_task.value:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("The camera stopped returning frames")
            if frame.shape != shared.shape:
                frame = cv2.resize(frame, (width, height))
            with frame_lock:
                shared[:] = frame
            ready.set()
    finally:
        flag_task.value = 0
        camera.release()
