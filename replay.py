"""Inspect a detection frame and the decisions it would produce without a robot."""

import json

from arm.coord_2_claw import arm_coord_2_claw
from obj_detection.decisions import most_common_label
from obj_detection.yolo_decoder import encode_yolo_frame, yolo_decoder
from params import DETECTION_CLASSES, fruit_code, img_size, sign_code


def read_image(path):
    import cv2
    import numpy as np

    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode image: {}".format(path))
    return image


def summarize(size, detections):
    frame_id, count, objects = yolo_decoder(encode_yolo_frame(1, detections))
    route = most_common_label(objects, sign_code)
    fruit = most_common_label(objects, fruit_code)
    width, height = size
    targets = []
    for item in objects:
        if item["label"] == fruit:
            x1, y1, x2, y2 = item["bbox"]
            pixel = ((x1 + x2) * img_size[0] / (2 * width), (y1 + y2) * img_size[1] / (2 * height))
            targets.append([round(value, 3) for value in arm_coord_2_claw(pixel)])
    return {
        "frame_id": frame_id,
        "input_detections": len(detections),
        "decoded_detections": count,
        "route_marker": None if route is None else DETECTION_CLASSES[route],
        "fruit": None if fruit is None else DETECTION_CLASSES[fruit],
        "grasp_targets_cm_and_degrees": targets,
        "detections": objects,
    }


def save_overlay(image, objects, output):
    import cv2

    overlay = image.copy()
    for item in objects:
        x1, y1, x2, y2 = item["bbox"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (54, 218, 246), 2)
        cv2.putText(overlay, DETECTION_CLASSES[item["label"]], (x1, max(y1 - 6, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (54, 218, 246), 2)
    saved, data = cv2.imencode(output.suffix, overlay)
    if not saved:
        raise ValueError("Could not encode {}".format(output))
    data.tofile(output)


def replay_sample(path, output=None):
    sample = json.loads(path.read_text(encoding="utf-8"))
    image_path = path.parent / sample["image"]
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    width, height = sample["image_size"]
    if width <= 0 or height <= 0:
        raise ValueError("The recorded image size must be positive")
    for item in sample["detections"]:
        label = item["label"]
        x1, y1, x2, y2 = item["bbox"]
        if not 0 <= label < len(DETECTION_CLASSES) or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("The sample has a label or box outside the valid range")
    result = summarize((width, height), sample["detections"])
    if output:
        image = read_image(image_path)
        if image.shape[:2] != (height, width):
            raise ValueError("The image does not match the recorded dimensions")
        save_overlay(image, result["detections"], output)
    return {"source": "recorded detections", **result}


def run_inference(model, image_path, output=None):
    from obj_detection.onnx_detector import OnnxDetector

    image = read_image(image_path)
    detections = OnnxDetector(model).detect(image)
    result = summarize((image.shape[1], image.shape[0]), detections)
    if output:
        save_overlay(image, result["detections"], output)
    return {"source": "YOLOv5 ONNX on CPU", **result}
