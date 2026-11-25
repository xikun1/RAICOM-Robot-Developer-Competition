"""CPU inference for a standard YOLOv5 ONNX export before non-maximum suppression."""

import cv2
import numpy as np
import onnxruntime as ort

from params import CONF_THRESH, DETECTION_CLASSES, IOU_THRESHOLD


class OnnxDetector:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        image_input, = self.session.get_inputs()
        if len(image_input.shape) != 4 or image_input.type != "tensor(float)":
            raise ValueError("Expected a float32 NCHW YOLOv5 image input")
        self.input_name = image_input.name
        self.height, self.width = [side if isinstance(side, int) else 640 for side in image_input.shape[-2:]]

    def detect(self, image):
        height, width = image.shape[:2]
        scale = min(self.width / width, self.height / height)
        resized = cv2.resize(image, (round(width * scale), round(height * scale)))
        left = (self.width - resized.shape[1]) // 2
        top = (self.height - resized.shape[0]) // 2
        padded = np.full((self.height, self.width, 3), 114, dtype=np.uint8)
        padded[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
        tensor = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None]
        prediction = self.session.run(None, {self.input_name: tensor.astype(np.float32) / 255})[0]
        if prediction.ndim != 3 or prediction.shape[0] != 1 or prediction.shape[-1] != 5 + len(DETECTION_CLASSES):
            raise ValueError("Expected [1, boxes, 14] predictions from the nine-class YOLOv5 export")

        rows = prediction[0]
        class_scores = rows[:, 5:]
        labels = np.argmax(class_scores, axis=1)
        scores = rows[:, 4] * class_scores[np.arange(len(rows)), labels]
        mask = scores >= CONF_THRESH
        rows, labels, scores = rows[mask], labels[mask], scores[mask]
        candidates = []
        for row, label, score in zip(rows, labels, scores):
            x, y, w, h = row[:4]
            x1, x2 = np.clip([(x - w / 2 - left) / scale, (x + w / 2 - left) / scale], 0, width)
            y1, y2 = np.clip([(y - h / 2 - top) / scale, (y + h / 2 - top) / scale], 0, height)
            x1, y1, x2, y2 = map(round, (x1, y1, x2, y2))
            if x2 > x1 and y2 > y1:
                candidates.append({"label": int(label), "bbox": [x1, y1, x2, y2], "score": float(score)})

        detections = []
        for label in set(item["label"] for item in candidates):
            group = [item for item in candidates if item["label"] == label]
            xywh = [[a, b, c - a, d - b] for a, b, c, d in (item["bbox"] for item in group)]
            keep = cv2.dnn.NMSBoxes(xywh, [item["score"] for item in group], CONF_THRESH, IOU_THRESHOLD)
            detections.extend(group[int(index)] for index in np.asarray(keep).reshape(-1))
        return sorted(detections, key=lambda item: item["score"], reverse=True)
