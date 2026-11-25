import unittest

from obj_detection.decisions import most_common_label
from obj_detection.yolo_decoder import FRAME_BYTES, encode_yolo_frame, yolo_decoder
from params import fruit_code, sign_code


class DetectionFrameTests(unittest.TestCase):
    def test_frame_never_splits_a_box(self):
        detections = [{"label": 5, "bbox": [510, 350, 630, 470]}] * 15
        frame = encode_yolo_frame(123456, detections)
        self.assertEqual(len(frame), FRAME_BYTES)
        frame_id, count, decoded = yolo_decoder(frame)
        self.assertEqual(frame_id, 123456)
        self.assertTrue(0 < count < len(detections))
        self.assertEqual(decoded, detections[:count])

    def test_incomplete_messages_are_ignored(self):
        self.assertEqual(yolo_decoder(b"#6@2@5@10@20@30@40@*"), (0, 0, []))
        self.assertEqual(yolo_decoder(encode_yolo_frame(6, [])), (6, 0, []))

    def test_direction_markers_and_fruit_use_separate_votes(self):
        detections = [{"label": label} for label in (7, 7, 7, 0, 1, 1, 6)]
        self.assertEqual(most_common_label(detections, sign_code), 1)
        self.assertEqual(most_common_label(detections, fruit_code), 6)
