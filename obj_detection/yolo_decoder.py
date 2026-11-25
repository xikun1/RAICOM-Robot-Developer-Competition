"""Fixed-size detection messages shared by the inference and task processes."""

FRAME_BYTES = 128


def encode_yolo_frame(frame_id, detections):
    """Write as many complete detections as fit, in the caller's priority order."""
    rows = []
    for detection in detections:
        label = int(detection["label"])
        x1, y1, x2, y2 = (int(value) for value in detection["bbox"])
        if label < 0 or x2 <= x1 or y2 <= y1:
            continue
        row = "{}@{}@{}@{}@{}@".format(label, x1, y1, x2, y2)
        header = "#{}@{}@".format(frame_id, len(rows) + 1)
        if len(header) + sum(map(len, rows)) + len(row) + 1 > FRAME_BYTES:
            break
        rows.append(row)
    frame = "#{}@{}@{}*".format(frame_id, len(rows), "".join(rows)).encode("ascii")
    if len(frame) > FRAME_BYTES:
        raise ValueError("Frame ID does not fit the detection buffer")
    return frame.ljust(FRAME_BYTES, b"*")


def yolo_decoder(frame):
    """Decode a complete message. Ignore empty or partially written messages."""
    try:
        if isinstance(frame, bytes):
            frame = frame.decode("ascii")
        if not isinstance(frame, str) or not frame.startswith("#") or "*" not in frame:
            return 0, 0, []
        fields = frame[1:].split("*", 1)[0].rstrip("@").split("@")
        frame_id, count = map(int, fields[:2])
        if frame_id < 0 or count < 0 or len(fields) != 2 + 5 * count:
            return 0, 0, []
        boxes = []
        for offset in range(2, len(fields), 5):
            label, x1, y1, x2, y2 = map(int, fields[offset:offset + 5])
            if label < 0 or x2 <= x1 or y2 <= y1:
                return 0, 0, []
            boxes.append({"label": label, "bbox": [x1, y1, x2, y2]})
        return frame_id, count, boxes
    except (UnicodeDecodeError, ValueError, TypeError):
        return 0, 0, []
