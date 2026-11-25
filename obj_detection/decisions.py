"""Label selection shared by the controller and the desktop replay."""

from collections import Counter


def most_common_label(detections, allowed):
    counts = Counter(item["label"] for item in detections if item["label"] in allowed)
    return min(counts, key=lambda label: (-counts[label], label)) if counts else None
