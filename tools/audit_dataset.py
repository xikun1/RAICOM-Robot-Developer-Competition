"""Audit local YOLO image and label pairs without changing the dataset."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path


def files_by_stem(root, extensions):
    if not root.is_dir():
        raise ValueError("Directory does not exist: {}".format(root))
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            key = path.relative_to(root).with_suffix("").as_posix()
            if key in files:
                raise ValueError("Duplicate path stem: {}".format(key))
            files[key] = path
    return files


def audit(image_root, label_root):
    images = files_by_stem(image_root, {".jpg", ".jpeg", ".png"})
    labels = files_by_stem(label_root, {".txt"})
    counts = Counter()
    invalid = []
    for key in sorted(images.keys() & labels.keys()):
        path = labels[key]
        for number, row in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not row.strip():
                continue
            try:
                label, x, y, width, height = row.split()
                label_id = int(label)
                coordinates = tuple(map(float, (x, y, width, height)))
                if label_id < 0 or not all(math.isfinite(v) and 0 <= v <= 1 for v in coordinates):
                    raise ValueError("Invalid normalized coordinates or class ID")
                if min(coordinates[2:]) <= 0:
                    raise ValueError("Empty bounding box")
                counts[label_id] += 1
            except ValueError:
                invalid.append({"file": path.relative_to(label_root).as_posix(), "line": number})
    return {
        "paired_images": len(images.keys() & labels.keys()),
        "images_without_labels": sorted(images.keys() - labels.keys()),
        "labels_without_images": sorted(labels.keys() - images.keys()),
        "instances_by_class_id": dict(sorted(counts.items())),
        "invalid_rows": invalid,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path)
    parser.add_argument("labels", type=Path)
    args = parser.parse_args()
    try:
        report = audit(args.images, args.labels)
    except (OSError, ValueError) as error:
        parser.exit(1, "Error: {}\n".format(error))
    print(json.dumps(report, indent=2))
    if report["invalid_rows"] or report["images_without_labels"] or report["labels_without_images"]:
        parser.exit(1)
