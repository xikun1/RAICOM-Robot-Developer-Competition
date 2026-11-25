"""Run a recorded sample, CPU inference, or the physical robot controller."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")

    replay = commands.add_parser("replay", help="Replay recorded detections offline")
    replay.add_argument("sample", type=Path, help="Path to a detection JSON file")
    replay.add_argument("--overlay", type=Path, help="Save a visualization of the decoded frame")

    infer = commands.add_parser("infer", help="Run a local YOLOv5 ONNX model on the CPU")
    infer.add_argument("model", type=Path)
    infer.add_argument("image", type=Path)
    infer.add_argument("--overlay", type=Path, help="Save a visualization of the decoded frame")

    robot = commands.add_parser("robot", help="Run the competition controller on the Jetson robot")
    robot.add_argument("--engine", required=True, type=Path, help="Compatible TensorRT engine")
    robot.add_argument("--plugin", required=True, type=Path, help="Matching TensorRT plugin library")
    robot.add_argument("--serial-port", required=True, help="Chassis controller serial device")
    robot.add_argument("--ik-socket", required=True, type=Path, help="Arm kinematics server Unix socket")
    robot.add_argument("--camera", type=int, default=0)
    robot.add_argument("--display", action="store_true", help="Show live detections with GUI OpenCV")

    args = parser.parse_args()
    try:
        if args.command == "replay":
            from replay import replay_sample

            print(json.dumps(replay_sample(args.sample, args.overlay), indent=2))
        elif args.command == "infer":
            from replay import run_inference

            print(json.dumps(run_inference(args.model, args.image, args.overlay), indent=2))
        elif args.command == "robot":
            from robot_runtime import run_robot

            run_robot(args)
        else:
            parser.print_help()
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.exit(1, "Error: {}\n".format(error))


if __name__ == "__main__":
    main()
