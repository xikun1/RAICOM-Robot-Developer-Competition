"""Send a Cartesian grasp pose to the robot's local inverse-kinematics service."""

import math
import os
import socket


def socket_communicate(data):
    address = os.environ.get("RAICOM_IK_SOCKET")
    if not address:
        raise RuntimeError("Provide the robot's kinematics socket with --ik-socket")
    if len(data) != 6 or not all(math.isfinite(value) for value in data):
        raise ValueError("The kinematics service expects six finite pose values")
    message = (" ".join(map(str, data)) + " ").encode("ascii")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(10)
        client.connect(address)
        client.sendall(message)
        angles = [float(value) for value in client.recv(1024).decode("ascii").split()]
    if len(angles) < 4 or not all(math.isfinite(angle) for angle in angles):
        raise RuntimeError("The kinematics service returned an incomplete joint solution")
    return angles
