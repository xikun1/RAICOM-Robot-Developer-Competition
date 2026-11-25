"""Forward route commands to the chassis's line-following controller."""

from collections import deque
from threading import Lock, Thread
import time

import serial


def wait_for_reply(port, active):
    deadline = time.monotonic() + 30
    while active.value and time.monotonic() < deadline:
        waiting = port.in_waiting
        if waiting:
            port.read(waiting)
            return
        time.sleep(0.01)
    if active.value:
        raise TimeoutError("The chassis did not acknowledge a serial command")


def chassis_control(active, mailbox, queue_length, serial_port):
    pending = deque()
    lock = Lock()
    receiver = None

    def collect_commands():
        while active.value:
            with mailbox.get_lock():
                if mailbox[0] == b"1":
                    with lock:
                        pending.append(mailbox[1])
                        queue_length.value += 1
                    mailbox[0] = b"0"
            time.sleep(0.01)

    try:
        with serial.Serial(serial_port, 115200, timeout=0.2, write_timeout=2) as port:
            port.write(b"0")
            wait_for_reply(port, active)
            receiver = Thread(target=collect_commands)
            receiver.start()
            while active.value:
                with lock:
                    command = pending.popleft() if pending else None
                if command is None:
                    time.sleep(0.02)
                    continue
                port.write(command)
                wait_for_reply(port, active)
                with lock:
                    queue_length.value -= 1
    finally:
        active.value = 0
        if receiver is not None:
            receiver.join()
