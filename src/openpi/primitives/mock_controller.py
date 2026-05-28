"""Mock ARX controller for testing the REPL driver without a real robot.

Acts as a TCP client connecting to the driver's TCP server. Sends fake
state data and camera images, receives action trajectories, and simulates
simple EEF motion.

Usage:
    python -m openpi.primitives.mock_controller --ip 127.0.0.1 --port 57770

Start the REPL driver FIRST (in --tcp-server mode), then run this script.
It will send one state+image frame, wait for the action trajectory, apply
simple position integration, and send the next frame.
"""

import argparse
import json
import logging
import socket
import struct
import sys
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fake image generator
# ---------------------------------------------------------------------------

def _make_fake_image(width: int = 224, height: int = 224) -> np.ndarray:
    """Generate a fake camera image (gray gradient)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        v = int(128 + 64 * np.sin(i / height * np.pi))
        img[i, :] = [v, v, v]
    return img


def _encode_jpeg(img: np.ndarray, quality: int = 80) -> bytes:
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Simple state simulation
# ---------------------------------------------------------------------------

class FakeRobotState:
    """Simple simulated EEF state that tracks position from received actions."""

    def __init__(self):
        # [x, y, z, roll, pitch, yaw, gripper] per arm
        self.left_pos = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        self.right_pos = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)

    def apply_action(self, trajectory: dict):
        """Move to the last waypoint in the trajectory."""
        follow1 = trajectory.get("follow1_pos", [])
        follow2 = trajectory.get("follow2_pos", [])
        if follow1:
            self.left_pos = np.array(follow1[-1], dtype=np.float32)
        if follow2:
            self.right_pos = np.array(follow2[-1], dtype=np.float32)

    def to_dict(self):
        return {
            "follow1_pos": self.left_pos.tolist(),
            "follow2_pos": self.right_pos.tolist(),
        }


# ---------------------------------------------------------------------------
# TCP client (connects to driver's TCP server)
# ---------------------------------------------------------------------------

def _send_frame(conn: socket.socket, state: dict):
    """Send one observation frame: JSON state + 3 JPEG images."""
    data = json.dumps(state).encode("utf-8")
    conn.sendall(struct.pack("<L", len(data)))
    conn.sendall(data)

    for _ in range(3):
        img_bytes = _encode_jpeg(_make_fake_image())
        conn.sendall(struct.pack("<L", len(img_bytes)))
        conn.sendall(img_bytes)


def _recv_trajectory(conn: socket.socket) -> dict | None:
    """Receive an action trajectory from the driver. Returns None if disconnected."""
    header = conn.recv(4)
    if not header or len(header) < 4:
        return None
    data_size = struct.unpack("<L", header)[0]
    buf = b""
    while len(buf) < data_size:
        chunk = conn.recv(data_size - len(buf))
        if not chunk:
            return None
        buf += chunk
    return json.loads(buf.decode("utf-8"))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mock ARX controller for testing")
    parser.add_argument("--ip", default="127.0.0.1", help="Driver TCP IP")
    parser.add_argument("--port", type=int, default=57770, help="Driver TCP port")
    parser.add_argument("--max-cycles", type=int, default=100, help="Max send/receive cycles")
    parser.add_argument("--step-mode", action="store_true",
                        help="Wait for user input between each cycle (for debugging)")
    args = parser.parse_args()

    robot = FakeRobotState()
    cycle = 0

    logger.info("connecting to driver at %s:%d ...", args.ip, args.port)
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    conn.settimeout(30.0)
    conn.connect((args.ip, args.port))
    logger.info("connected")

    try:
        while cycle < args.max_cycles:
            cycle += 1
            logger.info("=== cycle %d ===", cycle)

            # Send current state + fake images
            _send_frame(conn, robot.to_dict())
            logger.info("state sent: left=%s right=%s",
                         np.round(robot.left_pos, 3), np.round(robot.right_pos, 3))

            # Receive action trajectory
            trajectory = _recv_trajectory(conn)
            if trajectory is None:
                logger.error("driver disconnected")
                break
            follow1 = trajectory.get("follow1_pos", [])
            follow2 = trajectory.get("follow2_pos", [])
            logger.info("received trajectory: follow1=%d waypoints, follow2=%d waypoints",
                         len(follow1), len(follow2))
            if follow1:
                logger.info("  follow1[0]=%s  follow1[-1]=%s",
                             np.round(follow1[0], 3), np.round(follow1[-1], 3))

            # Apply action
            robot.apply_action(trajectory)
            logger.info("new state: left=%s right=%s",
                         np.round(robot.left_pos, 3), np.round(robot.right_pos, 3))

            if args.step_mode:
                input("Press Enter for next cycle...")
    except KeyboardInterrupt:
        logger.info("interrupted")
    except ConnectionError as e:
        logger.error("connection error: %s", e)
    finally:
        conn.close()
        logger.info("mock controller stopped after %d cycles", cycle)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [mock] %(message)s", datefmt="%H:%M:%S")
    main()
