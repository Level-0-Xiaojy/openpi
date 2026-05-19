#!/usr/bin/env python3
"""保存头部相机 RGB 和深度图到本地。

复用了 online_multi_class_head_camera.py 中的 RemoteHeadCameraReader 连接逻辑，
但剥离了所有 FoundationPose 推理代码，仅做采集与保存。

用法示例:
    python save_head_camera_frames.py --output ./captured_frames --max-frames 100
    python save_head_camera_frames.py --robot-host 192.168.120.155 --no-gui
"""

import argparse
import os
import signal
import socket
import struct
import threading
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import paramiko

# ════════════════════════════════════════════════════════════════
#  常量与默认配置（与 online_multi_class_head_camera.py 保持一致）
# ════════════════════════════════════════════════════════════════
FRAME_MAGIC = 0x44435732
TYPE_RGB = 1
TYPE_DEPTH = 2

ROBOT_HOST = "192.168.120.71"
# ROBOT_HOST = "192.168.120.155"
ROBOT_USER = "arm"
ROBOT_PASSWORD = "123"
DEPTH_SCALE = 1000.0
DEFAULT_CAM_K_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "assets", "cola", "cam_K.txt"
)
DEFAULT_DOCKER_CONTAINER = "robo_avatar_slave_dev_qyn"
# DEFAULT_DOCKER_CONTAINER = "robo_avatar_slave_dev"


# 远端脚本：在机器人上执行，订阅 ROS topic 并回传帧数据
REMOTE_SCRIPT = r"""
import argparse
import socket
import struct
import threading
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import cv2

FRAME_MAGIC = 0x44435732
TYPE_RGB = 1
TYPE_DEPTH = 2


def send_all(conn, data):
    view = memoryview(data)
    while view:
        sent = conn.send(view)
        if sent <= 0:
            raise ConnectionError("socket send failed")
        view = view[sent:]


def send_frame(conn, frame_type, fid, rows, cols, raw):
    header = struct.pack("<IIIIII", FRAME_MAGIC, frame_type, fid, rows, cols, len(raw))
    send_all(conn, header)
    send_all(conn, raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tcp-host", required=True)
    parser.add_argument("--tcp-port", type=int, required=True)
    parser.add_argument("--color-topic", default="/camera_dcw2/color/image_raw/compressed")
    parser.add_argument("--depth-topic", default="/camera_dcw2/depth/image_raw")
    parser.add_argument("--fast-stream", type=int, default=1)
    args = parser.parse_args()

    rospy.init_node("foundationpose_remote_head_stream", anonymous=True)

    bridge = CvBridge()
    rgb_frame = [None]
    depth_frame = [None]
    frame_id = [0]
    lock = threading.Lock()

    def rgb_cb(msg):
        try:
            if int(args.fast_stream) == 1:
                with lock:
                    rgb_frame[0] = bytes(msg.data)
                    frame_id[0] += 1
            else:
                arr = np.frombuffer(msg.data, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    with lock:
                        rgb_frame[0] = img
                        frame_id[0] += 1
        except Exception:
            pass

    def depth_cb(msg):
        try:
            depth = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if depth is not None:
                if depth.dtype == np.float32:
                    # 32FC1: 单位是米，需先乘 1000 转成毫米再发
                    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                    depth = (depth * 1000.0).astype(np.uint16)
                elif depth.dtype != np.uint16:
                    depth = depth.astype(np.uint16)
            with lock:
                depth_frame[0] = depth
        except Exception:
            pass

    rospy.Subscriber(args.color_topic, CompressedImage, rgb_cb, queue_size=1)
    rospy.Subscriber(args.depth_topic, Image, depth_cb, queue_size=1)

    conn = None
    while not rospy.is_shutdown() and conn is None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((args.tcp_host, args.tcp_port))
            sock.settimeout(None)
            conn = sock
        except Exception:
            rospy.sleep(0.5)

    if conn is None:
        return

    last_id = -1
    try:
        while not rospy.is_shutdown():
            with lock:
                fid = frame_id[0]
                if rgb_frame[0] is None:
                    rgb = None
                elif int(args.fast_stream) == 1:
                    rgb = rgb_frame[0]
                else:
                    rgb = rgb_frame[0].copy()
                depth = depth_frame[0].copy() if depth_frame[0] is not None else None
            if fid == last_id or rgb is None:
                rospy.sleep(0.005)
                continue
            last_id = fid
            if int(args.fast_stream) == 1:
                send_frame(conn, TYPE_RGB, fid, 0, 0, rgb)
            else:
                send_frame(conn, TYPE_RGB, fid, rgb.shape[0], rgb.shape[1], rgb.tobytes())
            if depth is not None:
                send_frame(conn, TYPE_DEPTH, fid, depth.shape[0], depth.shape[1], depth.tobytes())
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
"""


def _infer_local_ip(remote_host: str) -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((remote_host, 9))
        return sock.getsockname()[0]
    finally:
        sock.close()


class RemoteHeadCameraReader:
    """通过 SSH + TCP 从机器人头部相机流中获取 RGB 与深度图。"""

    def __init__(
        self,
        robot_host: str,
        robot_user: str,
        robot_password: str,
        tcp_port: int,
        color_topic: str,
        depth_topic: str,
        cam_k_path: str,
        depth_scale: float = 1000.0,
        local_host: Optional[str] = None,
        docker_container: str = DEFAULT_DOCKER_CONTAINER,
    ):
        self.robot_host = robot_host
        self.robot_user = robot_user
        self.robot_password = robot_password
        self.tcp_port = int(tcp_port)
        self.color_topic = color_topic
        self.depth_topic = depth_topic
        self.depth_scale = float(depth_scale)
        self.local_host = local_host or _infer_local_ip(robot_host)
        self.docker_container = docker_container

        if not os.path.isfile(cam_k_path):
            raise RuntimeError(f"相机内参文件不存在: {cam_k_path}")
        self.K = np.loadtxt(cam_k_path).reshape(3, 3)

        self._server_sock = None
        self._sock = None
        self._ssh_client = None
        self._ssh_stdin = None
        self._ssh_stdout = None
        self._ssh_stderr = None
        self._recv_thread = None
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._latest_rgb = None
        self._latest_depth = None
        self._latest_fid = -1
        self._last_served_fid = -1
        self._pending = {}
        self._running = False
        self.H = 0
        self.W = 0
        self.last_timing = {}

    def _recv_exact(self, n: int) -> bytes:
        data = b""
        while len(data) < n:
            chunk = self._sock.recv(n - len(data))
            if not chunk:
                raise ConnectionError("socket closed")
            data += chunk
        return data

    def _recv_loop(self):
        while self._running:
            try:
                header = self._recv_exact(24)
                magic, frame_type, fid, rows, cols, data_len = struct.unpack("<IIIIII", header)
                if magic != FRAME_MAGIC:
                    continue
                payload = self._recv_exact(data_len)
                with self._cv:
                    if frame_type == TYPE_RGB:
                        if rows > 0 and cols > 0:
                            rgb_bgr = np.frombuffer(payload, np.uint8).reshape(rows, cols, 3)
                        else:
                            rgb_bgr = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                            if rgb_bgr is None:
                                self._cv.notify_all()
                                continue
                        self._pending.setdefault(fid, {})["rgb"] = rgb_bgr
                    elif frame_type == TYPE_DEPTH:
                        depth = np.frombuffer(payload, np.uint16).reshape(rows, cols)
                        self._pending.setdefault(fid, {})["depth"] = depth
                    pair = self._pending.get(fid, {})
                    if "rgb" in pair and "depth" in pair:
                        self._latest_rgb = cv2.cvtColor(pair["rgb"], cv2.COLOR_BGR2RGB)
                        self._latest_depth = np.nan_to_num(
                            pair["depth"].astype(np.float32) / self.depth_scale,
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )
                        self._latest_fid = int(fid)
                        self.H, self.W = pair["rgb"].shape[:2]
                        old_keys = [k for k in self._pending.keys() if k < fid - 3]
                        for k in old_keys:
                            self._pending.pop(k, None)
                    self._cv.notify_all()
            except Exception:
                with self._cv:
                    self._running = False
                    self._cv.notify_all()
                break

    def start(self, timeout_sec: float = 20.0):
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind(("0.0.0.0", self.tcp_port))
        self._server_sock.listen(1)
        self._server_sock.settimeout(timeout_sec)

        remote_py_cmd = (
            f"python3 -u - --tcp-host {self.local_host} --tcp-port {self.tcp_port} "
            f"--color-topic '{self.color_topic}' --depth-topic '{self.depth_topic}' "
            f"--fast-stream 1"
        )
        remote_cmd = (
            "bash -lc '"
            "if python3 -c \"import rospy\" >/dev/null 2>&1; then "
            "  if [ -f /opt/ros/noetic/setup.bash ]; then source /opt/ros/noetic/setup.bash; fi; "
            "  if [ -f /home/arm/prj/turtle2/modules/devel/setup.bash ]; then source /home/arm/prj/turtle2/modules/devel/setup.bash; fi; "
            "  export ROS_MASTER_URI=${ROS_MASTER_URI:-http://localhost:11311}; "
            f"  {remote_py_cmd}; "
            f"elif docker inspect -f \"{{{{.State.Running}}}}\" {self.docker_container} >/dev/null 2>&1; then "
            f"  docker exec -i {self.docker_container} bash -lc \""
            "source /opt/ros/noetic/setup.bash; "
            "if [ -f /home/arm/prj/turtle2/modules/devel/setup.bash ]; then source /home/arm/prj/turtle2/modules/devel/setup.bash; fi; "
            "export ROS_MASTER_URI=http://localhost:11311; "
            f"{remote_py_cmd}"
            "\"; "
            "else "
            "  echo \"ERROR: rospy unavailable and docker ROS container not running\" 1>&2; "
            "  exit 2; "
            "fi'"
        )

        self._ssh_client = paramiko.SSHClient()
        self._ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self._ssh_client.connect(
            hostname=self.robot_host,
            username=self.robot_user,
            password=self.robot_password,
            timeout=10,
            look_for_keys=False,
            allow_agent=False,
        )
        self._ssh_stdin, self._ssh_stdout, self._ssh_stderr = self._ssh_client.exec_command(remote_cmd)
        self._ssh_stdin.write(REMOTE_SCRIPT)
        self._ssh_stdin.channel.shutdown_write()

        conn = None
        t_deadline = time.time() + timeout_sec
        self._server_sock.settimeout(1.0)
        while time.time() < t_deadline and conn is None:
            try:
                conn, _addr = self._server_sock.accept()
                break
            except socket.timeout:
                if self._ssh_stdout is not None and self._ssh_stdout.channel.exit_status_ready():
                    err_text = ""
                    try:
                        err_text = self._ssh_stderr.read().decode("utf-8", errors="ignore")
                    except Exception:
                        try:
                            err_text = self._ssh_stderr.read()
                        except Exception:
                            err_text = ""
                    raise RuntimeError(f"远端相机脚本提前退出: {err_text[:500]}")
                continue
        if conn is None:
            raise TimeoutError("等待机器人回连 TCP 超时，请检查机器人到本机网络、ROS topic 与本机 IP")
        conn.settimeout(None)
        self._sock = conn
        self._running = True
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

    def get_frames(self, frame_idx: int, require_depth: bool = True, timeout_sec: float = 5.0):
        _ = frame_idx
        t0 = time.perf_counter()
        t_deadline = time.time() + timeout_sec
        with self._cv:
            while self._running and self._latest_fid <= self._last_served_fid and time.time() < t_deadline:
                self._cv.wait(timeout=1.0)
            if not self._running and self._latest_fid <= self._last_served_fid:
                self.last_timing = {}
                return None, None
            while (
                self._running
                and require_depth
                and self._latest_depth is None
                and time.time() < t_deadline
            ):
                self._cv.wait(timeout=0.2)
            color = self._latest_rgb.copy() if self._latest_rgb is not None else None
            depth = self._latest_depth.copy() if self._latest_depth is not None else None
            self._last_served_fid = self._latest_fid
        t1 = time.perf_counter()
        self.last_timing = {"read.total": t1 - t0}
        return color, depth

    def close(self):
        self._running = False
        with self._cv:
            self._cv.notify_all()
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except Exception:
                pass
            self._server_sock = None
        if self._ssh_client is not None:
            try:
                if self._ssh_stdin is not None:
                    self._ssh_stdin.close()
            except Exception:
                pass
            try:
                if self._ssh_stdout is not None:
                    self._ssh_stdout.close()
            except Exception:
                pass
            try:
                if self._ssh_stderr is not None:
                    self._ssh_stderr.close()
            except Exception:
                pass
            try:
                self._ssh_client.close()
            except Exception:
                pass
            self._ssh_client = None


def visualize_depth(depth: np.ndarray, max_depth: float = 3.0) -> np.ndarray:
    """将 float32 深度图（米）转成 uint8 伪彩色图用于预览。"""
    disp = depth.copy()
    disp = np.clip(disp, 0.0, max_depth)
    disp = disp / max_depth
    disp = (disp * 255).astype(np.uint8)
    return cv2.applyColorMap(disp, cv2.COLORMAP_JET)


def main():
    parser = argparse.ArgumentParser(description="保存头部相机 RGB 和深度图")
    parser.add_argument("--robot-host", default=ROBOT_HOST, help="机器人 IP")
    parser.add_argument("--robot-user", default=ROBOT_USER, help="SSH 用户名")
    parser.add_argument("--robot-password", default=ROBOT_PASSWORD, help="SSH 密码")
    parser.add_argument("--local-host", default="", help="本机可被机器人访问的 IP，不填则自动推断")
    parser.add_argument("--tcp-port", type=int, default=19877, help="本机监听端口")
    parser.add_argument("--color-topic", default="/camera_dcw2/color/image_raw/compressed", help="RGB ROS topic")
    parser.add_argument("--depth-topic", default="/camera_dcw2/depth/image_raw", help="深度 ROS topic")
    parser.add_argument("--cam-k-path", default=DEFAULT_CAM_K_PATH, help="相机内参文件路径")
    parser.add_argument("--docker-container", default=DEFAULT_DOCKER_CONTAINER, help="ROS docker 容器名")
    parser.add_argument("--output", default="", help="保存目录，不填则自动生成 outputs/<timestamp>_capture")
    parser.add_argument("--max-frames", type=int, default=0, help="最大保存帧数，0 表示不限制（按 q 退出）")
    parser.add_argument("--no-gui", action="store_true", help="无预览窗口，后台静默保存")
    parser.add_argument("--depth-max-vis", type=float, default=3.0, help="深度可视化上限（米）")
    parser.add_argument("--save-vis", action="store_true", help="额外保存伪彩色深度可视化图（直观查看）")
    args = parser.parse_args()

    # 信号处理
    stopped = {"v": False}

    def _on_signal(_sig, _frm):
        stopped["v"] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # 输出目录
    if args.output:
        out_dir = args.output
    else:
        out_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "..",
            "outputs",
            datetime.now().strftime("%Y%m%d_%H%M%S") + "_capture",
        )
    rgb_dir = os.path.join(out_dir, "rgb")
    depth_dir = os.path.join(out_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    if args.save_vis:
        vis_dir = os.path.join(out_dir, "depth_vis")
        os.makedirs(vis_dir, exist_ok=True)

    print(f"[INFO] 输出目录: {out_dir}")
    print(f"[INFO] 连接 {args.robot_user}@{args.robot_host} ...")

    reader = RemoteHeadCameraReader(
        robot_host=args.robot_host,
        robot_user=args.robot_user,
        robot_password=args.robot_password,
        tcp_port=args.tcp_port,
        color_topic=args.color_topic,
        depth_topic=args.depth_topic,
        cam_k_path=args.cam_k_path,
        depth_scale=DEPTH_SCALE,
        local_host=args.local_host or None,
        docker_container=args.docker_container,
    )

    use_gui = (not args.no_gui) and bool(os.environ.get("DISPLAY"))
    if use_gui:
        cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)

    saved_count = 0
    try:
        reader.start()
        # 等待首帧
        color, depth = None, None
        for _ in range(15):
            color, depth = reader.get_frames(0, require_depth=True, timeout_sec=2.0)
            if color is not None and depth is not None:
                break
        if color is None or depth is None:
            print("[ERROR] 无法获取首帧，请检查 ROS topic 与网络连接")
            return

        print(f"[INFO] 图像尺寸: {color.shape[1]}x{color.shape[0]}, K=\n{reader.K}")
        print("[INFO] 开始采集，按 'q' 退出")

        frame_id = 0
        fps_t0 = time.perf_counter()
        fps_count = 0
        while not stopped["v"]:
            color, depth = reader.get_frames(frame_id, require_depth=True, timeout_sec=2.0)
            if color is None or depth is None:
                print("[WARN] 取流中断")
                break

            # 保存 RGB（PNG）
            rgb_path = os.path.join(rgb_dir, f"{frame_id:06d}.png")
            cv2.imwrite(rgb_path, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

            # 保存深度为 uint16 PNG（毫米），可直接用图片查看器/编辑器打开
            depth_uint16 = (depth * DEPTH_SCALE).astype(np.uint16)
            depth_path = os.path.join(depth_dir, f"{frame_id:06d}.png")
            cv2.imwrite(depth_path, depth_uint16)

            # 可选：额外保存伪彩色可视化图，更直观地看出远近层次
            if args.save_vis:
                depth_vis = visualize_depth(depth, max_depth=args.depth_max_vis)
                vis_path = os.path.join(vis_dir, f"{frame_id:06d}.png")
                cv2.imwrite(vis_path, depth_vis)

            saved_count += 1
            fps_count += 1
            elapsed = time.perf_counter() - fps_t0
            if elapsed >= 1.0:
                print(f"[INFO] 实际保存帧率: {fps_count / elapsed:.1f} FPS | 累计 {saved_count} 帧")
                fps_t0 = time.perf_counter()
                fps_count = 0

            if use_gui:
                bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                depth_vis = visualize_depth(depth, max_depth=args.depth_max_vis)
                cv2.imshow("RGB", bgr)
                cv2.imshow("Depth", depth_vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_id += 1
            if args.max_frames > 0 and saved_count >= args.max_frames:
                print(f"[INFO] 已达到最大帧数 {args.max_frames}，停止保存")
                break

    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        reader.close()
        if use_gui:
            cv2.destroyAllWindows()
        print(f"[INFO] 共保存 {saved_count} 帧到 {out_dir}")


if __name__ == "__main__":
    main()
