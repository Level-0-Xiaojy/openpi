#!/usr/bin/env python3
import sys
sys.path.append('/home/arm/prj/turtle2/modules/src/model2robot/scripts')
sys.path.append('/home/arm/prj/turtle2/modules/src/turtle2_controller')

# import rospkg
# rospack = rospkg.RosPack()
# package_path = rospack.get_path('turtle2_controller')
# sys.path.append(os.path.join(rospack_path, 'turtle2_controller'))
from turtle2_controller.Turtle2Controller import Turtle2Controller

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
import time
import json
import socket
import struct
import argparse
import signal
import threading
from collections import deque

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

import gradio as gr
import plotly.graph_objects as go

def interpolates_actions(actions, factor):
    """
    actions_shape: [num_actions, action_dim] (其中 action_dim 包含 [x, y, z, roll, pitch, yaw, gripper])
    """
    num_actions, action_dim = actions.shape
    target_num_actions = factor * (num_actions - 1) + 1
   
    original_indices = np.linspace(0, num_actions - 1, num_actions)
    target_indices = np.linspace(0, num_actions - 1, target_num_actions)
    interpolated_actions = np.zeros((target_num_actions, action_dim))

    # 对[x, y, z, gripper]使用线性插值
    for i in range(3):
        interpolated_actions[:, i] = np.interp(target_indices, original_indices, actions[:, i])
    interpolated_actions[:, -1] = np.interp(target_indices, original_indices, actions[:, -1])
    
    # 将欧拉角转换为四元数, 对四元数进行球面插值
    quaternions = R.from_euler('xyz', actions[:, 3:6]).as_quat()  # shape: [num_actions, 4]
    interpolated_quats = np.zeros((target_num_actions, 4))
    for i in range(4):  # 对四元数的每个分量进行插值
        interpolated_quats[:, i] = np.interp(target_indices, original_indices, quaternions[:, i])
    interpolated_quats = interpolated_quats / np.linalg.norm(interpolated_quats, axis=1, keepdims=True)
    interpolated_eulers = R.from_quat(interpolated_quats).as_euler('xyz')  # shape: [target_num_actions, 3]
    interpolated_actions[:, 3:6] = interpolated_eulers
    
    return interpolated_actions

class ArmSlaveRos():
    ''' Arm Slave Ros Class
    '''
    def __init__(
            self, 
            nodeName="ArmSlaveNode", 
            enable_gradio=False, 
            state_history_size=0,
            state_future_size=0,
            state_step=1,
            actions_factor=3,
            mask_history_slave_states=False,
            # start edit by qyn 2026-02-26
            enable_video_stream=False,
            video_server_ip=None,
            video_server_port=57771,
            video_fps=30.0,
            # end edit
        ):
        # 禁用默认 SIGINT 处理，使用自定义两阶段退出
        rospy.init_node(nodeName, disable_signals=True)
        self.bridge = CvBridge()
        self.controller = Turtle2Controller()
        self.rate1 = rospy.Rate(1)  # 1Hz
        self.rate100 = rospy.Rate(100)  # 100Hz
        self.enable_gradio = enable_gradio
        self.state_history_size = state_history_size
        self.state_future_size = state_future_size
        self.state_step = state_step
        self.actions_factor = actions_factor
        self.mask_history_slave_states = mask_history_slave_states
        
        self._sigint_count = 1 + int(enable_gradio)  # Gradio 模式下首次 Ctrl+C 不退出

        # 动作执行队列与同步原语
        self.action_queue = deque()
        self.obs_queue = deque(maxlen=100)  # 跟 VLA 动作同频率存储数据, deque of dict
        self._queue_lock = threading.Lock()
        self._fetch_lock = threading.Lock()  # 控制仅一次外部请求
        self._fetching = False
        self._fetch_tick = actions_factor * state_future_size * state_step  # 当 action_queue 长度等于此值时触发获取
        self._first_recv = True

        self._server_ip = None
        self._server_port = None
        self._socket_connected = False
        self._running_mode_prev = rospy.get_param("/running_mode", 1)

        # start edit by qyn 2026-02-26
        # 独立视频推流通道：可选，将三路压缩图像通过第二个 TCP 连接推送到 5090，用于遥操观看
        self._video_enabled = enable_video_stream
        self._video_server_ip = video_server_ip
        self._video_server_port = video_server_port
        self._video_fps = video_fps
        self._video_sock = None
        self._video_stop = False
        self._video_lock = threading.Lock()
        self._video_img1 = None
        self._video_img2 = None
        self._video_img3 = None

        if self._video_enabled:
            rospy.loginfo(
                "Video stream enabled: target=%s:%s fps=%.1f",
                self._video_server_ip,
                self._video_server_port,
                self._video_fps,
            )
            # 订阅三路压缩图像 topic
            self._video_sub1 = rospy.Subscriber(
                "/camera1/usb_cam1/image_raw/compressed",
                CompressedImage,
                self._video_cb1,
                queue_size=1,
            )
            self._video_sub2 = rospy.Subscriber(
                "/camera2/usb_cam2/image_raw/compressed",
                CompressedImage,
                self._video_cb2,
                queue_size=1,
            )
            self._video_sub3 = rospy.Subscriber(
                "/camera3/usb_cam3/image_raw/compressed",
                CompressedImage,
                self._video_cb3,
                queue_size=1,
            )
            
            # 启动独立推流线程
            self._video_thread = threading.Thread(
                target=self._video_loop, daemon=True
            )
            self._video_thread.start()
        # end edit

        # 定时器占位（在 start_async 中创建）
        self._exec_timer = None
        self._filler_timer = None

        if enable_gradio:
            # 可视化缓冲：分离目标与实测，按 step 轴存储
            self._target_buf = deque(maxlen=1000)  # (step, left_target, right_target)
            self._actual_buf = deque(maxlen=1000)  # (step, left_actual, right_actual)
            self._step_target = 0
            self._step_actual = 0
            self._chunk_steps = deque(maxlen=50)  # step positions for action chunks
            self._viz_lock = threading.Lock()

            # 启动 Gradio 界面（非阻塞）
            threading.Thread(target=self._launch_gradio, daemon=True).start()

        self.rate1.sleep()
        self.controller.chassis_set_current_pose_as_virtual_zero()

        # 设置 SIGINT 处理：第一次仅停定时器，第二次退出
        signal.signal(signal.SIGINT, self._handle_sigint)

    # start edit by qyn 2026-02-26
    # ===== 视频推流相关：三路压缩图像独立 socket 通道 =====
    def _video_cb1(self, msg):
        with self._video_lock:
            self._video_img1 = msg

    def _video_cb2(self, msg):
        with self._video_lock:
            self._video_img2 = msg

    def _video_cb3(self, msg):
        with self._video_lock:
            self._video_img3 = msg

    def _preview_raw_cb1(self, msg):
        """仅用于本地预览的 image_raw 回调（不参与网络推流）。"""
        with self._video_lock:
            self._preview_raw1 = msg

    def _video_connect(self):
        """建立或重建视频推流 socket 连接。"""
        if not self._video_enabled:
            return
        if self._video_sock is not None:
            return
        if self._video_server_ip is None or self._video_server_port is None:
            rospy.logwarn("VideoStreamer: server ip/port not set, skip connect")
            return
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5.0)
            s.connect((self._video_server_ip, self._video_server_port))
            self._video_sock = s
            rospy.loginfo(
                "VideoStreamer: connected to %s:%s",
                self._video_server_ip,
                self._video_server_port,
            )
        except Exception as e:
            rospy.logwarn(
                # "VideoStreamer: connect to %s:%s failed: %s",
                self._video_server_ip,
                self._video_server_port,
                e,
            )
            self._video_sock = None

    def _video_send_frame_bytes(self, data1, data2, data3):
        """底层发送函数：直接按 [len1][data1][len2][data2][len3][data3] 发送原始字节。"""
        try:
            if self._video_sock is None:
                # 允许在仅本地预览时跳过发送
                return
            l1 = struct.pack("<L", len(data1))
            l2 = struct.pack("<L", len(data2))
            l3 = struct.pack("<L", len(data3))
            self._video_sock.sendall(l1); self._video_sock.sendall(data1)
            self._video_sock.sendall(l2); self._video_sock.sendall(data2)
            self._video_sock.sendall(l3); self._video_sock.sendall(data3)
        except Exception as e:
            rospy.logwarn("VideoStreamer: send_frame_bytes failed: %s", e)
            if self._video_sock is not None:
                try:
                    self._video_sock.close()
                except Exception:
                    pass
            self._video_sock = None

    def _video_send_frame(self, img1, img2, img3):
        """按 [len1][data1][len2][data2][len3][data3] 协议发送三路压缩图像。"""
        try:
            def _bytes_from_msg(msg):
                return bytes(msg.data)

            d1 = _bytes_from_msg(img1)
            d2 = _bytes_from_msg(img2)
            d3 = _bytes_from_msg(img3)
            self._video_send_frame_bytes(d1, d2, d3)
        except Exception as e:
            rospy.logwarn("VideoStreamer: send_frame failed: %s", e)

    def _video_loop(self):
        """独立线程：以固定 FPS 推送最新三路压缩图像到 5090。"""
        rate = rospy.Rate(self._video_fps)
        while not rospy.is_shutdown() and not self._video_stop:
            if not self._video_enabled:
                rate.sleep()
                continue

            if self._video_sock is None:
                self._video_connect()
                rate.sleep()
                continue

            with self._video_lock:
                img1 = self._video_img1
                img2 = self._video_img2
                img3 = self._video_img3
                # preview_raw1 = self._preview_raw1

            if img1 is None or img2 is None or img3 is None:
                rate.sleep()
                continue
            
            else:
                self._video_send_frame(img1, img2, img3)
            rate.sleep()
    # end edit

    def setRos2Socket(self, TCP_IP='192.168.1.53', TCP_PORT=57750):
        self._server_ip = TCP_IP
        self._server_port = TCP_PORT
        self._connect_socket()

    def _connect_socket(self):
        if self._socket_connected:
            return
        if self._server_ip is None or self._server_port is None:
            rospy.logwarn("Server IP/port not set, cannot connect socket")
            return
        try:
            self.socket_ = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket_.setblocking(True) # communication blocking
            self.socket_.connect((self._server_ip, self._server_port))
            time.sleep(1)
            self._socket_connected = True
            rospy.loginfo("Socket connected to %s:%s", self._server_ip, self._server_port)
        except Exception as exc:
            rospy.logwarn("Socket connect failed: %s", exc)
            self._socket_connected = False

    def _disconnect_socket(self):
        if not self._socket_connected:
            return
        try:
            self.socket_.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            self.socket_.close()
        except Exception:
            pass
        self._socket_connected = False
        rospy.loginfo("Socket disconnected")

    def _handle_running_mode_transition(self, current_mode):
        prev_mode = self._running_mode_prev
        if prev_mode != 1 and current_mode == 1:
            rospy.loginfo("Running mode switched to 1, reconnecting socket")
            self._disconnect_socket()
            self._connect_socket()
            self.action_queue.clear()
            self.obs_queue.clear()
            self._first_recv = True
        elif prev_mode == 1 and current_mode != 1:
            rospy.loginfo("Running mode left 1, disconnecting socket")
            self._disconnect_socket()
        self._running_mode_prev = current_mode

    def recv_all(self, count):
        print('count:', count)
        buf = b''
        while count:
            newbuf = self.socket_.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf
    
    def SendDict(self, data):
        # 判断是不是字典
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")
        data_str = json.dumps(data)
        data_bytes = data_str.encode('utf-8')
        self.socket_.sendall(struct.pack('<L', len(data_bytes)))
        self.socket_.sendall(data_bytes)
    
    def RecvDict(self):
        data_size = self.socket_.recv(4)
        if not data_size:
            return None
        data_size = struct.unpack('<L', data_size)[0]
        
        data = self.recv_all(data_size)
        if data is None: return None
        data_str = data.decode('utf-8')
        return json.loads(data_str)

    def SendImage(self, ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        image_bytes = cv2.imencode('.jpg', cv_image)[1].tobytes()
        image_size = len(image_bytes)
        self.socket_.sendall(struct.pack('<L', image_size))
        self.socket_.sendall(image_bytes)

    # ===== 异步运行：60Hz 执行器 + 120Hz 队列补充器 =====
    def start_async(self):
        # 60Hz 执行动作
        self._exec_timer = rospy.Timer(rospy.Duration(1.0/60.0), self._exec_cb)
        # 120Hz 检测队列并向 VLA 请求动作
        self._filler_timer = rospy.Timer(rospy.Duration(1.0/120.0), self._filler_cb)
        # 自定义 SIGINT：首次 Ctrl+C 停定时器保持 Gradio，第二次退出
        while self._sigint_count > 0 and not rospy.is_shutdown():
            time.sleep(0.2)

    def _handle_sigint(self, signum, frame):
        self._sigint_count -= 1
        if self._sigint_count == 1:
            rospy.loginfo("Ctrl+C detected: stopping timers, Gradio stays up. Press Ctrl+C again to exit.")
            self._exec_timer.shutdown()
            self._filler_timer.shutdown()
            # start edit by qyn 2026-02-26
            # 同时停止视频推流线程
            self._video_stop = True
            if getattr(self, "_video_sock", None) is not None:
                try:
                    self._video_sock.close()
                except Exception:
                    pass
                self._video_sock = None
            # end edit

    def _exec_cb(self, event):
        # 每次 tick 执行一个动作（若有）
        action = None
        save_obs = (len(self.action_queue) % self.actions_factor == 1) if self.actions_factor > 1 else True # 跟 VLA 动作同频率存储数据 
        if save_obs:
            slave1EndPos, slave2EndPos = self.controller.arms_data()
                
        with self._queue_lock:
            # 同步观测和执行
            if save_obs:
                self.obs_queue.append({
                    "follow1_pos": slave1EndPos,
                    "follow2_pos": slave2EndPos,
                })
            if self.action_queue:
                action = self.action_queue.popleft()
        
        if rospy.get_param("/running_mode", 1) != 1:
            return
        
        if action is None:
            if not self._first_recv and self.state_future_size > 0:
                rospy.logwarn("Action queue empty at exec tick")
            return

        left, right, head, lift, car = action
        self.controller.arms_control(left, right)
        if head is not None:
            self.controller.head_control([head[1], head[0]])
        if lift is not None:
            self.controller.lift_control(lift)
        if car is not None:
            self.controller.chassis_control_relative_pose(car)

        if self.enable_gradio:
            actual_left, actual_right = self.controller.arms_data()
            self._push_actual_sample(actual_left=actual_left, actual_right=actual_right)

    def _filler_cb(self, event):
        current_mode = rospy.get_param("/running_mode", 1)
        self._handle_running_mode_transition(current_mode)
        if current_mode != 1 or not self._socket_connected:
            print('current_mode:', current_mode, 'not self._socket_connected:', self._socket_connected)
            return 
        
        # 若队列长度不足，尝试触发一次获取（放到后台线程，避免阻塞）
        with self._queue_lock:
            qlen = len(self.action_queue)
        if qlen > self._fetch_tick:
            return

        # 控制仅一次并发获取
        with self._fetch_lock:
            if self._fetching:
                return
            self._fetching = True
        threading.Thread(target=self._fetch_and_enqueue, daemon=True).start()

    def _get_obs(self):
        """
        从 self.obs_queue 中获取最近 state_history_size + 1 帧观测，如果不足则用最早的帧补齐。
        """
        obs = {
            "follow1_pos": [],
            "follow2_pos": [],
        }

        with self._queue_lock:
            if not self._first_recv:
                if len(self.action_queue) < self._fetch_tick:
                    rospy.logwarn(f"Obs fetch: action queue length below threshold, {len(self.action_queue)} < {self._fetch_tick}")
                obs_list = list(self.obs_queue)[-(self.state_history_size * self.state_step + 1)::self.state_step]
                if self.mask_history_slave_states:
                    obs_list = obs_list[-1:] 
            else:
                slave1EndPos, slave2EndPos = self.controller.arms_data()
                obs_list = [{
                    "follow1_pos": slave1EndPos,
                    "follow2_pos": slave2EndPos,
                }]

        if len(obs_list) < self.state_history_size + 1:
            obs_list = [obs_list[0]] * (self.state_history_size + 1 - len(obs_list)) + obs_list

        for o in obs_list:
            obs["follow1_pos"].append(o["follow1_pos"])
            obs["follow2_pos"].append(o["follow2_pos"])

        return obs

    def _fetch_and_enqueue(self):
        try:
            obs = self._get_obs()
            self.SendDict(obs)

            # 发送多路相机
            image1 = self.controller.cam.image1Data
            image2 = self.controller.cam.image2Data
            image3 = self.controller.cam.image3Data
            self.SendImage(image1)
            self.SendImage(image2)
            self.SendImage(image3)

            # 接收 VLA 指令序列
            cmd = self.RecvDict()
            if cmd is None:
                rospy.logwarn("Received empty command from web server")
                return

            d1 = cmd["follow1_pos"]
            d2 = cmd["follow2_pos"]

            d1 = interpolates_actions(np.array(d1), self.actions_factor)
            d2 = interpolates_actions(np.array(d2), self.actions_factor)
            d1 = d1[1:]
            d2 = d2[1:]
            # todo: 插值至所需动作数

            for k in ["head_pos", "lift", "car_pose"]:
                if k not in cmd:
                    cmd[k] = [None] * len(d1)
            d3 = cmd["head_pos"]
            d4 = cmd["lift"]
            d5 = cmd["car_pose"]

            # 可视化：写入目标批次
            if self.enable_gradio:
                self._push_target_batch(d1, d2)

            # 入队
            new_items = []
            for l, r, h, lf, c in zip(d1, d2, d3, d4, d5):
                new_items.append((l, r, h, lf, c))
            with self._queue_lock:
                self.action_queue.extend(new_items)

            self._first_recv = False
            
        except (socket.error, socket.timeout, OSError, ConnectionError) as e:
            rospy.logwarn("Socket error in fetch_and_enqueue: %s. Will reconnect.", e)
            self._socket_connected = False
        except Exception as e:
            rospy.logerr("Unexpected error in fetch_and_enqueue: %s", e)
        finally:
            # 确保 _fetching 标志总是被重置
            with self._fetch_lock:
                self._fetching = False

    # ===== 可视化相关 =====
    def _push_target_batch(self, target_left_list, target_right_list):
        with self._viz_lock:
            start = max(self._step_target, self._step_actual)
            chunk_start = start + 1
            for idx, (lt, rt) in enumerate(zip(target_left_list, target_right_list)):
                step = start + idx + 1
                self._target_buf.append((step, lt, rt))
                self._step_target = step
            # 标记新的 action chunk 起点
            if not self._chunk_steps or self._chunk_steps[-1] != chunk_start:
                self._chunk_steps.append(chunk_start)

    def _push_actual_sample(self, actual_left, actual_right):
        with self._viz_lock:
            self._step_actual += 1
            step = self._step_actual
            self._actual_buf.append((step, actual_left, actual_right))

    def _launch_gradio(self):
        with gr.Blocks(title="Arm Live Plot") as demo:
            gr.Markdown("## 双臂实时执行 vs 目标 (step 轴)")
            with gr.Row(variant='compact'):
                with gr.Column(variant='compact'):
                    plot_left_xyz = gr.Plot(label="Left Arm XYZ")
                    plot_left_rpy = gr.Plot(label="Left Arm RPY")
                with gr.Column(variant='compact'):
                    plot_right_xyz = gr.Plot(label="Right Arm XYZ")
                    plot_right_rpy = gr.Plot(label="Right Arm RPY")
            def build_figures():
                window_len = 1000
                sample_k = 2  # 采样间隔，减少绘图数据量
                with self._viz_lock:
                    tgt_data = list(self._target_buf)
                    act_data = list(self._actual_buf)
                    chunk_marks = list(self._chunk_steps)
                if not tgt_data and not act_data:
                    return [go.Figure() for _ in range(4)]
                
                def _normalize(buf):
                    steps = []
                    vals = []
                    for step, data in buf:
                        steps.append(float(step))
                        row = [None] * 7
                        for j, v in enumerate(list(data)[:7]):
                            row[j] = float(v)
                        vals.append(row)
                    return steps, vals

                tgt_steps, tgt_l = _normalize([(s, l) for s, l, _ in tgt_data]) if tgt_data else ([], [])
                _, tgt_r = _normalize([(s, r) for s, _, r in tgt_data]) if tgt_data else ([], [])
                act_steps, act_l = _normalize([(s, l) for s, l, _ in act_data]) if act_data else ([], [])
                _, act_r = _normalize([(s, r) for s, _, r in act_data]) if act_data else ([], [])

                # x 轴窗口：默认 0-window_len，超过则显示最近 window_len 个 step
                max_step = max(max(act_steps) if act_steps else 0, max(tgt_steps))
                x0 = max(0, max_step - window_len)
                x1 = max(window_len, max_step)

                def fig_three(act_vals, tgt_vals, act_steps_in, tgt_steps_in, labels, title):
                    fig = go.Figure()
                    colors = {"x": "red", "y": "green", "z": "blue", "roll": "red", "pitch": "green", "yaw": "blue"}
                    light_colors = {"x": "salmon", "y": "lightgreen", "z": "lightskyblue", 
                                    "roll": "salmon", "pitch": "lightgreen", "yaw": "lightskyblue"}
                    for x in chunk_marks:
                        fig.add_vline(x=x, line=dict(color="white", width=1))
                    for idx, key in enumerate(labels):
                        if act_steps_in:
                            y_act = [row[idx] for row in act_vals]
                            fig.add_trace(go.Scatter(x=act_steps_in, y=y_act, mode="lines", name=f"{key} actual", line=dict(color=colors.get(key, "black"))))
                        if tgt_steps_in:
                            y_tgt = [row[idx] for row in tgt_vals]
                            fig.add_trace(go.Scatter(x=tgt_steps_in, y=y_tgt, mode="lines", name=f"{key} target", line=dict(color=light_colors.get(key, "gray"))))
                            fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    fig.update_layout(xaxis=dict(range=[x0, x1], showgrid=False))
                    return fig

                fig_l_xyz = fig_three([row[:3] for row in act_l], [row[:3] for row in tgt_l], act_steps, tgt_steps, ["x", "y", "z"], "Left XYZ")
                fig_l_rpy = fig_three([row[3:6] for row in act_l], [row[3:6] for row in tgt_l], act_steps, tgt_steps, ["roll", "pitch", "yaw"], "Left RPY")

                fig_r_xyz = fig_three([row[:3] for row in act_r], [row[:3] for row in tgt_r], act_steps, tgt_steps, ["x", "y", "z"], "Right XYZ")
                fig_r_rpy = fig_three([row[3:6] for row in act_r], [row[3:6] for row in tgt_r], act_steps, tgt_steps, ["roll", "pitch", "yaw"], "Right RPY")       
                      
                return [fig_l_xyz, fig_l_rpy, fig_r_xyz, fig_r_rpy]

            # 自动刷新，每 0.5s (2Hz)
            demo.queue(concurrency_count=1)
            demo.load(
                fn=build_figures,
                inputs=None,
                outputs=[
                    plot_left_xyz, plot_left_rpy,
                    plot_right_xyz, plot_right_rpy,
                ],
                every=0.5,
            )

        demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True, prevent_thread_lock=True)

# test 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Arm socket bridge")
    parser.add_argument("--enable-gradio", action="store_true", help="Enable Gradio visualization dashboard")
    parser.add_argument("--server-ip", type=str, default="192.168.110.16", help="VLA server IP")
    parser.add_argument("--server-port", type=int, default=57770, help="VLA server port")
    parser.add_argument("--mask-history-slave-states", action="store_true", help="Mask history slave states in observation")
    parser.add_argument("--state-history-size", type=int, default=0, help="Number of historical states to include in observation")
    parser.add_argument("--state-future-size", type=int, default=0, help="Number of future states to predict")
    parser.add_argument("--state-step", type=int, default=1, help="Step size between states in observation")
    parser.add_argument("--actions-factor", type=int, default=3, help="Action interpolation factor")
    # start edit by qyn 2026-02-26
    parser.add_argument(
        "--enable-video-stream",
        action="store_true",
        help="Enable independent video streaming socket",
    )
    parser.add_argument(
        "--video-server-ip",
        type=str,
        default=None,
        help="Video stream server IP (default: same as server-ip)",
    )
    parser.add_argument(
        "--video-server-port",
        type=int,
        default=57771,
        help="Video stream server port",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=15.0,
        help="Video streaming FPS",
    )
    # end edit
    args = parser.parse_args()

    arm = ArmSlaveRos(
        enable_gradio=args.enable_gradio, 
        mask_history_slave_states=args.mask_history_slave_states, 
        state_history_size=args.state_history_size,
        state_future_size=args.state_future_size,
        state_step=args.state_step,
        actions_factor=args.actions_factor,
        # start edit by qyn 2026-02-26
        enable_video_stream=args.enable_video_stream,
        video_server_ip=args.video_server_ip or args.server_ip,
        video_server_port=args.video_server_port,
        video_fps=args.video_fps,
        # end edit
    )
    arm.setRos2Socket(args.server_ip, args.server_port)
    arm.start_async()

