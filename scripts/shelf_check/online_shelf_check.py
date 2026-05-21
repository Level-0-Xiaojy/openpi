#!/usr/bin/env python3
import sys
import os
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

"""实时货架上货状态检测（头部相机 SSH 流）。

用法:
    python scripts/shelf_check/online_shelf_check.py --start outputs/.../depth/000500.png

交互按键:
    q  退出
    s  保存当前深度图
    r  开始/停止录制视频（录制完成后自动用 ffmpeg 合成 H.264 MP4）

依赖:
    录制视频需要系统安装 ffmpeg:
        sudo apt-get install ffmpeg
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime

import cv2
import numpy as np

from remote_camera import RemoteHeadCameraReader
from shelf_stock_checker import ShelfStockChecker, load_depth, pseudo_color

# 默认配置（和 online_multi_class_head_camera.py 保持一致）
ROBOT_HOST = "192.168.120.71"
ROBOT_USER = "arm"
ROBOT_PASSWORD = "123"
DEFAULT_CAM_K_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cam_K.txt")
DEFAULT_DOCKER_CONTAINER = "robo_avatar_slave_dev_qyn"
DEFAULT_REFERENCE_IMAGE_PATH = os.path.join(_project_root, "scripts", "shelf_check", "start.png")


def _encode_with_ffmpeg(frame_dir: str, output_path: str, fps: int = 17) -> bool:
    """用 ffmpeg 把 PNG 帧序列合成 H.264 MP4。"""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    cmd = [
        ffmpeg, "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frame_dir, "frame_%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        "-movflags", "+faststart",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def _write_status_atomic(path: str, payload: dict) -> None:
    """原子写入状态文件，避免读取端读到半截 JSON。"""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix=".shelf_status_", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    parser = argparse.ArgumentParser(description="实时货架上货状态检测（头部相机）")

    # ═══════════════════════════════════════════════════════════════
    #  相机连接参数
    # ═══════════════════════════════════════════════════════════════
    parser.add_argument("--robot-host", default=ROBOT_HOST, help="机器人 IP")
    parser.add_argument("--robot-user", default=ROBOT_USER, help="SSH 用户名")
    parser.add_argument("--robot-password", default=ROBOT_PASSWORD, help="SSH 密码")
    parser.add_argument("--tcp-port", type=int, default=19877, help="本机监听端口")
    parser.add_argument(
        "--color-topic",
        default="/camera_dcw2/color/image_raw/compressed",
        help="RGB ROS topic",
    )
    parser.add_argument(
        "--depth-topic",
        default="/camera_dcw2/depth/image_raw",
        help="深度 ROS topic",
    )
    parser.add_argument("--cam-k-path", default=DEFAULT_CAM_K_PATH, help="相机内参文件路径")
    parser.add_argument("--docker-container", default=DEFAULT_DOCKER_CONTAINER, help="ROS docker 容器名")

    # ═══════════════════════════════════════════════════════════════
    #  货架检测参数
    # ═══════════════════════════════════════════════════════════════
    parser.add_argument("--start", default=DEFAULT_REFERENCE_IMAGE_PATH, help="初始状态深度图路径（.png）")
    parser.add_argument("--row", type=int, default=305, help="start 图分析行号")
    parser.add_argument("--check-row", type=int, default=305, help="实时检测行号")
    parser.add_argument(
        "--start-valley-offset",
        type=float,
        default=0.02,
        help="start 谷底检测阈值偏移（米）",
    )
    parser.add_argument(
        "--check-valley-offset",
        type=float,
        default=0.08,
        help="实时检测谷底阈值偏移（米），越大越不容易漏检",
    )
    parser.add_argument("--min-width", type=int, default=40, help="谷底最小宽度（像素）")
    parser.add_argument("--median-size", type=int, default=5, help="中值滤波窗口大小")
    parser.add_argument("--smooth-window", type=int, default=15, help="平滑窗口大小")

    # ═══════════════════════════════════════════════════════════════
    #  显示参数
    # ═══════════════════════════════════════════════════════════════
    parser.add_argument("--no-gui", action="store_true", help="无预览窗口，仅终端打印")
    parser.add_argument(
        "--display-scale",
        type=float,
        default=0.6,
        help="预览窗口缩放比例",
    )
    parser.add_argument("--record-fps", type=int, default=17, help="录制视频帧率")
    parser.add_argument(
        "--status-file",
        default="/tmp/shelf_status.json",
        help="实时检测状态输出文件，供本机其他脚本读取",
    )

    args = parser.parse_args()

    # 检查 ffmpeg
    has_ffmpeg = shutil.which("ffmpeg") is not None
    if not has_ffmpeg:
        print("[WARN] 未检测到 ffmpeg，录制功能将不可用")
        print("       请安装: sudo apt-get install ffmpeg")

    # ─────────────────────────────────────────────────────────────
    # 创建带时间戳的输出目录
    # ─────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(_project_root, "outputs", "online_shelf_capture", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] 输出目录: {output_dir}")

    # ─────────────────────────────────────────────────────────────
    # 1. 初始化 ShelfStockChecker（从预保存的 start 图提取参考列）
    # ─────────────────────────────────────────────────────────────
    print(f"[INFO] 加载 start 参考图: {args.start}")
    start_depth = load_depth(args.start)

    checker = ShelfStockChecker(
        start_valley_offset=args.start_valley_offset,
        check_valley_offset=args.check_valley_offset,
        min_width=args.min_width,
        median_size=args.median_size,
        smooth_window=args.smooth_window,
    )
    n_cols, ref_valleys, bg_depth = checker.analyze_start(start_depth, row=args.row)
    print(f"[INFO] 参考列数: {n_cols}, 背景深度: {bg_depth:.3f}m")
    if n_cols == 0:
        print("[ERROR] start 图未检测到参考列，请检查 --start 和 --row 参数")
        return

    # ─────────────────────────────────────────────────────────────
    # 2. 连接头部相机
    # ─────────────────────────────────────────────────────────────
    print(f"[INFO] 连接相机 {args.robot_user}@{args.robot_host} ...")
    reader = RemoteHeadCameraReader(
        robot_host=args.robot_host,
        robot_user=args.robot_user,
        robot_password=args.robot_password,
        tcp_port=args.tcp_port,
        color_topic=args.color_topic,
        depth_topic=args.depth_topic,
        cam_k_path=args.cam_k_path,
        depth_scale=1000.0,
        docker_container=args.docker_container,
    )

    stopped = False

    def _on_signal(_sig, _frm):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    use_gui = (not args.no_gui) and bool(os.environ.get("DISPLAY"))
    disp_scale = args.display_scale

    # 录制 / 保存状态
    is_recording = False
    record_frame_dir = None
    record_frame_idx = 0
    record_count = 0   # 同一 session 内的录制序号
    save_count = 0     # 同一 session 内的手动保存序号

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

        print(f"[INFO] 图像尺寸: {color.shape[1]}x{color.shape[0]}")
        print("[INFO] 开始实时检测")
        print(f"[INFO] 状态文件: {args.status_file}")
        print("       按 'q' 退出 | 's' 保存深度图 | 'r' 开始/停止录制")

        frame_id = 0
        while not stopped:
            color, depth = reader.get_frames(frame_id, require_depth=True, timeout_sec=2.0)
            if color is None or depth is None:
                print("[WARN] 取流中断")
                break

            # ─────────────────────────────────────────────────────
            # 3. 实时检测当前帧
            # ─────────────────────────────────────────────────────
            is_full, filled_count, _, _, _, _ = checker.check_current(
                depth, check_row=args.check_row
            )
            status = "FULL" if is_full else "NOT FULL"
            _write_status_atomic(
                args.status_file,
                {
                    "is_full": bool(is_full),
                    "filled_count": int(filled_count),
                    "total_cols": int(n_cols),
                    "status": status,
                    "frame_id": int(frame_id),
                    "timestamp": time.time(),
                },
            )

            # 终端打印（每 30 帧或状态变化时打印，避免刷屏）
            if frame_id % 30 == 0 or frame_id == 0:
                print(f"Frame {frame_id:04d}: {filled_count}/{n_cols}  {status}")

            # ─────────────────────────────────────────────────────
            # 4. 可视化与交互
            # ─────────────────────────────────────────────────────
            if use_gui:
                # 左侧：RGB
                bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

                # 中间：深度伪彩色
                depth_vis = pseudo_color(depth, max_depth=max(depth.max(), 1.0))
                depth_vis_bgr = cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR)

                # 统一高度后水平拼接
                h = min(bgr.shape[0], depth_vis_bgr.shape[0])
                bgr = cv2.resize(bgr, (int(bgr.shape[1] * h / bgr.shape[0]), h))
                depth_vis_bgr = cv2.resize(
                    depth_vis_bgr, (int(depth_vis_bgr.shape[1] * h / depth_vis_bgr.shape[0]), h)
                )
                combined = np.concatenate([bgr, depth_vis_bgr], axis=1)

                # 叠加状态文字
                info = f"{filled_count}/{n_cols}  {status}"
                cv2.putText(combined, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 叠加分析行指示
                cv2.line(combined, (0, args.check_row), (combined.shape[1], args.check_row), (0, 0, 255), 1)

                # 叠加录制指示
                if is_recording:
                    cv2.circle(combined, (combined.shape[1] - 30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(combined, "REC", (combined.shape[1] - 80, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # 缩放显示
                if disp_scale != 1.0:
                    combined = cv2.resize(
                        combined,
                        (int(combined.shape[1] * disp_scale), int(combined.shape[0] * disp_scale)),
                        interpolation=cv2.INTER_AREA,
                    )

                cv2.imshow("Online Shelf Check", combined)

                # 如果正在录制，保存当前帧为 PNG
                if is_recording and record_frame_dir is not None:
                    frame_path = os.path.join(record_frame_dir, f"frame_{record_frame_idx:06d}.png")
                    cv2.imwrite(frame_path, combined)
                    record_frame_idx += 1

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    # 保存当前深度图到时间戳目录
                    save_count += 1
                    fname = os.path.join(output_dir, f"depth_{save_count:03d}.png")
                    depth_uint16 = (depth * 1000.0).astype(np.uint16)
                    cv2.imwrite(fname, depth_uint16)
                    print(f"[INFO] 已保存深度图 #{save_count}: {fname}")
                if key == ord("r"):
                    if not is_recording:
                        if not has_ffmpeg:
                            print("[ERROR] ffmpeg 未安装，无法录制。请执行: sudo apt-get install ffmpeg")
                        else:
                            # 开始录制：创建帧序列目录
                            record_frame_dir = os.path.join(output_dir, f"record_frames_{record_count:03d}")
                            os.makedirs(record_frame_dir, exist_ok=True)
                            is_recording = True
                            record_frame_idx = 0
                            record_count += 1
                            print(f"[INFO] 开始录制 #{record_count}，帧保存到: {record_frame_dir}")
                    else:
                        # 停止录制：用 ffmpeg 合成 MP4
                        is_recording = False
                        print(f"[INFO] 停止录制 #{record_count}，共 {record_frame_idx} 帧，正在合成视频...")
                        if record_frame_dir is not None and record_frame_idx > 0:
                            video_path = os.path.join(output_dir, f"record_{record_count:03d}.mp4")
                            ok = _encode_with_ffmpeg(record_frame_dir, video_path, fps=args.record_fps)
                            if ok:
                                print(f"[INFO] 视频已保存: {video_path}")
                                # 可选：删除临时帧目录以节省空间
                                # import shutil
                                # shutil.rmtree(record_frame_dir)
                            else:
                                print(f"[ERROR] ffmpeg 合成失败，原始帧保留在: {record_frame_dir}")
                        record_frame_dir = None

            frame_id += 1

    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        # 如果退出时仍在录制，尝试合成
        if is_recording and record_frame_dir is not None and record_frame_idx > 0:
            print("[INFO] 退出时正在录制，尝试合成视频...")
            video_path = os.path.join(output_dir, f"record_{record_count + 1:03d}.mp4")
            ok = _encode_with_ffmpeg(record_frame_dir, video_path, fps=args.record_fps)
            if ok:
                print(f"[INFO] 视频已保存: {video_path}")
            else:
                print(f"[ERROR] ffmpeg 合成失败，原始帧保留在: {record_frame_dir}")
        reader.close()
        cv2.destroyAllWindows()
        print("[INFO] 已断开相机连接")


if __name__ == "__main__":
    main()
