#!/usr/bin/env python3
"""批量检测货架上货状态，画出每帧上满列数的变化曲线，可选生成可视化视频。

用法:
    # 仅生成曲线图
    python batch_shelf_check.py \
        --start outputs/.../depth/000500.png \
        --depth-dir outputs/.../depth \
        --start-frame 500 --end-frame 2200 \
        --row 250 --check-row 310

    # 生成曲线图 + 可视化视频（左RGB、中深度、右曲线）
    python batch_shelf_check.py \
        --start outputs/.../depth/000500.png \
        --depth-dir outputs/.../depth \
        --rgb-dir outputs/.../rgb \
        --start-frame 500 --end-frame 2200 \
        --row 250 --check-row 310 \
        --vis --vis-output shelf_vis.mp4
"""

import argparse
import os
import sys
from io import BytesIO

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

from shelf_stock_checker import ShelfStockChecker, load_depth, pseudo_color

matplotlib.use("Agg")


def make_frame(rgb_img, depth_vis, curve_buf, target_h=480):
    """
    将 RGB 图、深度可视化图、曲线图拼接成一个画面。
    rgb_img:  (H, W, 3) uint8 RGB
    depth_vis: (H, W, 3) uint8 RGB
    curve_buf: BytesIO，包含 PNG 格式字节
    """
    # 读取曲线图
    curve_buf.seek(0)
    curve_arr = np.frombuffer(curve_buf.getvalue(), np.uint8)
    curve_img = cv2.imdecode(curve_arr, cv2.IMREAD_COLOR)
    if curve_img is None:
        curve_img = np.zeros((target_h, 640, 3), dtype=np.uint8)
    curve_img = cv2.cvtColor(curve_img, cv2.COLOR_BGR2RGB)

    # 统一高度
    def resize_to_h(img):
        if img.shape[0] != target_h:
            scale = target_h / img.shape[0]
            new_w = max(1, int(img.shape[1] * scale))
            return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
        return img

    rgb_img = resize_to_h(rgb_img)
    depth_vis = resize_to_h(depth_vis)
    curve_img = resize_to_h(curve_img)

    # 水平拼接
    frame = np.concatenate([rgb_img, depth_vis, curve_img], axis=1)
    return frame


def render_curve_to_buffer(frame_ids, filled_counts, filled_counts_smooth,
                           n_cols, current_idx, start_frame, end_frame,
                           target_w=640, target_h=480):
    """渲染到当前帧为止的曲线图，横坐标固定为 [start_frame, end_frame]，返回 BytesIO。"""
    buf = BytesIO()
    fig, ax = plt.subplots(figsize=(target_w / 100, target_h / 100), dpi=100)

    x = frame_ids[:current_idx + 1]
    y_raw = filled_counts[:current_idx + 1]
    y_smooth = filled_counts_smooth[:current_idx + 1]

    ax.plot(x, y_raw, color="steelblue", linewidth=0.8, alpha=0.4, label="Raw")
    ax.plot(x, y_smooth, color="steelblue", linewidth=2, marker="o", markersize=3, label="Median filtered")
    ax.axhline(y=n_cols, color="green", linestyle="--", linewidth=1.5, label=f"Full ({n_cols})")
    if len(y_smooth) > 0:
        ax.axhline(y=y_smooth.mean(), color="orange", linestyle="-.", linewidth=1, label=f"Avg ({y_smooth.mean():.2f})")

    # 标记当前帧
    ax.axvline(x=frame_ids[current_idx], color="red", linestyle=":", linewidth=1, alpha=0.7)

    ax.set_xlabel("Frame ID", fontsize=8)
    ax.set_ylabel("Filled Columns", fontsize=8)
    ax.set_title(f"Shelf Stock Curve | Frame {frame_ids[current_idx]}", fontsize=10)
    ax.set_xlim(start_frame, end_frame)
    ax.set_ylim(-0.2, n_cols + 0.5)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=7)

    plt.tight_layout(pad=0.5)
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return buf


def main():
    parser = argparse.ArgumentParser(description="批量货架上货状态检测")
    parser.add_argument("--root-dir", required=True, help="数据根目录（包含 depth/ 和 rgb/ 子目录）")
    parser.add_argument("--start", default="", help="初始状态深度图路径，不填则自动从 root-dir/depth/{start-frame}.png 获取")
    parser.add_argument("--depth-dir", default="", help="深度图目录，不填则自动 root-dir/depth")
    parser.add_argument("--rgb-dir", default="", help="RGB 图目录（用于可视化），不填则自动 root-dir/rgb")
    parser.add_argument("--start-frame", type=int, default=500, help="起始帧号")
    parser.add_argument("--end-frame", type=int, default=2200, help="结束帧号")
    parser.add_argument("--row", type=int, default=250, help="start 图分析行号")
    parser.add_argument("--check-row", type=int, default=310, help="current 图检测行号")
    parser.add_argument("--start-valley-offset", type=float, default=0.02)
    parser.add_argument("--check-valley-offset", type=float, default=0.10)
    parser.add_argument("--min-width", type=int, default=40)
    parser.add_argument("--median-size", type=int, default=5)
    parser.add_argument("--smooth-window", type=int, default=15)
    parser.add_argument("--median-window", type=int, default=10, help="filled_counts 中值滤波窗口大小")
    parser.add_argument("--vis", action="store_true", help="生成可视化视频")
    parser.add_argument("--vis-output", default="shelf_vis.mp4", help="可视化视频输出路径")
    parser.add_argument("--vis-fps", type=int, default=17, help="视频帧率")
    parser.add_argument("--save", default="", help="保存最终曲线图路径")
    args = parser.parse_args()

    # 自动推断路径
    root_dir = args.root_dir
    depth_dir = args.depth_dir if args.depth_dir else os.path.join(root_dir, "depth")
    rgb_dir = args.rgb_dir if args.rgb_dir else os.path.join(root_dir, "rgb")
    start_path = args.start if args.start else os.path.join(depth_dir, f"{args.start_frame:06d}.png")

    if args.vis and not os.path.isdir(rgb_dir):
        print(f"[ERROR] 使用 --vis 时 RGB 目录不存在: {rgb_dir}")
        sys.exit(1)

    # 1. 分析初始状态（只执行一次）
    print("=" * 60)
    print("Step 1: 分析初始状态")
    print("=" * 60)
    start_depth = load_depth(start_path)
    checker = ShelfStockChecker(
        start_valley_offset=args.start_valley_offset,
        check_valley_offset=args.check_valley_offset,
        min_width=args.min_width,
        median_size=args.median_size,
        smooth_window=args.smooth_window,
    )
    n_cols, ref_valleys, bg_depth = checker.analyze_start(start_depth, row=args.row)
    print(f"  参考列数: {n_cols}")
    print(f"  背景深度: {bg_depth:.3f} m")
    if n_cols == 0:
        print("[ERROR] 未检测到参考列，退出")
        sys.exit(1)

    # 2. 批量检测
    print("\n" + "=" * 60)
    print("Step 2: 批量检测")
    print("=" * 60)

    frame_ids = []
    filled_counts = []
    is_full_flags = []
    missing_frames = []

    for fid in range(args.start_frame, args.end_frame + 1):
        fname = f"{fid:06d}.png"
        fpath = os.path.join(depth_dir, fname)

        if not os.path.isfile(fpath):
            missing_frames.append(fname)
            continue

        try:
            current_depth = load_depth(fpath)
            is_full, filled_count, _, _, _, _ = checker.check_current(
                current_depth, check_row=args.check_row
            )
            frame_ids.append(fid)
            filled_counts.append(filled_count)
            is_full_flags.append(is_full)

            if fid % 100 == 0:
                status = "FULL" if is_full else "NOT FULL"
                print(f"  Frame {fid}: {filled_count}/{n_cols}  {status}")
        except Exception as e:
            print(f"  [WARN] Frame {fid} 处理失败: {e}")

    if missing_frames:
        print(f"\n[INFO] 缺失 {len(missing_frames)} 帧（已跳过）")

    if not frame_ids:
        print("[ERROR] 没有成功处理任何帧")
        sys.exit(1)

    # 3. 统计
    filled_counts = np.array(filled_counts, dtype=np.float32)
    if args.median_window >= 3 and len(filled_counts) >= args.median_window:
        filled_counts_smooth = median_filter(filled_counts, size=args.median_window)
    else:
        filled_counts_smooth = filled_counts.copy()

    full_rate = np.mean(is_full_flags) * 100
    print(f"\n统计结果:")
    print(f"  总帧数      : {len(frame_ids)}")
    print(f"  平均上满列数: {filled_counts_smooth.mean():.2f}/{n_cols}")
    print(f"  上满率      : {full_rate:.1f}%")
    print(f"  最小列数    : {int(filled_counts.min())}")
    print(f"  最大列数    : {int(filled_counts.max())}")

    # 4. 可视化视频
    if args.vis:
        print("\n" + "=" * 60)
        print("Step 3: 生成可视化视频")
        print("=" * 60)
        # 先拼一帧拿到尺寸，再初始化 VideoWriter
        test_rgb = cv2.imread(os.path.join(rgb_dir, f"{frame_ids[0]:06d}.png"), cv2.IMREAD_COLOR)
        if test_rgb is None:
            test_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        test_depth = pseudo_color(load_depth(os.path.join(depth_dir, f"{frame_ids[0]:06d}.png")), max_depth=1.0)
        test_curve = render_curve_to_buffer(frame_ids, filled_counts, filled_counts_smooth, n_cols, 0, args.start_frame, args.end_frame, target_w=640, target_h=480)
        test_frame = make_frame(cv2.cvtColor(test_rgb, cv2.COLOR_BGR2RGB), test_depth, test_curve, target_h=480)
        frame_h, frame_w = test_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.vis_output, fourcc, args.vis_fps, (frame_w, frame_h))

        for i, fid in enumerate(frame_ids):
            # 读取 RGB
            rgb_path = os.path.join(rgb_dir, f"{fid:06d}.png")
            if os.path.isfile(rgb_path):
                rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
                rgb_img = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            else:
                rgb_img = np.zeros((480, 640, 3), dtype=np.uint8)

            # 读取深度并生成伪彩色
            depth_path = os.path.join(depth_dir, f"{fid:06d}.png")
            depth = load_depth(depth_path)
            depth_vis = pseudo_color(depth, max_depth=max(depth.max(), 1.0))

            # 生成右侧曲线图
            curve_buf = render_curve_to_buffer(
                frame_ids, filled_counts, filled_counts_smooth,
                n_cols, i, args.start_frame, args.end_frame,
                target_w=640, target_h=480
            )

            # 拼接（RGB）
            frame_rgb = make_frame(rgb_img, depth_vis, curve_buf, target_h=480)
            # OpenCV 需要 BGR
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

            if (i + 1) % 50 == 0 or i == len(frame_ids) - 1:
                print(f"  已生成 {i + 1}/{len(frame_ids)} 帧")

        writer.release()
        print(f"视频已保存: {args.vis_output}")

    # 5. 画最终曲线图
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(frame_ids, filled_counts, color="steelblue", linewidth=0.8, alpha=0.4, label="Raw")
    ax.plot(frame_ids, filled_counts_smooth, color="steelblue", linewidth=2, marker="o", markersize=3, label="Median filtered")
    ax.axhline(y=n_cols, color="green", linestyle="--", linewidth=1.5, label=f"Full ({n_cols})")
    ax.axhline(y=filled_counts_smooth.mean(), color="orange", linestyle="-.", linewidth=1, label=f"Avg ({filled_counts_smooth.mean():.2f})")

    full_frames = [fid for fid, full in zip(frame_ids, is_full_flags) if full]
    if full_frames:
        ax.scatter(full_frames, [n_cols] * len(full_frames), color="green", s=15, marker="|", zorder=5, label="Full frame")

    ax.set_xlabel("Frame ID", fontsize=12)
    ax.set_ylabel("Filled Columns", fontsize=12)
    ax.set_title(
        f"Shelf Stock Batch Check | StartRow={args.row} CheckRow={args.check_row} | FullRate={full_rate:.1f}%",
        fontsize=14,
    )
    ax.set_ylim(-0.2, n_cols + 0.5)
    ax.set_xticks(np.arange(args.start_frame, args.end_frame + 1, 100))
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)

    for i in range(0, len(frame_ids), max(1, len(frame_ids) // 20)):
        ax.text(frame_ids[i], filled_counts[i] + 0.15, str(int(filled_counts[i])),
                fontsize=7, ha="center", color="steelblue")

    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"\n曲线图已保存: {args.save}")
    plt.show()


if __name__ == "__main__":
    main()
