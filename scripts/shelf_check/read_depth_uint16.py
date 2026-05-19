#!/usr/bin/env python3
"""读取 16-bit uint16 PNG 深度图，左侧显示伪彩色深度图并标注指定行，
右侧绘制该行深度变化曲线。

用法示例:
    python read_depth_uint16.py depth/000000.png
    python read_depth_uint16.py depth/000000.png --row 300
    python read_depth_uint16.py depth/000000.png --save result.png
"""

import argparse
import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TkAgg")
DEPTH_SCALE = 1000.0


def pseudo_color(depth_m: np.ndarray, max_depth: float = 3.0) -> np.ndarray:
    """将 float32 深度图（米）转成 uint8 伪彩色 RGB 图。"""
    disp = np.clip(depth_m, 0.0, max_depth)
    disp = disp / max_depth if max_depth > 0 else disp
    disp = (disp * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser(description="读取并可视化 uint16 深度图")
    parser.add_argument("input", help="深度图文件路径 (.png)")
    parser.add_argument("--row", type=int, default=-1, help="要绘制的行号，-1 表示图像中间行")
    parser.add_argument("--max-depth", type=float, default=3.0, help="伪彩色映射深度上限（米）")
    parser.add_argument("--threshold", type=float, default=-1.0, help="深度阈值（米），-1 表示不显示")
    parser.add_argument("--save", default="", help="保存结果图路径")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[ERROR] 文件不存在: {args.input}")
        return

    # 1. 读取 uint16 深度图（必须加 UNCHANGED，否则默认转 8-bit 会丢失精度）
    depth_uint16 = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if depth_uint16 is None:
        print(f"[ERROR] 无法读取: {args.input}")
        return

    if depth_uint16.dtype != np.uint16:
        print(f"[WARN] 数据类型为 {depth_uint16.dtype}，预期为 uint16")

    # 2. 转回 float32（米）
    depth_m = depth_uint16.astype(np.float32) / DEPTH_SCALE
    H, W = depth_m.shape

    # 3. 确定要分析的行
    row = args.row if args.row >= 0 else H // 2
    row = max(0, min(row, H - 1))

    # 4. 统计信息
    valid = depth_m[depth_m > 0]
    print(f"文件 : {args.input}")
    print(f"尺寸 : {W} x {H}")
    print(f"选定行: {row}")
    print(f"最小值 : {depth_m.min():.3f} m")
    print(f"最大值 : {depth_m.max():.3f} m")
    print(f"平均值 : {depth_m.mean():.3f} m")
    if len(valid) > 0:
        print(f"有效深度(>0) 平均: {valid.mean():.3f} m")
        print(f"有效深度(>0) 中位: {np.median(valid):.3f} m")

    # 5. 生成伪彩色图并标注选定行
    depth_rgb = pseudo_color(depth_m, max_depth=args.max_depth)
    depth_rgb_annotated = depth_rgb.copy()
    cv2.line(depth_rgb_annotated, (0, row), (W - 1, row), (255, 255, 255), 2)
    cv2.line(depth_rgb_annotated, (0, row), (W - 1, row), (255, 0, 0), 1)

    # 6. 提取该行深度数据
    row_depth = depth_m[row, :].copy()
    x_coords = np.arange(W)

    # 区分有效/无效点（0 视为无效）
    valid_mask = row_depth > 0
    invalid_mask = ~valid_mask

    # 7. 画图
    fig, (ax_img, ax_curve) = plt.subplots(
        1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 1.2]}
    )
    fig.suptitle(f"Depth Visualization  |  File: {os.path.basename(args.input)}", fontsize=14)

    # 左侧：伪彩色深度图
    ax_img.imshow(depth_rgb_annotated)
    ax_img.set_title(f"Pseudo-color Depth (max={args.max_depth}m)\nRow {row} highlighted")
    ax_img.set_xlabel("Pixel Column")
    ax_img.set_ylabel("Pixel Row")
    ax_img.axhline(y=row, color="white", linewidth=2, alpha=0.7)
    ax_img.axhline(y=row, color="red", linewidth=1, alpha=0.9)

    # 右侧：该行深度曲线
    ax_curve.plot(
        x_coords[valid_mask],
        row_depth[valid_mask],
        color="steelblue",
        linewidth=1.5,
        label="Valid depth",
    )
    if np.any(invalid_mask):
        ax_curve.scatter(
            x_coords[invalid_mask],
            np.zeros(np.count_nonzero(invalid_mask)),
            color="red",
            s=8,
            marker="x",
            label="Invalid (depth=0)",
            zorder=5,
        )

    ax_curve.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # 深度阈值线
    if args.threshold >= 0:
        ax_curve.axhline(
            y=args.threshold,
            color="crimson",
            linestyle="-.",
            linewidth=1.5,
            alpha=0.8,
            label=f"Threshold ({args.threshold}m)",
        )
        ax_curve.legend(loc="upper right", fontsize=8)

    ax_curve.set_xlim(0, W - 1)
    ax_curve.set_ylim(min(-0.05, row_depth[valid_mask].min() - 0.1) if np.any(valid_mask) else -0.05,
                      row_depth[valid_mask].max() + 0.2 if np.any(valid_mask) else args.max_depth)
    ax_curve.set_xlabel("Pixel Column")
    ax_curve.set_ylabel("Depth (m)")
    ax_curve.set_title(f"Depth Profile at Row {row}")
    ax_curve.legend(loc="upper right", fontsize=8)
    ax_curve.grid(True, linestyle=":", alpha=0.6)

    # 在曲线上标注统计值
    if np.any(valid_mask):
        mean_d = row_depth[valid_mask].mean()
        ax_curve.axhline(y=mean_d, color="orange", linestyle="--", linewidth=1, alpha=0.7)
        ax_curve.text(
            W * 0.02, mean_d + 0.05,
            f"mean={mean_d:.3f}m",
            color="orange", fontsize=9,
        )

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"\n已保存结果图: {args.save}")
    else:
        print("\n显示图像，关闭窗口后退出...")

    plt.show()


if __name__ == "__main__":
    main()
