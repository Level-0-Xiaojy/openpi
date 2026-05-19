#!/usr/bin/env python3
"""货架上货状态检测：从初始深度图提取列信息，判断当前是否上满。

用法:
    # 分析初始状态，提取 4 列货物信息
    python shelf_stock_checker.py --start depth/start.png

    # 对比当前状态判断是否上满
    python shelf_stock_checker.py --start depth/start.png --current depth/end.png

    # 指定分析行（默认自动选图像中间偏上货架区域）
    python shelf_stock_checker.py --start depth/start.png --current depth/mid.png --row 250
"""

import argparse
import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

matplotlib.use("TkAgg")
DEPTH_SCALE = 1000.0


def load_depth(path: str) -> np.ndarray:
    """读取 uint16 PNG 深度图，返回 float32（米）。"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"无法读取深度图: {path}")
    return img.astype(np.float32) / DEPTH_SCALE


def interpolate_zeros(profile: np.ndarray) -> np.ndarray:
    """对深度曲线中的 0 值（无效深度）做线性插值填充，避免影响平滑和谷底检测。"""
    profile = profile.copy()
    zeros = profile <= 0
    if not np.any(zeros):
        return profile
    valid = ~zeros
    x_valid = np.where(valid)[0]
    if len(x_valid) < 2:
        return profile
    x_zeros = np.where(zeros)[0]
    profile[zeros] = np.interp(x_zeros, x_valid, profile[x_valid])
    return profile


def smooth_profile(profile: np.ndarray, window: int = 15) -> np.ndarray:
    """一维移动平均平滑。"""
    if window < 3 or len(profile) < window:
        return profile
    kernel = np.ones(window) / window
    # mode='same' 保持长度一致，边界用边缘值填充避免偏移
    padded = np.pad(profile, (window // 2, window // 2), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    # 长度可能对不齐，截取到原长度
    return smoothed[: len(profile)]


def preprocess_profile(profile: np.ndarray, median_size: int = 5, smooth_window: int = 15) -> np.ndarray:
    """
    深度曲线预处理流水线：
    1. 插值填充 0 值（无效深度）
    2. 中值滤波去除尖锐脉冲噪声（0 值影响的尖峰）
    3. 移动平均平滑
    """
    profile = interpolate_zeros(profile)
    if median_size >= 3:
        profile = median_filter(profile, size=median_size)
    if smooth_window >= 3:
        profile = smooth_profile(profile, window=smooth_window)
    return profile


def find_valleys(profile: np.ndarray, threshold: float, min_width: int = 20, prominence: float = 0.02) -> list:
    """
    在深度曲线中检测谷底（深度小 = 距离近 = 有货物）。
    使用 scipy.signal.find_peaks 对 -profile 找峰值，更鲁棒，不会因中间小幅波动而断开谷底。

    Returns:
        list[dict]: 每个谷底包含 center/left/right/width/depth
    """
    inverted = -profile
    peaks, props = find_peaks(
        inverted,
        height=(-threshold,),      # profile < threshold 才计入
        distance=min_width,        # 谷底之间最小间距
        prominence=prominence,     # 谷底相对两侧的“深度”，过滤浅噪声
        width=1,                   # 必须设置 width 才能拿到 left_ips / right_ips
    )

    valleys = []
    for i, peak in enumerate(peaks):
        left = int(props["left_ips"][i])
        right = int(props["right_ips"][i])
        valleys.append(
            {
                "center": int(peak),
                "left": left,
                "right": right,
                "width": right - left,
                "depth": float(profile[peak]),
            }
        )

    return valleys


def detect_shelf_region(profile: np.ndarray, edge_percentile: float = 90.0) -> tuple:
    """
    基于深度梯度检测货架左右边界。
    货架区域外通常是背景（深度突变）。
    """
    smooth = smooth_profile(profile, window=7)
    grad = np.abs(np.diff(smooth))
    grad = np.pad(grad, (1, 0), mode="edge")

    mid = len(profile) // 2

    # 左边界：左半部分梯度最大的前 N 个点中取最靠左的
    left_thresh = np.percentile(grad[:mid], edge_percentile)
    left_candidates = np.where(grad[:mid] >= left_thresh)[0]
    left = int(left_candidates[0]) if len(left_candidates) > 0 else 0

    # 右边界：右半部分梯度最大的前 N 个点中取最靠右的
    right_thresh = np.percentile(grad[mid:], edge_percentile)
    right_candidates = np.where(grad[mid:] >= right_thresh)[0]
    right = int(mid + right_candidates[-1]) if len(right_candidates) > 0 else len(profile) - 1

    # 做一些安全限制，避免边界贴太近
    left = min(left, mid - 50)
    right = max(right, mid + 50)

    return left, right


def pseudo_color(depth: np.ndarray, max_depth: float = 1.0) -> np.ndarray:
    """float32 深度图 -> uint8 伪彩色 RGB。"""
    disp = np.clip(depth, 0.0, max_depth)
    disp = disp / max_depth if max_depth > 0 else disp
    disp = (disp * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class ShelfStockChecker:
    def __init__(
        self,
        depth_scale: float = 1000.0,
        start_valley_offset: float = 0.06,
        check_valley_offset: float = 0.10,
        min_width: int = 25,
        median_size: int = 5,
        smooth_window: int = 15,
    ):
        self.depth_scale = depth_scale
        self.start_valley_offset = start_valley_offset
        self.check_valley_offset = check_valley_offset
        self.min_width = min_width
        self.median_size = median_size
        self.smooth_window = smooth_window
        self.n_cols = 0
        self.ref_valleys = []  # start 中提取的谷底参考
        self.shelf_bounds = (0, 0)
        self.bg_depth = 0.0
        self.row = 0

    # ------------------------------------------------------------------
    # 1) 从初始状态提取货架列信息
    # ------------------------------------------------------------------
    def analyze_start(self, start_depth: np.ndarray, row: int = -1):
        H, W = start_depth.shape
        self.row = row if row >= 0 else H // 2 + 20  # 稍微偏下一点，避开最顶部
        self.row = max(0, min(self.row, H - 1))

        profile_raw = start_depth[self.row, :]
        profile = preprocess_profile(profile_raw, median_size=self.median_size, smooth_window=self.smooth_window)

        # 自动检测货架左右边界
        left, right = detect_shelf_region(profile)
        left = max(0, left - 10)
        right = min(W - 1, right + 10)
        self.shelf_bounds = (left, right)

        # 估计背景深度（峰值平均）—— 先找货架区域内的"高平台"
        shelf_profile = profile[left:right]
        # 背景 = 深度较大的那些点的中位数
        bg_candidates = shelf_profile[shelf_profile > np.percentile(shelf_profile, 60)]
        self.bg_depth = float(np.median(bg_candidates)) if len(bg_candidates) > 0 else 0.75

        # 阈值 = 背景深度 - 容差，低于此值视为货物
        valley_threshold = self.bg_depth - self.start_valley_offset

        # 在货架区域内检测谷底
        self.ref_valleys = find_valleys(
            profile[left:right], threshold=valley_threshold, min_width=self.min_width
        )
        for v in self.ref_valleys:
            v["center"] += left
            v["left"] += left
            v["right"] += left

        self.n_cols = len(self.ref_valleys)
        return self.n_cols, self.ref_valleys, self.bg_depth

    # ------------------------------------------------------------------
    # 2) 判断当前是否上满
    # ------------------------------------------------------------------
    def check_current(self, current_depth: np.ndarray, check_row: int = -1) -> tuple:
        H, W = current_depth.shape
        row = check_row if check_row >= 0 else self.row
        row = max(0, min(row, H - 1))

        profile_raw = current_depth[row, :]
        profile = preprocess_profile(profile_raw, median_size=self.median_size, smooth_window=self.smooth_window)

        # 在当前帧中重新检测货架边界（允许视角小偏移）
        left, right = detect_shelf_region(profile)
        left = max(0, left - 10)
        right = min(W - 1, right + 10)

        # 重新估计当前行的背景深度（不同高度背景深度可能不同）
        shelf_profile = profile[left:right]
        bg_candidates = shelf_profile[shelf_profile > np.percentile(shelf_profile, 60)]
        bg_depth_current = float(np.median(bg_candidates)) if len(bg_candidates) > 0 else self.bg_depth

        # 在当前货架区域内重新检测谷底
        valley_threshold = bg_depth_current - self.check_valley_offset
        current_valleys = find_valleys(
            profile[left:right], threshold=valley_threshold, min_width=self.min_width
        )
        for v in current_valleys:
            v["center"] += left
            v["left"] += left
            v["right"] += left

        # 判断每一列是否上满：
        # 在 current 的 check_row 行，检查参考列位置附近是否存在谷底。
        # 上满时货物堆得高，check_row（高处）会被货物占据，深度曲线上会出现谷底；
        # 未上满时 check_row 处是背景/隔板，深度较大，不会形成谷底。
        search_margin = max(50, int(W * 0.10))  # 允许 10% 图像宽度的偏移，应对视角左右移动
        filled_flags = []

        for ref in self.ref_valleys:
            # 在 current 谷底中寻找与参考列位置最接近的匹配
            matched = None
            min_dist = float("inf")
            for v in current_valleys:
                dist = abs(v["center"] - ref["center"])
                if dist < min_dist and dist <= search_margin:
                    min_dist = dist
                    matched = v

            if matched is not None:
                # check_row 高度处有谷底 → 该列货物堆到了这个高度 → 上满
                has_stock = True
                local_min = matched["depth"]
            else:
                # check_row 高度处没有谷底 → 该列没堆到这个高度 → 未上满
                has_stock = False
                roi_l = max(0, ref["center"] - search_margin)
                roi_r = min(W, ref["center"] + search_margin)
                local_min = float(profile[roi_l:roi_r].min())

            filled_flags.append(
                {
                    "ref_center": ref["center"],
                    "local_min": local_min,
                    "has_stock": has_stock,
                    "matched_valley": matched,
                }
            )

        filled_count = sum(1 for f in filled_flags if f["has_stock"])
        is_full = filled_count >= self.n_cols

        return is_full, filled_count, filled_flags, current_valleys, profile, bg_depth_current


# ====================================================================
# 可视化
# ====================================================================
def visualize_result(checker: ShelfStockChecker, start_depth, current_depth, is_full, filled_count,
                     filled_flags, current_valleys, current_profile, bg_depth_current, check_row, save_path=""):
    row = checker.row
    left, right = checker.shelf_bounds
    n_cols = checker.n_cols

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Shelf Stock Check | StartRow={row} CheckRow={check_row} | Ref Columns={n_cols} | "
        f"Current Filled={filled_count}/{n_cols} | {'FULL ✅' if is_full else 'NOT FULL ❌'}",
        fontsize=14,
    )

    # ---------- 第 1 行：伪彩色深度图 ----------
    vmax = max(start_depth.max(), current_depth.max(), 1.0)

    ax = axes[0, 0]
    ax.imshow(pseudo_color(start_depth, vmax))
    ax.axhline(y=row, color="white", linewidth=2)
    ax.axhline(y=row, color="red", linewidth=1)
    for v in checker.ref_valleys:
        ax.axvline(x=v["center"], color="lime", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_title("Start (Reference)")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    ax = axes[0, 1]
    ax.imshow(pseudo_color(current_depth, vmax))
    ax.axhline(y=check_row, color="white", linewidth=2)
    ax.axhline(y=check_row, color="red", linewidth=1)
    for v in current_valleys:
        ax.axvline(x=v["center"], color="yellow", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_title("Current")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    # 留空或放文字说明
    ax = axes[0, 2]
    ax.axis("off")
    info_text = (
        f"Reference columns: {n_cols}\n"
        f"Background depth (start): {checker.bg_depth:.3f} m\n"
        f"Background depth (current): {bg_depth_current:.3f} m\n"
        f"Start valley threshold: {checker.bg_depth - checker.start_valley_offset:.3f} m\n"
        f"Check valley threshold: {bg_depth_current - checker.check_valley_offset:.3f} m\n\n"
        f"Current detected valleys: {len(current_valleys)}\n"
        f"Filled columns: {filled_count}/{n_cols}\n"
        f"Result: {'FULL ✅' if is_full else 'NOT FULL ❌'}"
    )
    ax.text(0.1, 0.5, info_text, fontsize=12, family="monospace",
            verticalalignment="center", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # ---------- 第 2 行：深度曲线 ----------
    start_profile = preprocess_profile(
        start_depth[row, :],
        median_size=checker.median_size,
        smooth_window=checker.smooth_window,
    )

    # Start 曲线
    ax = axes[1, 0]
    x = np.arange(len(start_profile))
    ax.plot(x, start_profile, color="steelblue", linewidth=1.5, label="Depth")
    ax.axvspan(left, right, alpha=0.1, color="green", label="Shelf region")
    for v in checker.ref_valleys:
        ax.axvline(x=v["center"], color="lime", linestyle="--", linewidth=1)
        ax.scatter([v["center"]], [v["depth"]], color="lime", s=50, zorder=5)
    ax.axhline(y=checker.bg_depth, color="orange", linestyle="--", linewidth=1, label=f"Bg={checker.bg_depth:.2f}m")
    ax.axhline(y=checker.bg_depth - checker.start_valley_offset, color="red", linestyle=":", linewidth=1, label="Threshold")
    ax.set_title("Start Profile")
    ax.set_xlabel("Column")
    ax.set_ylabel("Depth (m)")
    ax.legend(fontsize=7)
    ax.grid(True, linestyle=":", alpha=0.5)

    # Current 曲线
    ax = axes[1, 1]
    x = np.arange(len(current_profile))
    ax.plot(x, current_profile, color="steelblue", linewidth=1.5, label="Depth")
    ax.axvspan(left, right, alpha=0.1, color="green", label="Shelf region")
    for v in current_valleys:
        ax.axvline(x=v["center"], color="yellow", linestyle="--", linewidth=1)
        ax.scatter([v["center"]], [v["depth"]], color="yellow", s=50, zorder=5)
    for f in filled_flags:
        color = "green" if f["has_stock"] else "red"
        ax.scatter([f["ref_center"]], [f["local_min"]], color=color, s=80, marker="*", zorder=6)
    ax.axhline(y=bg_depth_current, color="orange", linestyle="--", linewidth=1, label=f"Bg={bg_depth_current:.2f}m")
    ax.axhline(y=bg_depth_current - checker.check_valley_offset, color="red", linestyle=":", linewidth=1, label="Threshold")
    ax.set_title("Current Profile")
    ax.set_xlabel("Column")
    ax.set_ylabel("Depth (m)")
    ax.legend(fontsize=7)
    ax.grid(True, linestyle=":", alpha=0.5)

    # 对比结果柱状图
    ax = axes[1, 2]
    col_labels = [f"Col{i + 1}" for i in range(n_cols)]
    col_depths = [f["local_min"] for f in filled_flags]
    colors = ["green" if f["has_stock"] else "red" for f in filled_flags]
    bars = ax.bar(col_labels, col_depths, color=colors, edgecolor="black")
    ax.axhline(
        y=bg_depth_current - checker.check_valley_offset,
        color="crimson",
        linestyle="-.",
        linewidth=1.5,
        label="Check threshold",
    )
    ax.set_ylim(0, max(checker.bg_depth, bg_depth_current) + 0.15)
    ax.set_title("Per-Column Check")
    ax.set_ylabel("Local Min Depth (m)")
    for bar, d in zip(bars, col_depths):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{d:.2f}", ha="center", va="bottom", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"结果图已保存: {save_path}")
    plt.show()


# ====================================================================
# 主函数
# ====================================================================
def main():
    parser = argparse.ArgumentParser(description="货架上货状态检测")
    parser.add_argument("--start", required=True, help="初始满货状态深度图 (.png)")
    parser.add_argument("--current", default="", help="当前状态深度图 (.png)")
    parser.add_argument("--row", type=int, default=310, help="start 图分析行号，-1 自动")
    parser.add_argument("--check-row", type=int, default=310,
                        help="current 图检测行号，-1 表示与 --row 相同")
    parser.add_argument("--start-valley-offset", type=float, default=0.02,
                        help="开始状态谷底检测阈值 = 背景深度 - 此偏移量（米），越小检测越严格")
    parser.add_argument("--check-valley-offset", type=float, default=0.09,
                        help="结束状态谷底检测阈值 = 背景深度 - 此偏移量（米），越大越不容易漏检")
    parser.add_argument("--min-width", type=int, default=40,
                        help="谷底最小宽度（像素），过滤噪声")
    parser.add_argument("--median-size", type=int, default=5,
                        help="中值滤波窗口大小，去除 0 值造成的尖锐谷底")
    parser.add_argument("--smooth-window", type=int, default=15,
                        help="移动平均平滑窗口大小")
    parser.add_argument("--save", default="", help="保存可视化结果路径")
    args = parser.parse_args()

    # 1. 读取并分析初始状态
    print("=" * 60)
    print("Step 1: 分析初始状态 (Start)")
    print("=" * 60)
    start_depth = load_depth(args.start)
    checker = ShelfStockChecker(
        start_valley_offset=args.start_valley_offset,
        check_valley_offset=args.check_valley_offset,
        min_width=args.min_width,
        median_size=args.median_size,
        smooth_window=args.smooth_window,
    )
    n_cols, ref_valleys, bg_depth = checker.analyze_start(start_depth, row=args.row)
    check_row = args.check_row if args.check_row >= 0 else args.row

    print(f"  图像尺寸     : {start_depth.shape[1]} x {start_depth.shape[0]}")
    print(f"  分析行       : {checker.row}")
    print(f"  货架边界     : {checker.shelf_bounds}")
    print(f"  背景深度     : {bg_depth:.3f} m")
    print(f"  start 谷底阈值: {bg_depth - checker.start_valley_offset:.3f} m")
    print(f"  check 谷底阈值: {bg_depth - checker.check_valley_offset:.3f} m (current 用)")
    print(f"  检测到货列数 : {n_cols}")
    for i, v in enumerate(ref_valleys, 1):
        print(f"    Col {i}: center={v['center']}, depth={v['depth']:.3f}m, width={v['width']}px")

    if not args.current:
        print("\n[INFO] 未提供 --current，仅完成初始分析。")
        return

    # 2. 检测当前状态
    print("\n" + "=" * 60)
    print("Step 2: 检测当前状态 (Current)")
    print("=" * 60)
    current_depth = load_depth(args.current)
    is_full, filled_count, filled_flags, current_valleys, current_profile, bg_depth_current = checker.check_current(
        current_depth, check_row=check_row
    )

    print(f"  当前检测到谷底数: {len(current_valleys)}")
    print(f"  有货列数        : {filled_count}/{n_cols}")
    for i, f in enumerate(filled_flags, 1):
        status = "✅ 有货" if f["has_stock"] else "❌ 缺货"
        print(f"    Col {i}: ref_center={f['ref_center']}, local_min={f['local_min']:.3f}m  -> {status}")

    if is_full:
        print(f"\n🟢 结果: 货架已上满 ({filled_count}/{n_cols})")
    else:
        print(f"\n🔴 结果: 货架未上满 ({filled_count}/{n_cols})")

    # 3. 可视化
    visualize_result(
        checker, start_depth, current_depth,
        is_full, filled_count, filled_flags,
        current_valleys, current_profile, bg_depth_current,
        check_row=check_row,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
