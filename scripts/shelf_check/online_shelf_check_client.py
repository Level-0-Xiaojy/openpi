#!/usr/bin/env python3
"""实时读取 online_shelf_check.py 输出的货架状态文件。

用法:
    python scripts/shelf_check/read_shelf_status.py
    python scripts/shelf_check/read_shelf_status.py --status-file /tmp/shelf_status.json
"""

import argparse
import json
import time


def main():
    parser = argparse.ArgumentParser(description="实时读取货架是否上满的状态信号")
    parser.add_argument(
        "--status-file",
        default="/tmp/shelf_status.json",
        help="online_shelf_check.py 写出的状态文件路径",
    )
    parser.add_argument("--interval", type=float, default=0.1, help="读取间隔（秒）")
    parser.add_argument(
        "--print-every-frame",
        action="store_true",
        help="每次读到新帧都打印；默认只在状态变化时打印",
    )
    args = parser.parse_args()

    last_is_full = None
    last_frame_id = None

    print(f"[INFO] 读取状态文件: {args.status_file}")
    print("[INFO] 按 Ctrl-C 退出")

    while True:
        try:
            with open(args.status_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print("[WARN] 状态文件不存在，等待检测脚本启动...")
            time.sleep(1.0)
            continue
        except json.JSONDecodeError:
            # 写入端使用原子替换，正常不会读到半截 JSON；这里保留容错。
            time.sleep(args.interval)
            continue

        is_full = bool(data.get("is_full", False))
        frame_id = data.get("frame_id")
        should_print = args.print_every_frame or is_full != last_is_full
        if should_print and frame_id != last_frame_id:
            status_text = "货架已满" if is_full else "货架未满"
            filled_count = data.get("filled_count", "?")
            total_cols = data.get("total_cols", "?")
            timestamp = data.get("timestamp", 0.0)
            age_sec = max(0.0, time.time() - float(timestamp))
            print(
                f"[STATUS] {status_text} | {filled_count}/{total_cols} | "
                f"frame={frame_id} | age={age_sec:.2f}s"
            )
            last_is_full = is_full
            last_frame_id = frame_id

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
