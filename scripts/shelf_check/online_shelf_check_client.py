#!/usr/bin/env python3
"""实时读取 online_shelf_check.py 输出的货架状态文件。

用法:
    python scripts/shelf_check/online_shelf_check_client.py
    python scripts/shelf_check/online_shelf_check_client.py --status-file /tmp/shelf_status.json
    python scripts/shelf_check/online_shelf_check_client.py --csv-output outputs/shelf_status.csv
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime


def _default_csv_output() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("outputs", "shelf_status_logs", f"shelf_status_{timestamp}.csv")


def _write_csv(path: str, rows: list[dict]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    fieldnames = [
        "read_time",
        "timestamp",
        "age_sec",
        "frame_id",
        "is_full",
        "status",
        "filled_count",
        "total_cols",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="实时读取货架是否上满的状态信号")
    parser.add_argument(
        "--status-file",
        default="/tmp/shelf_status.json",
        help="online_shelf_check.py 写出的状态文件路径",
    )
    parser.add_argument("--interval", type=float, default=0.05, help="读取间隔（秒）")
    parser.add_argument(
        "--print-every-frame",
        action="store_true",
        help="每次读到新帧都打印；默认只在状态变化时打印",
    )
    parser.add_argument(
        "--csv-output",
        default=_default_csv_output(),
        help="程序退出时保存读取记录的 CSV 路径",
    )
    args = parser.parse_args()

    last_is_full = None
    last_frame_id = None
    rows = []

    print(f"[INFO] 读取状态文件: {args.status_file}")
    print(f"[INFO] 退出时保存 CSV: {args.csv_output}")
    print("[INFO] 按 Ctrl-C 退出")

    try:
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
            timestamp = data.get("timestamp", 0.0)
            age_sec = max(0.0, time.time() - float(timestamp))

            if frame_id != last_frame_id:
                rows.append(
                    {
                        "read_time": time.time(),
                        "timestamp": timestamp,
                        "age_sec": age_sec,
                        "frame_id": frame_id,
                        "is_full": is_full,
                        "status": data.get("status", "FULL" if is_full else "NOT FULL"),
                        "filled_count": data.get("filled_count", ""),
                        "total_cols": data.get("total_cols", ""),
                    }
                )

            should_print = args.print_every_frame or is_full != last_is_full
            if should_print and frame_id != last_frame_id:
                status_text = "货架已满" if is_full else "货架未满"
                filled_count = data.get("filled_count", "?")
                total_cols = data.get("total_cols", "?")
                print(
                    f"[STATUS] {status_text} | {filled_count}/{total_cols} | "
                    f"frame={frame_id} | age={age_sec:.2f}s"
                )
                last_is_full = is_full

            last_frame_id = frame_id
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[INFO] 收到退出信号")
    finally:
        if rows:
            _write_csv(args.csv_output, rows)
            print(f"[INFO] 已保存 {len(rows)} 条记录到: {args.csv_output}")
        else:
            print("[INFO] 没有读取到状态数据，未生成 CSV")


if __name__ == "__main__":
    main()
