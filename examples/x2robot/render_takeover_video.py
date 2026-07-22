"""Render takeover + subtask verification videos for LeRobot dagger episodes.

Face camera only. Overlays:
  * current subtask (id + name), colour-coded
  * a red "TAKEOVER" flash (1s) at each `takeover_frames` position
  * a bottom timeline: one coloured segment per subtask, red takeover ticks,
    green playhead

Subtask boundaries come from `subtask_end_frames` in meta/episodes.jsonl
(11 fixed subtasks). A None entry means Gemini reported observed=false for that
subtask; consecutive unknowns are MERGED into the next known boundary and shown
as a range (e.g. "8-10") — we never invent a boundary.

Usage:
    python render_takeover_video.py --episodes 16 17 22 26 34
"""
import argparse
import json
import os

import cv2
import numpy as np

LEROBOT = "/mnt/public/guqiuyi/huggingface/lerobot/open_giftbox_gqy07010702_steam_dagger"
CAM = "face_view"
FPS = 20
CW, CH = 640, 480
FLASH = 20  # frames the TAKEOVER banner stays lit (~1s)
FONT = cv2.FONT_HERSHEY_SIMPLEX

SUBTASK_NAMES = [
    "Cut open the carton",
    "Open the carton",
    "Grasp the gift box",
    "Place the gift box on the table",
    "Lay the gift box down",
    "Hold the gift box with the left arm",
    "Open the gift box",
    "Remove standee 1 and lay it on the table",
    "Remove standee 2 and lay it on the table",
    "Close the gift box",
    "Done",
]
# 11 distinct BGR colours
PALETTE = [
    (180, 119, 31), (14, 127, 255), (44, 160, 44), (40, 39, 214), (189, 103, 148),
    (75, 86, 140), (194, 119, 227), (127, 127, 127), (34, 189, 188), (207, 190, 23),
    (170, 170, 90),
]


def subtask_spans(ends, length):
    """-> [(start, end, ids, exact)] using only boundaries Gemini gave us."""
    spans, prev, pending = [], 0, []
    for i, e in enumerate(ends):
        if e is None:
            pending.append(i)
            continue
        ids = pending + [i]
        spans.append((prev, e, ids, len(ids) == 1))
        prev, pending = e, []
    if pending:
        spans.append((prev, length, pending, False))
    return [s for s in spans if s[1] > s[0]]


def span_label(ids, exact):
    if exact:
        i = ids[0]
        return f"[{i+1}/11] {SUBTASK_NAMES[i]}"
    a, b = ids[0], ids[-1]
    return f"[{a+1}-{b+1}/11] {SUBTASK_NAMES[a]} ... (boundary unknown)"


def render(idx, meta, lerobot, out_path):
    tks = meta["takeover_frames"]
    length = meta["length"]
    ends = meta.get("subtask_end_frames") or []
    spans = subtask_spans(ends, length) if ends else []

    cap = cv2.VideoCapture(f"{lerobot}/videos/chunk-000/{CAM}/episode_{idx:06d}.mp4")
    out_w, out_h = CW, CH + 96
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (out_w, out_h))

    by, bh = CH + 62, 20  # timeline bar geometry
    span_of = lambda f: next((s for s in spans if s[0] <= f < s[1]), None)

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        canvas = np.zeros((out_h, out_w, 3), np.uint8)
        canvas[:CH] = cv2.resize(frame, (CW, CH))

        sp = span_of(i)
        col = PALETTE[sp[2][0] % len(PALETTE)] if sp else (200, 200, 200)

        active = any(t <= i < t + FLASH for t in tks)
        if active:
            cv2.rectangle(canvas, (0, 0), (CW - 1, CH - 1), (0, 0, 255), 8)
            cv2.putText(canvas, "TAKEOVER", (CW - 300, 46), FONT, 1.2, (0, 0, 255), 4)

        # line 1: frame / time
        cv2.putText(canvas, f"ep{idx}  frame {i}/{length}  t={i/FPS:5.1f}s",
                    (6, CH + 20), FONT, 0.5, (255, 255, 255), 1)
        # line 2: current subtask, in its colour
        txt = span_label(sp[2], sp[3]) if sp else "(no subtask)"
        cv2.putText(canvas, txt[:64], (6, CH + 44), FONT, 0.5, col, 1)

        # timeline: one coloured segment per subtask
        cv2.rectangle(canvas, (5, by), (out_w - 5, by + bh), (45, 45, 45), -1)
        W = out_w - 10
        for s, e, ids, exact in spans:
            x0 = 5 + int(W * s / max(1, length))
            x1 = 5 + int(W * e / max(1, length))
            c = PALETTE[ids[0] % len(PALETTE)]
            cv2.rectangle(canvas, (x0, by), (x1, by + bh), c, -1)
            if not exact:  # hatch the uncertain span
                for x in range(x0, x1, 6):
                    cv2.line(canvas, (x, by), (x, by + bh), (30, 30, 30), 1)
            if x1 - x0 > 12:
                cv2.putText(canvas, str(ids[0] + 1), (x0 + 2, by + bh - 5),
                            FONT, 0.35, (255, 255, 255), 1)
        for t in tks:  # takeover ticks
            x = 5 + int(W * t / max(1, length))
            cv2.rectangle(canvas, (x - 2, by - 5), (x + 2, by + bh + 5), (0, 0, 255), -1)
        px = 5 + int(W * min(i, length) / max(1, length))  # playhead
        cv2.rectangle(canvas, (px - 1, by - 7), (px + 1, by + bh + 7), (0, 255, 0), -1)

        writer.write(canvas)
        i += 1
    cap.release()
    writer.release()
    return i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--lerobot", default=LEROBOT)
    ap.add_argument("--out-dir", default="/mnt/public/guqiuyi/RLinf_active/logs/takeover_verify")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    meta = [json.loads(l) for l in open(f"{args.lerobot}/meta/episodes.jsonl")]
    for idx in args.episodes:
        m = meta[idx]
        out = os.path.join(args.out_dir, f"episode_{idx:06d}_takeover_subtask.mp4")
        n = render(idx, m, args.lerobot, out)
        nmiss = sum(1 for x in (m.get("subtask_end_frames") or []) if x is None)
        print(f"ep{idx}: {n} frames | takeovers={m['takeover_seconds']} | "
              f"subtask 缺失={nmiss} -> {out}")


if __name__ == "__main__":
    main()
