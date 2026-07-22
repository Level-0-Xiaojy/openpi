"""Detect human-takeover moments from the RAW dagger data and write them into
the LeRobot dataset (open_giftbox_gqy07010702_steam_dagger).

Takeover mechanism (from bi_teleop_master.py): when the operator triggers
takeover, the SLAVE (follow) arm holds still while the master arm re-aligns to
the slave pose. The fixed choreography is:
    1.0s fixed pause + (0-1.0s) wait-for-slave-snapshot + 1.5s master interp
    + 1.0s stabilize  =>  the FOLLOW arm is frozen for >= ~3.5s.
Normal manipulation pauses are shorter, so a follow-frozen run of length
>= RESET_MIN frames (default 70 = 3.5s @ 20fps) marks one takeover.

The takeover moment is the last real frame BEFORE the frozen reset segment
("从静止前的那一刻"). Because the static-frame filter removed the frozen
segment, we map that raw moment to its position in the filtered/LeRobot
timeline via ``keep_indices``.

Writes per episode into meta/episodes.jsonl:
    "takeover_frames":  [filtered frame indices]   (LeRobot frame space)
    "takeover_seconds": [t/fps for each]           (seconds)
"""
import argparse
import bisect
import glob
import json
import os
import shutil
import sys

import numpy as np

_FILT_DIR = os.path.join(os.path.dirname(__file__), "mp4_json_edit")
sys.path.insert(0, _FILT_DIR)
import filter_x2robot_data_v2 as F  # noqa: E402

FPS = 20
# Raw dagger source dirs in the SAME order used for the LeRobot conversion
# (0701 -> episodes 0-15, 0702 -> episodes 16-34).
RAW_ROOTS = [
    "/mnt/resource/steam_dataset/open_giftbox/beijing_guqiuyi_20260701_pm_dagger",
    "/mnt/resource/steam_dataset/open_giftbox/beijing_guqiuyi_20260702_pm_dagger",
]
LEROBOT = "/mnt/public/guqiuyi/huggingface/lerobot/open_giftbox_gqy07010702_steam_dagger"


def follow_frozen_mask(data: list) -> np.ndarray:
    """Per-frame boolean: follow arm (either side) NOT moving vs previous frame."""
    fl = np.array([f["follow_left_position"] for f in data], dtype=np.float32)
    fr = np.array([f["follow_right_position"] for f in data], dtype=np.float32)
    n = len(data)
    frozen = np.zeros(n, dtype=bool)
    for i in range(1, n):
        moved = (
            np.linalg.norm(fl[i] - fl[i - 1]) > F.POS_THRESHOLD
            or np.linalg.norm(fr[i] - fr[i - 1]) > F.POS_THRESHOLD
        )
        frozen[i] = not moved
    return frozen  # frame 0 is False (no delta)


def frozen_runs(mask: np.ndarray) -> list[tuple[int, int, int]]:
    runs, s = [], None
    for i in range(len(mask)):
        if mask[i]:
            if s is None:
                s = i
        else:
            if s is not None:
                runs.append((s, i - 1, i - s))
                s = None
    if s is not None:
        runs.append((s, len(mask) - 1, len(mask) - s))
    return runs


def _uniq_ratio(data, s, e):
    """Fraction of DISTINCT follow positions in raw frames [s, e].

    A true takeover reset locks the slave arm (identical positions repeat), so
    the ratio is low; a model-controlled autonomous pause jitters, ratio ~1.
    """
    fl = np.array([f["follow_left_position"] for f in data[s:e + 1]], dtype=np.float32)
    fr = np.array([f["follow_right_position"] for f in data[s:e + 1]], dtype=np.float32)
    pos = np.round(np.hstack([fl, fr]), 4)
    return len(set(map(tuple, pos))) / max(1, (e - s + 1))


def detect_takeovers(data, keep_indices, reset_min, max_uniq_ratio=None):
    """Return (filtered_takeover_indices, raw_segments) for one episode."""
    frozen = follow_frozen_mask(data)
    lero_len = len(keep_indices) - 1  # LeRobot drops the last frame
    tk, segs = [], []
    for s, e, L in frozen_runs(frozen):
        if L < reset_min or s == 0:
            continue
        if max_uniq_ratio is not None and _uniq_ratio(data, s, e) >= max_uniq_ratio:
            continue  # too jittery to be a locked-arm reset
        # last kept frame before the frozen reset segment
        c = bisect.bisect_left(keep_indices, s)
        if c <= 0:
            continue
        filt = c - 1
        if 0 <= filt < lero_len:
            tk.append(filt)
            segs.append((s, e, L))
    # de-dup while keeping the segment info aligned
    seen, tk2, segs2 = set(), [], []
    for f, sg in sorted(zip(tk, segs)):
        if f in seen:
            continue
        seen.add(f)
        tk2.append(f)
        segs2.append(sg)
    return tk2, segs2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset-min", type=int, default=70,
                    help="Min follow-frozen run length (frames) that counts as a "
                         "takeover reset (default 70 = 3.5s @ 20fps).")
    ap.add_argument("--max-uniq-ratio", type=float, default=0.3,
                    help="Keep only frozen segments whose distinct-position ratio "
                         "is below this (locked-arm reset vs jittery autonomous "
                         "pause). Set to 1.0 to disable. Default 0.3.")
    ap.add_argument("--lerobot", default=LEROBOT)
    ap.add_argument("--dry-run", action="store_true",
                    help="Only report; do not modify episodes.jsonl.")
    args = ap.parse_args()

    raw_eps = []
    for root in RAW_ROOTS:
        raw_eps += sorted(d for d in glob.glob(f"{root}/*") if os.path.isdir(d))

    ep_file = os.path.join(args.lerobot, "meta", "episodes.jsonl")
    lero = [json.loads(l) for l in open(ep_file)]
    assert len(lero) == len(raw_eps), f"count mismatch {len(lero)} vs {len(raw_eps)}"

    total_tk = 0
    print(f"reset_min={args.reset_min} frames ({args.reset_min / FPS:.2f}s)\n")
    for idx, e in enumerate(raw_eps):
        nm = os.path.basename(e)
        data = json.load(open(f"{e}/{nm}.json"))["data"]
        arrays = F.get_state_arrays(data)
        keep = F.filter_stationary_frames(arrays, 0)
        assert len(keep) - 1 == lero[idx]["length"], \
            f"ep{idx} len mismatch {len(keep)-1} vs {lero[idx]['length']}"
        tk, segs = detect_takeovers(data, keep, args.reset_min,
                                    max_uniq_ratio=args.max_uniq_ratio)
        lero[idx]["takeover_frames"] = tk
        lero[idx]["takeover_seconds"] = [round(t / FPS, 2) for t in tk]
        total_tk += len(tk)
        seg_desc = ", ".join(f"raw[{s}-{e2}]={L}f({L/FPS:.1f}s)" for s, e2, L in segs)
        print(f"  ep{idx:2d} {nm[-18:]}: {len(tk)} takeover(s) -> filtered {tk}  "
              f"| {seg_desc}")

    print(f"\nTotal takeovers across {len(lero)} episodes: {total_tk}")
    if args.dry_run:
        print("(dry-run: episodes.jsonl NOT modified)")
        return
    bak = ep_file + ".bak_no_takeover"
    if not os.path.exists(bak):
        shutil.copy(ep_file, bak)
        print(f"Backed up original -> {bak}")
    with open(ep_file, "w") as f:
        for r in lero:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote takeover_frames / takeover_seconds into {ep_file}")


if __name__ == "__main__":
    main()
