# adaptive_punch_flagger.py
import collections, collections.abc
for _n in ("Mapping","MutableMapping","Sequence"):
    if not hasattr(collections,_n): setattr(collections,_n,getattr(collections.abc,_n))

import cv2, math, argparse, csv, os
import numpy as np
from collections import deque

try:
    import mediapipe as mp
except Exception as e:
    raise SystemExit("Please install mediapipe: pip install mediapipe==0.10.*") from e

PLM = mp.solutions.pose.PoseLandmark

def to_xy(lm, W, H):
    return np.array([lm.x * W, lm.y * H], dtype=np.float32)

def angle_deg(a, b, c):
    ba = a - b; bc = c - b
    nba = np.linalg.norm(ba) + 1e-9
    nbc = np.linalg.norm(bc) + 1e-9
    cosang = np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def tup(p): return tuple(p.astype(int))

def local_max_adaptive(ext_hist, win, k, min_prom):
    """Return True if the middle of the last 3 samples is a local max with adaptive prominence."""
    if len(ext_hist) < 3: return False
    a, b, c = ext_hist[-3], ext_hist[-2], ext_hist[-1]
    if not (a < b and c < b):  # strictly a local max at b
        return False
    window = list(ext_hist)[-win:]
    prom = b - min(window)
    std  = np.std(window) if len(window) >= 3 else 0.0
    return prom > max(k * std, min_prom)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Input video")
    ap.add_argument("--out", default=None, help="Annotated mp4 (default: *_flagged.mp4)")
    ap.add_argument("--csv", default=None, help="CSV (default: *_flags.csv)")
    # Easy / lenient knobs
    ap.add_argument("--elbow_min_deg", type=float, default=160.0, help="Min elbow angle at peak")
    ap.add_argument("--guard_margin", type=float, default=0.05, help="Guard down if (current > rolling_median + margin)")
    ap.add_argument("--guard_fallback", type=float, default=0.40, help="Fallback guard threshold if too few samples")
    ap.add_argument("--win", type=int, default=15, help="Rolling window (frames) for adaptive stats")
    ap.add_argument("--k", type=float, default=1.0, help="Peak prominence multiplier on rolling std")
    ap.add_argument("--min_prom", type=float, default=0.04, help="Absolute min prominence (normalized units)")
    ap.add_argument("--flash_ms", type=int, default=150, help="Red flash duration for flagged peaks")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = args.out or os.path.splitext(args.video)[0] + "_flagged.mp4"
    csv_path = args.csv or os.path.splitext(args.video)[0] + "_flags.csv"
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Rolling histories (about ~4s max)
    maxlen = max(60, args.win * 4)
    ext_hist   = {"L": deque(maxlen=maxlen), "R": deque(maxlen=maxlen)}
    guard_hist = {"L": deque(maxlen=maxlen), "R": deque(maxlen=maxlen)}
    elbow_hist = {"L": deque(maxlen=maxlen), "R": deque(maxlen=maxlen)}

    flags = []  # [time_s, hand, reason]
    flash_until = -1
    flash_text  = ""

    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        lm  = res.pose_landmarks.landmark if res.pose_landmarks else None
        if not lm:
            writer.write(frame); continue

        def vis_ok(i, thr=0.3): return lm[i].visibility >= thr

        need = [PLM.LEFT_SHOULDER, PLM.RIGHT_SHOULDER, PLM.LEFT_ELBOW, PLM.RIGHT_ELBOW,
                PLM.LEFT_WRIST, PLM.RIGHT_WRIST]
        if not all(vis_ok(i) for i in need if i in range(len(lm))):
            writer.write(frame); continue

        LSH = to_xy(lm[PLM.LEFT_SHOULDER], W, H)
        RSH = to_xy(lm[PLM.RIGHT_SHOULDER], W, H)
        shoulder_span = np.linalg.norm(LSH - RSH) + 1e-6

        # Mouth center (fallback to nose)
        if vis_ok(PLM.MOUTH_LEFT) and vis_ok(PLM.MOUTH_RIGHT):
            MOUTH = 0.5 * (to_xy(lm[PLM.MOUTH_LEFT], W, H) + to_xy(lm[PLM.MOUTH_RIGHT], W, H))
        elif vis_ok(PLM.NOSE):
            MOUTH = to_xy(lm[PLM.NOSE], W, H)
        else:
            writer.write(frame); continue

        hands = {
            "L": {"WR": to_xy(lm[PLM.LEFT_WRIST], W, H), "EL": to_xy(lm[PLM.LEFT_ELBOW], W, H), "SH": to_xy(lm[PLM.LEFT_SHOULDER], W, H)},
            "R": {"WR": to_xy(lm[PLM.RIGHT_WRIST], W, H), "EL": to_xy(lm[PLM.RIGHT_ELBOW], W, H), "SH": to_xy(lm[PLM.RIGHT_SHOULDER], W, H)},
        }

        # Per-hand normalized metrics
        cur = {}
        for s, d in hands.items():
            ext   = np.linalg.norm(d["WR"] - d["SH"]) / shoulder_span
            elbow = angle_deg(d["SH"], d["EL"], d["WR"])
            guard = np.linalg.norm(d["WR"] - MOUTH) / shoulder_span
            cur[s] = {"ext": ext, "elbow": elbow, "guard": guard}
            ext_hist[s].append(ext)
            elbow_hist[s].append(elbow)
            guard_hist[s].append(guard)

        # Peak detection at b = frame_idx-1 when ext[-2] is a local max with adaptive prominence
        def maybe_peak(side):
            if local_max_adaptive(ext_hist[side], args.win, args.k, args.min_prom):
                # Use metrics at the middle sample of the last 3 -> index -2
                ext_b   = ext_hist[side][-2]
                elbow_b = elbow_hist[side][-2]
                guard_b = guard_hist[side][-2]

                other   = "R" if side == "L" else "L"
                # Guard baseline (rolling median) for the OTHER hand over window
                oth_hist = list(guard_hist[other])[-args.win:]
                if len(oth_hist) >= max(5, args.win//2):
                    base = float(np.median(oth_hist))
                    guard_down_other = guard_hist[other][-2] > (base + args.guard_margin)
                else:
                    # Fallback to absolute, lenient threshold
                    guard_down_other = guard_hist[other][-2] > args.guard_fallback

                bad_elbow = elbow_b < args.elbow_min_deg

                if guard_down_other or bad_elbow:
                    t = (frame_idx - 1) / fps
                    reasons = []
                    if guard_down_other: reasons.append("guard down (other hand)")
                    if bad_elbow:        reasons.append("incomplete extension")
                    flags.append([round(t, 3), side, "; ".join(reasons)])
                    return True, "; ".join(reasons)
            return False, ""

        peak_L, reason_L = maybe_peak("L")
        peak_R, reason_R = maybe_peak("R")

        # Minimal overlay: wrists + transient red flash on flagged peak frames
        cv2.circle(frame, tup(hands["L"]["WR"]), 5, (255,255,255), -1)
        cv2.circle(frame, tup(hands["R"]["WR"]), 5, (180,255,180), -1)

        if peak_L:
            flash_until = frame_idx + int((args.flash_ms / 1000.0) * fps)
            flash_text  = f"L punch: {reason_L}"
        if peak_R:
            flash_until = frame_idx + int((args.flash_ms / 1000.0) * fps)
            flash_text  = f"R punch: {reason_R}"

        if frame_idx <= flash_until:
            ov = frame.copy()
            cv2.rectangle(ov, (0,0), (W,H), (0,0,255), -1)
            frame = cv2.addWeighted(ov, 0.12, frame, 0.88, 0)
            if flash_text:
                cv2.putText(frame, flash_text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2, cv2.LINE_AA)

        writer.write(frame)

    cap.release()
    writer.release()

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "hand", "reason"])
        w.writerows(flags)

    print(f"Done.\nVideo: {out_path}\nCSV: {csv_path}\nTotal flagged: {len(flags)}")

if __name__ == "__main__":
    main()
