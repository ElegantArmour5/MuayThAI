# torso_side_metrics.py
# Extract reliable jab/cross metrics from a torso-up side-view video using MediaPipe Pose.

import collections, collections.abc
for _n in ("Mapping","MutableMapping","Sequence"):
    if not hasattr(collections,_n): setattr(collections,_n,getattr(collections.abc,_n))

import cv2, argparse, math, csv, os
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict

try:
    import mediapipe as mp
except Exception as e:
    raise SystemExit("Please install mediapipe: pip install mediapipe==0.10.*") from e

POSE_LANDMARKS = mp.solutions.pose.PoseLandmark

def to_xy(lm, w, h):
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)

def angle_deg(a, b, c):
    # angle at b, between BA and BC
    ba = a - b
    bc = c - b
    na = np.linalg.norm(ba) + 1e-9
    nc = np.linalg.norm(bc) + 1e-9
    cosang = np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def line_point_distance(a, b, p):
    # distance from point p to line through a->b
    ab = b - a
    if np.allclose(ab, 0):
        return float(np.linalg.norm(p - a))
    return float(np.abs(np.cross(ab, p - a)) / (np.linalg.norm(ab) + 1e-9))

def vec_angle_deg(u, v):
    nu = np.linalg.norm(u) + 1e-9
    nv = np.linalg.norm(v) + 1e-9
    cu = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
    return float(np.degrees(np.arccos(cu)))

def angle_vs_vertical(u):
    # angle between vector u and vertical (0,1)
    return vec_angle_deg(u, np.array([0.0, 1.0], dtype=np.float32))

def ema(prev, x, alpha):
    return alpha * x + (1 - alpha) * prev

@dataclass
class FrameMetrics:
    frame: int
    tsec: float
    S: float
    elbow_L: float
    elbow_R: float
    wristflex_L: float
    wristflex_R: float
    line_off_L: float
    line_off_R: float
    wristdrop_L: float
    wristdrop_R: float
    chintuck_deg: float
    torso_tilt_deg: float
    reach_L: float
    reach_R: float
    guard_ok_L: int
    guard_ok_R: int
    punch_arm: str  # 'L' or 'R'

THRESH = {
    "elbow_min": 165.0, "elbow_max": 175.0,
    "line_off_S": 0.06,
    "wrist_drop_S": 0.20,
    "wrist_flex_abs": 12.0,
    "chin_min": 10.0, "chin_max": 20.0,
    "torso_tilt_max": 10.0,
    "guard_eye_S": 0.05,
    "guard_horiz_S": 0.20,
    "impact_near_elbow": 5.0,   # elbow within 5° of 170° at impact
}

def compute_metrics(landmarks, w, h, prev_state: Dict) -> Optional[FrameMetrics]:
    get = lambda idx: landmarks[idx.value]

    try:
        LSH, RSH = to_xy(get(POSE_LANDMARKS.LEFT_SHOULDER), w, h), to_xy(get(POSE_LANDMARKS.RIGHT_SHOULDER), w, h)
        LEL, REL = to_xy(get(POSE_LANDMARKS.LEFT_ELBOW), w, h), to_xy(get(POSE_LANDMARKS.RIGHT_ELBOW), w, h)
        LWR, RWR = to_xy(get(POSE_LANDMARKS.LEFT_WRIST), w, h), to_xy(get(POSE_LANDMARKS.RIGHT_WRIST), w, h)
        LIN, RIN = to_xy(get(POSE_LANDMARKS.LEFT_INDEX), w, h), to_xy(get(POSE_LANDMARKS.RIGHT_INDEX), w, h)
        LEAR, REAR = to_xy(get(POSE_LANDMARKS.LEFT_EAR), w, h), to_xy(get(POSE_LANDMARKS.RIGHT_EAR), w, h)
        LEYE, REYE = to_xy(get(POSE_LANDMARKS.LEFT_EYE), w, h), to_xy(get(POSE_LANDMARKS.RIGHT_EYE), w, h)
        NOSE = to_xy(get(POSE_LANDMARKS.NOSE), w, h)
    except Exception:
        return None

    S = float(np.linalg.norm(LSH - RSH) + 1e-9)

    elbow_L = angle_deg(LSH, LEL, LWR)
    elbow_R = angle_deg(RSH, REL, RWR)

    # Wrist flexion: angle between forearm (elbow->wrist) and hand (wrist->index)
    wristflex_L = vec_angle_deg(LWR - LEL, LIN - LWR)
    wristflex_R = vec_angle_deg(RWR - REL, RIN - RWR)

    # Line offset: elbow distance to shoulder->wrist line, normalized by S
    line_off_L = line_point_distance(LSH, LWR, LEL) / S
    line_off_R = line_point_distance(RSH, RWR, REL) / S

    # Wrist drop vs shoulder (vertical distance), normalized by S
    wristdrop_L = abs(LWR[1] - LSH[1]) / S
    wristdrop_R = abs(RWR[1] - RSH[1]) / S

    # Chin tuck: pick better ear (closer to camera heuristic via x distance to nose)
    # Use vector ear->shoulder; compare vs vertical
    use_left_head = abs(LEAR[0] - NOSE[0]) < abs(REAR[0] - NOSE[0])
    ear, sh = (LEAR, LSH) if use_left_head else (REAR, RSH)
    chintuck_deg = angle_vs_vertical(ear - sh)

    # Torso tilt proxy: line from shoulder-mid -> nose vs vertical
    mid_sh = 0.5 * (LSH + RSH)
    torso_tilt_deg = angle_vs_vertical(NOSE - mid_sh)

    # Reach (absolute horizontal distance from shoulder to wrist)
    reach_L = abs(LWR[0] - LSH[0])
    reach_R = abs(RWR[0] - RSH[0])

    # Guard discipline: non-punch hand near face
    eye_y = 0.5 * (LEYE[1] + REYE[1])
    # distance from wrist to nearest cheek (use same-side ear)
    horiz_L = abs(LWR[0] - (LEAR[0]))
    horiz_R = abs(RWR[0] - (REAR[0]))
    guard_ok_L = int((LWR[1] <= eye_y + THRESH["guard_eye_S"] * S) and (horiz_L <= THRESH["guard_horiz_S"] * S))
    guard_ok_R = int((RWR[1] <= eye_y + THRESH["guard_eye_S"] * S) and (horiz_R <= THRESH["guard_horiz_S"] * S))

    # Decide current punching arm (which wrist is farther from its shoulder horizontally)
    punch_arm = 'L' if reach_L >= reach_R else 'R'

    return FrameMetrics(
        frame=prev_state["frame_idx"],
        tsec=prev_state["frame_idx"] / max(prev_state["fps"], 1e-6),
        S=S,
        elbow_L=elbow_L, elbow_R=elbow_R,
        wristflex_L=wristflex_L, wristflex_R=wristflex_R,
        line_off_L=line_off_L, line_off_R=line_off_R,
        wristdrop_L=wristdrop_L, wristdrop_R=wristdrop_R,
        chintuck_deg=chintuck_deg,
        torso_tilt_deg=torso_tilt_deg,
        reach_L=reach_L, reach_R=reach_R,
        guard_ok_L=guard_ok_L, guard_ok_R=guard_ok_R,
        punch_arm=punch_arm
    )

def detect_impacts(history, elbow_hist, reach_hist, fps):
    # Use absolute reach peaks near elbow ~170° as impact frames; require local maxima with sign change in velocity.
    impacts = []
    N = len(history)
    if N < 5: return impacts

    reach = np.array(reach_hist, dtype=np.float32)
    elbow = np.array(elbow_hist, dtype=np.float32)
    vel = np.gradient(reach)
    for i in range(2, N-2):
        is_peak = (reach[i-1] < reach[i] >= reach[i+1]) and (vel[i-1] > 0 >= vel[i+1])
        elbow_ok = abs(elbow[i] - 170.0) <= THRESH["impact_near_elbow"]
        if is_peak and elbow_ok:
            impacts.append(i)
    # Suppress near-duplicates within ~0.12s
    min_sep = max(2, int(0.12 * fps))
    dedup = []
    for idx in impacts:
        if not dedup or idx - dedup[-1] >= min_sep:
            dedup.append(idx)
    return dedup

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--out_dir", default="out_metrics", help="Folder for CSV outputs")
    ap.add_argument("--show", action="store_true", help="Show live overlay")
    ap.add_argument("--max_frames", type=int, default=0, help="Optional cap for frames (0 = all)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)
    frame_idx = 0

    rows = []
    # For impact detection we keep per-arm histories
    reach_hist_L, reach_hist_R = [], []
    elbow_hist_L, elbow_hist_R = [], []

    # Smoothed reach for overlay (EMA)
    ema_reach_L = 0.0
    ema_reach_R = 0.0
    alpha = 0.2

    while True:
        ok, frame = cap.read()
        if not ok: break
        if args.max_frames and frame_idx >= args.max_frames: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        prev_state = {"frame_idx": frame_idx, "fps": fps}
        if res.pose_landmarks is not None:
            lm = res.pose_landmarks.landmark
            fm = compute_metrics(lm, w, h, prev_state)
        else:
            fm = None

        if fm:
            rows.append(fm)
            reach_hist_L.append(fm.reach_L)
            reach_hist_R.append(fm.reach_R)
            elbow_hist_L.append(fm.elbow_L)
            elbow_hist_R.append(fm.elbow_R)
            ema_reach_L = ema(ema_reach_L, fm.reach_L, alpha) if frame_idx else fm.reach_L
            ema_reach_R = ema(ema_reach_R, fm.reach_R, alpha) if frame_idx else fm.reach_R

            if args.show:
                txt = f"Frame {fm.frame}  t={fm.tsec:.2f}s  S={fm.S:.1f}"
                cv2.putText(frame, txt, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                # Elbow/line/wrist labels for selected punching arm
                arm = fm.punch_arm
                elbow = fm.elbow_L if arm=='L' else fm.elbow_R
                line_off = fm.line_off_L if arm=='L' else fm.line_off_R
                wristflex = fm.wristflex_L if arm=='L' else fm.wristflex_R
                wristdrop = fm.wristdrop_L if arm=='L' else fm.wristdrop_R
                guard_ok = fm.guard_ok_R if arm=='L' else fm.guard_ok_L  # non-punch hand

                y0 = 50
                disp = [
                    f"Arm={arm}  Elbow={elbow:.1f}°  LineOff/S={line_off:.3f}",
                    f"WristFlex={wristflex:.1f}°  WristDrop/S={wristdrop:.3f}",
                    f"ChinTuck={fm.chintuck_deg:.1f}°  TorsoTilt={fm.torso_tilt_deg:.1f}°",
                    f"Reach(L/R)={ema_reach_L:.1f}/{ema_reach_R:.1f} px  GuardOK(non-punch)={guard_ok}"
                ]
                for i,s in enumerate(disp):
                    cv2.putText(frame, s, (12, y0+22*i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50,220,50), 2)

        if args.show:
            cv2.imshow("Torso Side Metrics", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        frame_idx += 1

    cap.release()
    if args.show:
        cv2.destroyAllWindows()

    if not rows:
        raise SystemExit("No pose detected—check framing/lighting or try model_complexity=2.")

    # Write per-frame CSV
    cf = os.path.join(args.out_dir, "per_frame_metrics.csv")
    with open(cf, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow([
            "frame","tsec","S",
            "elbow_L","elbow_R",
            "wristflex_L","wristflex_R",
            "line_off_L","line_off_R",
            "wristdrop_L","wristdrop_R",
            "chintuck_deg","torso_tilt_deg",
            "reach_L","reach_R",
            "guard_ok_L","guard_ok_R",
            "punch_arm"
        ])
        for r in rows:
            wr.writerow([
                r.frame, f"{r.tsec:.4f}", f"{r.S:.3f}",
                f"{r.elbow_L:.2f}", f"{r.elbow_R:.2f}",
                f"{r.wristflex_L:.2f}", f"{r.wristflex_R:.2f}",
                f"{r.line_off_L:.4f}", f"{r.line_off_R:.4f}",
                f"{r.wristdrop_L:.4f}", f"{r.wristdrop_R:.4f}",
                f"{r.chintuck_deg:.2f}", f"{r.torso_tilt_deg:.2f}",
                f"{r.reach_L:.2f}", f"{r.reach_R:.2f}",
                r.guard_ok_L, r.guard_ok_R,
                r.punch_arm
            ])
    # Detect impacts per arm
    fps = float(fps)
    impacts_L = detect_impacts(rows, [r.elbow_L for r in rows], [r.reach_L for r in rows], fps)
    impacts_R = detect_impacts(rows, [r.elbow_R for r in rows], [r.reach_R for r in rows], fps)

    # Summarize each impact with pass/fail vs thresholds
    ci = os.path.join(args.out_dir, "per_impact_summary.csv")
    with open(ci, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow([
            "arm","impact_frame","tsec",
            "elbow_deg","within_range",
            "line_off_over_S","line_off_ok",
            "wristflex_deg","wristflex_ok",
            "wristdrop_over_S","wristdrop_ok",
            "chintuck_deg","chintuck_ok",
            "torso_tilt_deg","torso_tilt_ok",
            "guard_ok_nonpunch"
        ])
        def eval_impact(idx, arm):
            r = rows[idx]
            elbow = r.elbow_L if arm=='L' else r.elbow_R
            line_off = r.line_off_L if arm=='L' else r.line_off_R
            wristflex = r.wristflex_L if arm=='L' else r.wristflex_R
            wristdrop = r.wristdrop_L if arm=='L' else r.wristdrop_R
            non_punch_guard = r.guard_ok_R if arm=='L' else r.guard_ok_L

            elbow_ok = (THRESH["elbow_min"] <= elbow <= THRESH["elbow_max"])
            line_ok = (line_off <= THRESH["line_off_S"])
            flex_ok = (abs(wristflex) <= THRESH["wrist_flex_abs"])
            drop_ok = (wristdrop <= THRESH["wrist_drop_S"])
            chin_ok = (THRESH["chin_min"] <= r.chintuck_deg <= THRESH["chin_max"])
            torso_ok = (r.torso_tilt_deg <= THRESH["torso_tilt_max"])

            wr.writerow([
                arm, idx, f"{r.tsec:.3f}",
                f"{elbow:.1f}", int(elbow_ok),
                f"{line_off:.3f}", int(line_ok),
                f"{wristflex:.1f}", int(flex_ok),
                f"{wristdrop:.3f}", int(drop_ok),
                f"{r.chintuck_deg:.1f}", int(chin_ok),
                f"{r.torso_tilt_deg:.1f}", int(torso_ok),
                int(non_punch_guard)
            ])

        for idx in impacts_L: eval_impact(idx, 'L')
        for idx in impacts_R: eval_impact(idx, 'R')

    print(f"[OK] Wrote per-frame metrics: {cf}")
    print(f"[OK] Wrote per-impact summary: {ci}")
    print(f"Detected impacts: L={len(impacts_L)}  R={len(impacts_R)}")

if __name__ == "__main__":
    main()
