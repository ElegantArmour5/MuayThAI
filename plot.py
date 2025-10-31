import collections, collections.abc
for _n in ("Mapping","MutableMapping","Sequence"):
    if not hasattr(collections,_n): setattr(collections,_n,getattr(collections.abc,_n))

import cv2, argparse, math, os, numpy as np
import matplotlib.pyplot as plt

try:
    import mediapipe as mp
except Exception as e:
    raise SystemExit("Please install mediapipe: pip install mediapipe==0.10.*") from e

POSE = mp.solutions.pose
LM   = POSE.PoseLandmark
CONN = mp.solutions.pose.POSE_CONNECTIONS

# ---- Thresholds (from your spec) ----
JAB_ELBOW_MIN_DEG   = 170.0      # elbow >= 170°
JAB_FACE_MIN_FRAC   = 0.90       # wrist_face >= 0.90 * shoulder_to_wrist
HOOK_ELBOW_MIN_DEG  = 70.0       # 70° <= elbow <= 120°
HOOK_ELBOW_MAX_DEG  = 120.0
HOOK_FACE_MAX_FRAC  = 0.80       # wrist_face <= 0.80 * shoulder_to_wrist

VIS_T               = 0.5
SMOOTH_ALPHA        = 0.3        # EMA smoothing of 2D landmarks

def to_xyv(lms, w, h):
    xy = np.array([(lm.x*w, lm.y*h) for lm in lms], dtype=np.float32)
    v  = np.array([lm.visibility for lm in lms], dtype=np.float32)
    return xy, v

def angle_deg(a, b, c):
    ba = a - b; bc = c - b
    nba = np.linalg.norm(ba) + 1e-9
    nbc = np.linalg.norm(bc) + 1e-9
    cosang = np.clip(np.dot(ba, bc)/(nba*nbc), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def dist(a, b):
    return float(np.linalg.norm(a - b))

def ema(prev, cur, a):
    return cur if prev is None else (a*cur + (1-a)*prev)

def draw_pose(img, xy, vis, color=(180,220,255), jt=(255,255,255)):
    for a,b in CONN:
        a,b = int(a), int(b)
        if vis[a] > VIS_T and vis[b] > VIS_T:
            pa, pb = xy[a], xy[b]
            cv2.line(img, (int(pa[0]),int(pa[1])), (int(pb[0]),int(pb[1])), color, 2, cv2.LINE_AA)
    for i,p in enumerate(xy):
        if vis[i] > VIS_T:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, jt, -1, cv2.LINE_AA)

def classify_frame_side(side, xy, vis):
    if side == 'L':
        s, e, w = int(LM.LEFT_SHOULDER), int(LM.LEFT_ELBOW), int(LM.LEFT_WRIST)
    else:
        s, e, w = int(LM.RIGHT_SHOULDER), int(LM.RIGHT_ELBOW), int(LM.RIGHT_WRIST)
    ls, rs, nose = int(LM.LEFT_SHOULDER), int(LM.RIGHT_SHOULDER), int(LM.NOSE)
    if min(vis[[s,e,w]]) <= VIS_T:
        return None, None, None, None

    # face proxy
    if vis[nose] > VIS_T:
        face_pt = xy[nose]
    elif vis[ls] > VIS_T and vis[rs] > VIS_T:
        face_pt = 0.5*(xy[ls] + xy[rs])
    else:
        return None, None, None, None

    elbow = angle_deg(xy[s], xy[e], xy[w])
    sw    = dist(xy[s], xy[w]) + 1e-6
    wf    = dist(xy[w], face_pt)

    label = None
    if (elbow >= JAB_ELBOW_MIN_DEG) and (wf >= JAB_FACE_MIN_FRAC * sw):
        label = f"JAB/CROSS-{side}"
    elif (HOOK_ELBOW_MIN_DEG <= elbow <= HOOK_ELBOW_MAX_DEG) and (wf <= HOOK_FACE_MAX_FRAC * sw):
        label = f"HOOK-{side}"
    return elbow, wf, sw, label

def ensure_dir(p):
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def main():
    ap = argparse.ArgumentParser(description="Plot elbow angles & wrist-face distance over time; classifies per frame")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out_video", default="annotated.mp4")
    ap.add_argument("--out_csv",   default="metrics.csv")
    ap.add_argument("--plot_dir",  default="plots")
    ap.add_argument("--model_complexity", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=SMOOTH_ALPHA)
    ap.add_argument("--max_frames", type=int, default=0)
    args = ap.parse_args()

    ensure_dir(args.plot_dir)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open: {args.video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W,H))
    pose = POSE.Pose(static_image_mode=False, model_complexity=args.model_complexity, enable_segmentation=False)

    # Time series buffers
    t_s = []
    L_elbow, R_elbow = [], []
    L_wf, R_wf = [], []
    L_sw, R_sw = [], []
    L_label, R_label = [], []

    prev_xy = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if args.max_frames and i >= args.max_frames: break

        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        out = frame.copy()

        le, re = np.nan, np.nan
        lwf, rwf = np.nan, np.nan
        lsw, rsw = np.nan, np.nan
        llab, rlab = None, None

        if res.pose_landmarks:
            xy, vis = to_xyv(res.pose_landmarks.landmark, W, H)
            xy_s = ema(prev_xy, xy, args.alpha) if args.alpha>0 else xy
            prev_xy = xy_s

            # draw
            draw_pose(out, xy_s, vis)

            # left & right
            le, lwf, lsw, llab = classify_frame_side('L', xy_s, vis)
            re, rwf, rsw, rlab = classify_frame_side('R', xy_s, vis)

            # HUD small readouts
            x0, y0 = 12, 28
            if not np.isnan(le):
                cv2.putText(out, f"L elbow {le:.1f} deg", (x0,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1); y0+=20
                cv2.putText(out, f"L wf {lwf:.1f}px  sw {lsw:.1f}px", (x0,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1); y0+=20
                if llab: cv2.putText(out, llab, (x0,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60,200,255), 2); y0+=24
            x1, y1 = W-300, 28
            if not np.isnan(re):
                cv2.putText(out, f"R elbow {re:.1f} deg", (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1); y1+=20
                cv2.putText(out, f"R wf {rwf:.1f}px  sw {rsw:.1f}px", (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1); y1+=20
                if rlab: cv2.putText(out, rlab, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60,200,255), 2); y1+=24

        # write frame & push series
        writer.write(out)
        t_s.append(i / FPS)
        L_elbow.append(le); R_elbow.append(re)
        L_wf.append(lwf);   R_wf.append(rwf)
        L_sw.append(lsw);   R_sw.append(rsw)
        L_label.append(llab if llab else "")
        R_label.append(rlab if rlab else "")
        i += 1

    writer.release()
    cap.release()

    # ----- Save CSV -----
    import csv
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame","time_s",
                    "L_elbow_deg","R_elbow_deg",
                    "L_wrist_face_px","R_wrist_face_px",
                    "L_shoulder_wrist_px","R_shoulder_wrist_px",
                    "L_wf_over_sw","R_wf_over_sw",
                    "L_label","R_label"])
        for k in range(len(t_s)):
            lratio = (L_wf[k]/L_sw[k]) if (not np.isnan(L_wf[k]) and not np.isnan(L_sw[k]) and L_sw[k]>0) else np.nan
            rratio = (R_wf[k]/R_sw[k]) if (not np.isnan(R_wf[k]) and not np.isnan(R_sw[k]) and R_sw[k]>0) else np.nan
            w.writerow([
                k, t_s[k],
                L_elbow[k], R_elbow[k],
                L_wf[k], R_wf[k],
                L_sw[k], R_sw[k],
                lratio, rratio,
                L_label[k], R_label[k]
            ])

    # ----- Plots -----
    # 1) Elbow angle vs time
    plt.figure()
    plt.plot(t_s, L_elbow, label="Left elbow (deg)")
    plt.plot(t_s, R_elbow, label="Right elbow (deg)")
    plt.xlabel("Time (s)")
    plt.ylabel("Elbow angle (deg)")
    plt.title("Elbow angle over time")
    plt.legend()
    elbow_path = os.path.join(args.plot_dir, "elbow_angles.png")
    plt.savefig(elbow_path, dpi=160, bbox_inches="tight")
    plt.close()

    # 2) Wrist–face distance vs time
    plt.figure()
    plt.plot(t_s, L_wf, label="Left wrist–face (px)")
    plt.plot(t_s, R_wf, label="Right wrist–face (px)")
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (px)")
    plt.title("Wrist–face distance over time")
    plt.legend()
    wf_path = os.path.join(args.plot_dir, "wrist_face_distance.png")
    plt.savefig(wf_path, dpi=160, bbox_inches="tight")
    plt.close()

    # (Optional) Normalized wrist–face (÷ shoulder–wrist)
    L_ratio = [ (L_wf[k]/L_sw[k]) if (not np.isnan(L_wf[k]) and not np.isnan(L_sw[k]) and L_sw[k]>0) else np.nan for k in range(len(t_s)) ]
    R_ratio = [ (R_wf[k]/R_sw[k]) if (not np.isnan(R_wf[k]) and not np.isnan(R_sw[k]) and R_sw[k]>0) else np.nan for k in range(len(t_s)) ]
    plt.figure()
    plt.plot(t_s, L_ratio, label="Left (wf/sw)")
    plt.plot(t_s, R_ratio, label="Right (wf/sw)")
    plt.axhline(JAB_FACE_MIN_FRAC, linestyle="--", label="Jab min frac (0.90)")
    plt.axhline(HOOK_FACE_MAX_FRAC, linestyle="--", label="Hook max frac (0.80)")
    plt.xlabel("Time (s)")
    plt.ylabel("Ratio (wrist–face / shoulder–wrist)")
    plt.title("Normalized wrist–face distance over time")
    plt.legend()
    ratio_path = os.path.join(args.plot_dir, "wrist_face_ratio.png")
    plt.savefig(ratio_path, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"Saved video: {args.out_video}")
    print(f"Saved CSV:   {args.out_csv}")
    print(f"Saved plots: {elbow_path}, {wf_path}, {ratio_path}")

if __name__ == "__main__":
    main()
