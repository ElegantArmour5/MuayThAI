import collections, collections.abc
for _n in ("Mapping","MutableMapping","Sequence"):
    if not hasattr(collections,_n): setattr(collections,_n,getattr(collections.abc,_n))

import cv2, argparse, os, math, csv
import numpy as np
import matplotlib.pyplot as plt

try:
    import mediapipe as mp
except Exception as e:
    raise SystemExit("Please install mediapipe: pip install mediapipe==0.10.*") from e

POSE = mp.solutions.pose
LM   = POSE.PoseLandmark
CONN = mp.solutions.pose.POSE_CONNECTIONS

# Reference bands/lines (for plot context)
JAB_ELBOW_MIN_DEG   = 170.0
JAB_FACE_MIN_FRAC   = 0.90
HOOK_ELBOW_MIN_DEG  = 70.0
HOOK_ELBOW_MAX_DEG  = 120.0
HOOK_FACE_MAX_FRAC  = 0.80

VIS_T        = 0.5
SMOOTH_ALPHA = 0.3  # EMA for 2D landmarks

def ensure_dir(p):
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def to_xyv(lms, w, h):
    xy = np.array([(lm.x*w, lm.y*h) for lm in lms], dtype=np.float32)
    v  = np.array([lm.visibility for lm in lms], dtype=np.float32)
    return xy, v

def ema(prev, cur, a):
    return cur if prev is None else (a*cur + (1-a)*prev)

def dist(a, b):
    return float(np.linalg.norm(a - b))

def angle_deg(a, b, c):
    ba = a - b; bc = c - b
    nba = np.linalg.norm(ba) + 1e-9
    nbc = np.linalg.norm(bc) + 1e-9
    cosang = np.clip(np.dot(ba, bc) / (nba*nbc), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def draw_pose(img, xy, vis, color=(180,220,255), jt=(255,255,255)):
    for a,b in CONN:
        a,b = int(a), int(b)
        if vis[a] > VIS_T and vis[b] > VIS_T:
            pa, pb = xy[a], xy[b]
            cv2.line(img, (int(pa[0]),int(pa[1])), (int(pb[0]),int(pb[1])), color, 2, cv2.LINE_AA)
    for i,p in enumerate(xy):
        if vis[i] > VIS_T:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, jt, -1, cv2.LINE_AA)

def safe_grad(y, dt):
    """First derivative with NaN-tolerant filling (marks edges back as NaN)."""
    y = y.astype(float).copy()
    nanmask = np.isnan(y)
    if np.all(nanmask):
        return np.full_like(y, np.nan)
    # forward fill
    for k in range(1, len(y)):
        if np.isnan(y[k]) and not np.isnan(y[k-1]):
            y[k] = y[k-1]
    # back fill
    for k in range(len(y)-2, -1, -1):
        if np.isnan(y[k]) and not np.isnan(y[k+1]):
            y[k] = y[k+1]
    dy = np.gradient(y, dt)
    bad = nanmask | np.roll(nanmask, 1) | np.roll(nanmask, -1)
    dy[bad] = np.nan
    return dy

def safe_grad2(y, dt):
    """Second derivative as gradient of first derivative, NaN-aware."""
    dy = safe_grad(y, dt)
    d2y = safe_grad(dy, dt)
    return d2y

def main():
    ap = argparse.ArgumentParser(description="Plots elbow/wrist-face + first & second derivatives")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out_csv", default="metrics.csv")
    ap.add_argument("--plot_dir", default="plots")
    ap.add_argument("--out_video", default=None, help="Optional annotated overlay")
    ap.add_argument("--model_complexity", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=SMOOTH_ALPHA)
    ap.add_argument("--plot_ratio", action="store_true", help="Also plot wf/sw and its derivatives")
    ap.add_argument("--max_frames", type=int, default=0)
    args = ap.parse_args()

    ensure_dir(args.plot_dir)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened(): raise SystemExit(f"Cannot open: {args.video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
    DT  = 1.0 / float(FPS)

    writer = None
    if args.out_video:
        writer = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W,H))

    pose = POSE.Pose(static_image_mode=False, model_complexity=args.model_complexity, enable_segmentation=False)

    # Buffers
    t_s = []
    L_elb, R_elb = [], []
    L_wf,  R_wf  = [], []
    L_sw,  R_sw  = [], []
    L_ratio, R_ratio = [], []

    prev_xy = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if args.max_frames and i >= args.max_frames: break

        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        out = frame.copy()

        le = re = np.nan
        lw = rw = np.nan
        lsw = rsw = np.nan
        lr = rr = np.nan

        if res.pose_landmarks:
            xy, vis = to_xyv(res.pose_landmarks.landmark, W, H)
            xy_s = ema(prev_xy, xy, args.alpha) if args.alpha>0 else xy
            prev_xy = xy_s

            draw_pose(out, xy_s, vis)

            LS, RS = int(LM.LEFT_SHOULDER), int(LM.RIGHT_SHOULDER)
            LE, RE = int(LM.LEFT_ELBOW),   int(LM.RIGHT_ELBOW)
            LW, RW = int(LM.LEFT_WRIST),   int(LM.RIGHT_WRIST)
            NOSE   = int(LM.NOSE)

            face = None
            if vis[NOSE] > VIS_T:
                face = xy_s[NOSE]
            elif vis[LS] > VIS_T and vis[RS] > VIS_T:
                face = 0.5*(xy_s[LS] + xy_s[RS])

            if face is not None:
                if min(vis[[LS,LE,LW]]) > VIS_T:
                    le  = angle_deg(xy_s[LS], xy_s[LE], xy_s[LW])
                    lw  = dist(xy_s[LW], face)
                    lsw = dist(xy_s[LS], xy_s[LW]) + 1e-6
                    lr  = lw / lsw
                if min(vis[[RS,RE,RW]]) > VIS_T:
                    re  = angle_deg(xy_s[RS], xy_s[RE], xy_s[RW])
                    rw  = dist(xy_s[RW], face)
                    rsw = dist(xy_s[RS], xy_s[RW]) + 1e-6
                    rr  = rw / rsw

            # Tiny HUD
            cx = 12; cy = 24
            if not np.isnan(le):
                cv2.putText(out, f"L elbow {le:.1f} deg", (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1); cy+=20
                cv2.putText(out, f"L wf {lw:.1f}px  sw {lsw:.1f}px  R={lr:.2f}", (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1); cy+=20
            cx = W-320; cy = 24
            if not np.isnan(re):
                cv2.putText(out, f"R elbow {re:.1f} deg", (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1); cy+=20
                cv2.putText(out, f"R wf {rw:.1f}px  sw {rsw:.1f}px  R={rr:.2f}", (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1); cy+=20

            y = H-10
            cv2.putText(out, "Ref: Jab elbow ≥170°, R≥0.90 | Hook elbow 70–120°, R≤0.80",
                        (12,y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1, cv2.LINE_AA)

        # push
        t_s.append(i*DT)
        L_elb.append(le); R_elb.append(re)
        L_wf.append(lw);  R_wf.append(rw)
        L_sw.append(lsw); R_sw.append(rsw)
        L_ratio.append(lr); R_ratio.append(rr)

        if writer is not None:
            writer.write(out)
        i += 1

    cap.release()
    if writer is not None: writer.release()

    # Arrays & derivatives
    t    = np.array(t_s, dtype=float)
    Le   = np.array(L_elb, dtype=float); Re = np.array(R_elb, dtype=float)
    Lwf  = np.array(L_wf,  dtype=float); Rwf = np.array(R_wf,  dtype=float)

    dLe_dt   = safe_grad(Le,  DT); dRe_dt   = safe_grad(Re,  DT)
    dLwf_dt  = safe_grad(Lwf, DT); dRwf_dt  = safe_grad(Rwf, DT)

    d2Le_dt2  = safe_grad2(Le,  DT); d2Re_dt2  = safe_grad2(Re,  DT)
    d2Lwf_dt2 = safe_grad2(Lwf, DT); d2Rwf_dt2 = safe_grad2(Rwf, DT)

    # CSV
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "frame","time_s",
            "L_elbow_deg","R_elbow_deg",
            "L_wrist_face_px","R_wrist_face_px",
            "dL_elbow_deg_dt","dR_elbow_deg_dt",
            "dR_wrist_face_px_dt","dL_wrist_face_px_dt",  # keep both orders present
            "d2L_elbow_deg_dt2","d2R_elbow_deg_dt2",
            "d2L_wrist_face_px_dt2","d2R_wrist_face_px_dt2",
            "L_shoulder_wrist_px","R_shoulder_wrist_px",
            "L_ratio_wf_over_sw","R_ratio_wf_over_sw"
        ])
        for k in range(len(t)):
            w.writerow([
                k, t[k],
                L_elb[k], R_elb[k],
                L_wf[k],  R_wf[k],
                dLe_dt[k], dRe_dt[k],
                dRwf_dt[k], dLwf_dt[k],
                d2Le_dt2[k], d2Re_dt2[k],
                d2Lwf_dt2[k], d2Rwf_dt2[k],
                L_sw[k], R_sw[k],
                L_ratio[k], R_ratio[k]
            ])

    # PLOTS — elbow angle
    plt.figure()
    plt.plot(t, Le, label="Left elbow (deg)")
    plt.plot(t, Re, label="Right elbow (deg)")
    plt.axhline(JAB_ELBOW_MIN_DEG, linestyle="--", label="Jab elbow ≥170°")
    plt.axhspan(HOOK_ELBOW_MIN_DEG, HOOK_ELBOW_MAX_DEG, color="0.9", alpha=0.5, label="Hook elbow 70–120°")
    plt.xlabel("Time (s)"); plt.ylabel("deg"); plt.title("Elbow angle over time"); plt.legend()
    plt.savefig(os.path.join(args.plot_dir, "elbow_angles.png"), dpi=160, bbox_inches="tight"); plt.close()

    plt.figure()
    plt.plot(t, dLe_dt, label="d(L elbow)/dt (deg/s)")
    plt.plot(t, dRe_dt, label="d(R elbow)/dt (deg/s)")
    plt.xlabel("Time (s)"); plt.ylabel("deg/s"); plt.title("First derivative of elbow angle"); plt.legend()
    plt.savefig(os.path.join(args.plot_dir, "elbow_angles_d1.png"), dpi=160, bbox_inches="tight"); plt.close()

    plt.figure()
    plt.plot(t, d2Le_dt2, label="d2(L elbow)/dt2 (deg/s²)")
    plt.plot(t, d2Re_dt2, label="d2(R elbow)/dt2 (deg/s²)")
    plt.xlabel("Time (s)"); plt.ylabel("deg/s²"); plt.title("Second derivative of elbow angle"); plt.legend()
    plt.savefig(os.path.join(args.plot_dir, "elbow_angles_d2.png"), dpi=160, bbox_inches="tight"); plt.close()

    # PLOTS — wrist-face
    plt.figure()
    plt.plot(t, Lwf, label="Left wrist–face (px)")
    plt.plot(t, Rwf, label="Right wrist–face (px)")
    plt.xlabel("Time (s)"); plt.ylabel("px"); plt.title("Wrist–face distance over time"); plt.legend()
    plt.savefig(os.path.join(args.plot_dir, "wrist_face_distance.png"), dpi=160, bbox_inches="tight"); plt.close()

    plt.figure()
    plt.plot(t, dLwf_dt, label="d(L wf)/dt (px/s)")
    plt.plot(t, dRwf_dt, label="d(R wf)/dt (px/s)")
    plt.xlabel("Time (s)"); plt.ylabel("px/s"); plt.title("First derivative of wrist–face distance"); plt.legend()
    plt.savefig(os.path.join(args.plot_dir, "wrist_face_distance_d1.png"), dpi=160, bbox_inches="tight"); plt.close()

    plt.figure()
    plt.plot(t, d2Lwf_dt2, label="d2(L wf)/dt2 (px/s²)")
    plt.plot(t, d2Rwf_dt2, label="d2(R wf)/dt2 (px/s²)")
    plt.xlabel("Time (s)"); plt.ylabel("px/s²"); plt.title("Second derivative of wrist–face distance"); plt.legend()
    plt.savefig(os.path.join(args.plot_dir, "wrist_face_distance_d2.png"), dpi=160, bbox_inches="tight"); plt.close()

    # Optional: ratio & its derivatives
    if args.plot_ratio:
        Lr = np.array(L_ratio, dtype=float); Rr = np.array(R_ratio, dtype=float)
        dLr_dt  = safe_grad(Lr, DT); dRr_dt  = safe_grad(Rr, DT)
        d2Lr_dt2 = safe_grad2(Lr, DT); d2Rr_dt2 = safe_grad2(Rr, DT)

        plt.figure()
        plt.plot(t, Lr, label="Left R = wf/sw")
        plt.plot(t, Rr, label="Right R = wf/sw")
        plt.axhline(JAB_FACE_MIN_FRAC, linestyle="--", label="Jab R ≥ 0.90")
        plt.axhline(HOOK_FACE_MAX_FRAC, linestyle="--", label="Hook R ≤ 0.80")
        plt.xlabel("Time (s)"); plt.ylabel("ratio"); plt.title("Normalized wrist–face (wf/sw)"); plt.legend()
        plt.savefig(os.path.join(args.plot_dir, "wf_over_sw.png"), dpi=160, bbox_inches="tight"); plt.close()

        plt.figure()
        plt.plot(t, dLr_dt, label="d(L R)/dt (1/s)")
        plt.plot(t, dRr_dt, label="d(R R)/dt (1/s)")
        plt.xlabel("Time (s)"); plt.ylabel("1/s"); plt.title("First derivative of wf/sw"); plt.legend()
        plt.savefig(os.path.join(args.plot_dir, "wf_over_sw_d1.png"), dpi=160, bbox_inches="tight"); plt.close()

        plt.figure()
        plt.plot(t, d2Lr_dt2, label="d2(L R)/dt2 (1/s²)")
        plt.plot(t, d2Rr_dt2, label="d2(R R)/dt2 (1/s²)")
        plt.xlabel("Time (s)"); plt.ylabel("1/s²"); plt.title("Second derivative of wf/sw"); plt.legend()
        plt.savefig(os.path.join(args.plot_dir, "wf_over_sw_d2.png"), dpi=160, bbox_inches="tight"); plt.close()

    print(f"Saved CSV: {args.out_csv}")
    print(f"Saved plots to: {args.plot_dir}")
    if args.out_video:
        print(f"Saved annotated video: {args.out_video}")

if __name__ == "__main__":
    main()
