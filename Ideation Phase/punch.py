import collections, collections.abc
for _n in ("Mapping","MutableMapping","Sequence"):
    if not hasattr(collections,_n): setattr(collections,_n,getattr(collections.abc,_n))

import cv2, argparse, math, numpy as np

try:
    import mediapipe as mp
except Exception as e:
    raise SystemExit("Please install mediapipe: pip install mediapipe==0.10.*") from e

POSE = mp.solutions.pose
LM   = POSE.PoseLandmark
CONN = mp.solutions.pose.POSE_CONNECTIONS

# ---------- Thresholds (as requested) ----------
JAB_ELBOW_MIN_DEG   = 170.0     # elbow >= 170°  (nearly straight)
JAB_FACE_MIN_FRAC   = 0.90      # wrist_face >= 0.90 * shoulder_to_wrist

HOOK_ELBOW_MIN_DEG  = 70.0      # 70° <= elbow <= 120°
HOOK_ELBOW_MAX_DEG  = 120.0
HOOK_FACE_MAX_FRAC  = 0.80      # wrist_face <= 0.80 * shoulder_to_wrist

# ---------- Misc ----------
VIS_T               = 0.5       # vis threshold
SMOOTH_ALPHA        = 0.3       # EMA smoothing of 2D landmarks (set to 0 to disable)
FONT                = cv2.FONT_HERSHEY_SIMPLEX

# ---------- Helpers ----------
def to_xyv(lms, w, h):
    xy = np.array([(lm.x*w, lm.y*h) for lm in lms], dtype=np.float32)
    v  = np.array([lm.visibility for lm in lms], dtype=np.float32)
    return xy, v

def angle_deg(a, b, c):
    ba = a - b; bc = c - b
    nba = np.linalg.norm(ba) + 1e-9
    nbc = np.linalg.norm(bc) + 1e-9
    cosang = np.clip(np.dot(ba, bc) / (nba*nbc), -1.0, 1.0)
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
    """
    side: 'L' or 'R'
    Returns: label (str or None), metrics dict
    """
    if side == 'L':
        s, e, w = int(LM.LEFT_SHOULDER), int(LM.LEFT_ELBOW), int(LM.LEFT_WRIST)
    else:
        s, e, w = int(LM.RIGHT_SHOULDER), int(LM.RIGHT_ELBOW), int(LM.RIGHT_WRIST)

    ls, rs, nose = int(LM.LEFT_SHOULDER), int(LM.RIGHT_SHOULDER), int(LM.NOSE)

    if min(vis[[s,e,w]]) <= VIS_T:
        return None, {"ok":False}

    # Face proxy
    if vis[nose] > VIS_T:
        face_pt = xy[nose]
    elif vis[ls] > VIS_T and vis[rs] > VIS_T:
        face_pt = 0.5*(xy[ls] + xy[rs])
    else:
        return None, {"ok":False}

    elbow = angle_deg(xy[s], xy[e], xy[w])
    sw    = dist(xy[s], xy[w]) + 1e-6  # shoulder-to-wrist
    wf    = dist(xy[w], face_pt)       # wrist-to-face

    # Frame-wise classification
    label = None
    if (elbow >= JAB_ELBOW_MIN_DEG) and (wf >= JAB_FACE_MIN_FRAC * sw):
        label = f"JAB/CROSS-{side}"
    elif (HOOK_ELBOW_MIN_DEG <= elbow <= HOOK_ELBOW_MAX_DEG) and (wf <= HOOK_FACE_MAX_FRAC * sw):
        label = f"HOOK-{side}"

    return label, {
        "ok":True, "elbow_deg":elbow, "wrist_face":wf, "shoulder_wrist":sw,
        "wf_over_sw": wf/sw
    }

def put_metrics(img, x, y, side, m, label):
    if not m.get("ok", False): return
    tcolor = (255,255,255)
    lcolor = (40,255,60) if (label and "JAB/CROSS" in label) else ((60,200,255) if (label and "HOOK" in label) else (200,200,200))
    cv2.putText(img, f"{side}: {label or '-'}", (x,y), FONT, 0.7, lcolor, 2, cv2.LINE_AA)
    y += 22
    cv2.putText(img, f"elbow: {m['elbow_deg']:.1f} deg", (x,y), FONT, 0.6, tcolor, 1, cv2.LINE_AA); y+=20
    cv2.putText(img, f"wrist-face: {m['wrist_face']:.1f}px", (x,y), FONT, 0.6, tcolor, 1, cv2.LINE_AA); y+=20
    cv2.putText(img, f"shoulder-wrist: {m['shoulder_wrist']:.1f}px", (x,y), FONT, 0.6, tcolor, 1, cv2.LINE_AA); y+=20
    cv2.putText(img, f"wf/sw: {m['wf_over_sw']:.2f}", (x,y), FONT, 0.6, tcolor, 1, cv2.LINE_AA)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Continuous elbow & wrist-face metrics; frame-wise Hook vs Jab/Cross")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="annotated.mp4")
    ap.add_argument("--model_complexity", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=SMOOTH_ALPHA, help="EMA smoothing [0..1]")
    ap.add_argument("--max_frames", type=int, default=0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened(): raise SystemExit(f"Cannot open: {args.video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W,H))
    pose = POSE.Pose(static_image_mode=False, model_complexity=args.model_complexity, enable_segmentation=False)

    prev_xy = None
    counts = {"HOOK-L":0,"HOOK-R":0,"JAB/CROSS-L":0,"JAB/CROSS-R":0}

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if args.max_frames and i >= args.max_frames: break

        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        out = frame.copy()

        if res.pose_landmarks:
            xy, vis = to_xyv(res.pose_landmarks.landmark, W, H)
            xy_s = ema(prev_xy, xy, args.alpha) if args.alpha>0 else xy
            prev_xy = xy_s

            # Draw skeleton
            draw_pose(out, xy_s, vis)

            # Per-side classification every frame
            labelL, mL = classify_frame_side('L', xy_s, vis)
            labelR, mR = classify_frame_side('R', xy_s, vis)

            # Count (simple running totals of frames meeting criteria)
            if labelL: counts[labelL] += 1
            if labelR: counts[labelR] += 1

            # HUD
            put_metrics(out, 12, 28, "L", mL, labelL)
            put_metrics(out, W-300, 28, "R", mR, labelR)

        writer.write(out)
        i += 1

    writer.release()
    cap.release()
    print(f"Saved: {args.out}")
    print("Frame-wise counts (frames meeting criteria):")
    for k,v in counts.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
