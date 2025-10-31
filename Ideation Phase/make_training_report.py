# make_training_report.py
# Generate an HTML report for stance detection using (and adapting to)
# training_pace_adjusted.py. Falls back to safe defaults if attributes
# are missing in the imported module.

import os
import io
import sys
import base64
import math
import argparse
from collections import deque, defaultdict

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

# --- Import your detector module
import training_pace_adjusted as tpa

# --------------------------- Defensive config ----------------------------
def cfg(name, default):
    return getattr(tpa, name, default)

# Defaults mirror the last full file I provided
FAST_MODE_DEFAULT = True
W_SHOULDER = cfg("W_SHOULDER", 1.5 if FAST_MODE_DEFAULT else 1.2)
W_HANDS    = cfg("W_HANDS",    0.6 if FAST_MODE_DEFAULT else 0.9)
W_HEAD     = cfg("W_HEAD",     0.45 if FAST_MODE_DEFAULT else 0.5)
FWD_BLEND  = cfg("FWD_BLEND",  0.50 if FAST_MODE_DEFAULT else 0.35)
WIN        = cfg("WIN",        140 if FAST_MODE_DEFAULT else 200)
BLUR_FLOOR = cfg("BLUR_FLOOR", 4.0 if FAST_MODE_DEFAULT else 5.0)
BLUR_RATIO = cfg("BLUR_RATIO", 0.08 if FAST_MODE_DEFAULT else 0.10)
DOWNSCALE_FACTOR = cfg("DOWNSCALE_FACTOR", 0.85 if FAST_MODE_DEFAULT else 1.0)

# Hysteresis defaults
FLIP_UP   = cfg("FLIP_UP", 0.35)
FLIP_DOWN = cfg("FLIP_DOWN", 0.20)
MIN_DWELL_FRAMES = cfg("MIN_DWELL_FRAMES", 30 if FAST_MODE_DEFAULT else 20)

# Mapping & math helpers
FULL2KEY = cfg("FULL2KEY", {
    "LEFT_SHOULDER": "LS", "RIGHT_SHOULDER": "RS",
    "LEFT_ELBOW": "LE", "RIGHT_ELBOW": "RE",
    "LEFT_WRIST": "LW", "RIGHT_WRIST": "RW",
    "LEFT_HIP": "LH", "RIGHT_HIP": "RH",
    "LEFT_ANKLE": "LA", "RIGHT_ANKLE": "RA",
    "NOSE": "NOSE", "LEFT_EAR": "LEAR", "RIGHT_EAR": "REAR",
})

# Reuse functions if present, else local fallbacks
depth_front   = cfg("depth_front",   lambda r,l: (0.0, 0.0))
proj_front    = cfg("proj_front",    lambda r,l,o,f: (0.0, 0.0))
estimate_forward = cfg("estimate_forward", lambda LHIP, RHIP, LSH, RSH: (np.array([0,0,0],np.float32), np.array([0,0,1],np.float32)))
head_yaw_score   = cfg("head_yaw_score",  lambda face_lms: (0.0, 0.0))
all_present      = cfg("all_present",     lambda *vals: all(v is not None for v in vals))

def decide_with_hysteresis_fallback(prev, val, frame_idx, last_change):
    cand = "SOUTHPAW" if val > 0 else "ORTHODOX" if val < 0 else "UNKNOWN"
    if cand == "UNKNOWN":
        return prev, last_change
    if prev == "UNKNOWN":
        if abs(val) >= FLIP_DOWN:
            return cand, frame_idx
        return prev, last_change
    if cand == prev:
        if abs(val) < FLIP_DOWN:
            return prev, last_change
        return prev, last_change
    strong_enough = abs(val) >= FLIP_UP
    long_enough   = (frame_idx - last_change) >= MIN_DWELL_FRAMES
    if strong_enough and long_enough:
        return cand, frame_idx
    return prev, last_change

decide_with_hysteresis = cfg("decide_with_hysteresis", decide_with_hysteresis_fallback)

# Smoothing: use tpa.LMFilter if available, else a compatible local class
if hasattr(tpa, "LMFilter") and hasattr(tpa, "OneEuro"):
    LMFilter = tpa.LMFilter
else:
    class OneEuro:
        def __init__(self, freq, min_cutoff=1.2, beta=0.03, d_cutoff=1.0):
            self.freq=freq; self.min_cutoff=min_cutoff; self.beta=beta; self.d_cutoff=d_cutoff
            self.x_prev=None; self.dx_prev=0.0
        def _alpha(self, cutoff):
            tau = 1.0/(2*math.pi*cutoff); te = 1.0/max(self.freq,1e-6)
            return 1.0/(1.0 + tau/te)
        def filter(self, x):
            dx = 0.0 if self.x_prev is None else (x - self.x_prev)*self.freq
            a_d = self._alpha(self.d_cutoff)
            dx_hat = a_d*dx + (1-a_d)*self.dx_prev
            cutoff = self.min_cutoff + self.beta*abs(dx_hat)
            a = self._alpha(cutoff)
            x_hat = x if self.x_prev is None else a*x + (1-a)*self.x_prev
            self.x_prev, self.dx_prev = x_hat, dx_hat
            return x_hat
    class LMFilter:
        def __init__(self, fps):
            # Use fast-mode-ish defaults
            self.fx=OneEuro(fps, 1.5, 0.060); self.fy=OneEuro(fps, 1.5, 0.060); self.fz=OneEuro(fps, 1.2, 0.040)
        def apply(self, p):
            if p is None: return None
            return np.array([self.fx.filter(p[0]), self.fy.filter(p[1]), self.fz.filter(p[2]), p[3]], dtype=np.float32)

# --------------------------- Small helpers -------------------------------
def b64_png(fig):
    bio = io.BytesIO()
    fig.savefig(bio, format="png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    return base64.b64encode(bio.getvalue()).decode("ascii")

def write_csv(rows, out_csv):
    import csv
    keys = [
        "frame", "time_s", "stance_val", "stance_label",
        "quality", "blur_var", "shoulder_s", "forearm_s", "head_s",
        "weight_total"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

def percent(n, d):
    return 0.0 if d <= 0 else 100.0 * (n / d)

def dwell_stats(labels, fps):
    segs = []
    if not labels: return {"count":0, "avg_s":0, "med_s":0, "max_s":0, "segs":[]}
    cur = labels[0]; length = 1
    for lab in labels[1:]:
        if lab == cur: length += 1
        else:
            if cur != "UNKNOWN": segs.append(length)
            cur = lab; length = 1
    if cur != "UNKNOWN": segs.append(length)
    if not segs: return {"count":0, "avg_s":0, "med_s":0, "max_s":0, "segs":[]}
    arr = np.array(segs, dtype=float)
    return {
        "count": len(segs),
        "avg_s": float(arr.mean()/fps),
        "med_s": float(np.median(arr)/fps),
        "max_s": float(arr.max()/fps),
        "segs": segs
    }

# ----------------------------- Core analysis -----------------------------
def analyze_video(video_path, save_frames_csv=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = N / fps if fps > 0 else 0

    # MediaPipe instances
    mp_pose = mp.solutions.pose
    mp_face = mp.solutions.face_mesh
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,  # heavy
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    face = mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    # Filters
    def make_filters(fps_):
        keys = set(FULL2KEY.values())
        return {k: LMFilter(fps_) for k in keys}
    filters = make_filters(fps)

    # Local helpers to mirror tpa’s LM() and get_lm()
    PLM = mp_pose.PoseLandmark
    def get_lm(plms, name, vis_thresh):
        idx = PLM[name].value
        pt = plms.landmark[idx]
        if pt.visibility < vis_thresh:
            return None
        return np.array([pt.x, pt.y, pt.z, pt.visibility], dtype=np.float32)

    def LM(plms, name, vis=0.25):
        p = get_lm(plms, name, vis)
        if p is None:
            return None
        key = FULL2KEY.get(name)
        if key is None:
            return p
        f = filters[key]
        return f.apply(p)

    # Rolling / hysteresis
    votes = deque(maxlen=WIN)
    wts   = deque(maxlen=WIN)
    last_output_stance = "UNKNOWN"
    last_change_frame  = -10**9

    # Blur quality history
    blur_hist = deque(maxlen=120)

    # Output rows
    rows = []

    frame = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame += 1
        t_s = frame / fps

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        lv = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur_hist.append(lv)
        med = float(np.median(blur_hist)) if blur_hist else lv
        dyn_thresh = max(BLUR_FLOOR, BLUR_RATIO * (med if med > 1e-6 else lv))
        quality = 1.0 / (1.0 + np.exp(-(lv - dyn_thresh) / 10.0))

        # Optional downscale (matches tpa behavior)
        if DOWNSCALE_FACTOR < 1.0:
            Wd = max(2, int(W * DOWNSCALE_FACTOR))
            Hd = max(2, int(H * DOWNSCALE_FACTOR))
            small_rgb = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                                   (Wd, Hd), interpolation=cv2.INTER_AREA)
            pose_res = pose.process(small_rgb)
            face_res = face.process(small_rgb)
        else:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pose_res = pose.process(frame_rgb)
            face_res = face.process(frame_rgb)

        score_total = 0.0
        weight_total = 0.0
        s_shoulder = 0.0
        s_forearm  = 0.0
        s_head     = 0.0

        if pose_res.pose_landmarks:
            plms = pose_res.pose_landmarks

            LSH = LM(plms, "LEFT_SHOULDER", 0.30);  RSH = LM(plms, "RIGHT_SHOULDER", 0.30)
            LE  = LM(plms, "LEFT_ELBOW",    0.20);  RE  = LM(plms, "RIGHT_ELBOW",    0.20)
            LW  = LM(plms, "LEFT_WRIST",    0.20);  RW  = LM(plms, "RIGHT_WRIST",    0.20)
            LH  = LM(plms, "LEFT_HIP",      0.20);  RH  = LM(plms, "RIGHT_HIP",      0.20)

            if all_present(LSH, RSH, LH, RH):
                pelvis, fwd = estimate_forward(LH, RH, LSH, RSH)
            else:
                pelvis, fwd = np.array([0,0,0], dtype=np.float32), np.array([0,0,1], dtype=np.float32)

            # Quality-adjusted effective weights (same policy as detector)
            W_SHOULDER_eff = W_SHOULDER * (0.70 + 0.30 * quality)
            W_HANDS_eff    = W_HANDS    * (0.20 + 0.80 * quality)
            W_HEAD_eff     = W_HEAD     * (0.40 + 0.60 * quality)

            # 1) Shoulders
            s_sd, w_sd = depth_front(RSH, LSH)
            s_shoulder = s_sd
            score_total  += W_SHOULDER_eff * s_sd
            weight_total += W_SHOULDER_eff * w_sd

            # 2) Forearms (wrists fallback to elbows)
            def limb_pair(primary_R, primary_L, fallback_R, fallback_L):
                R = primary_R if (primary_R is not None and primary_R[3] > 0.35) else fallback_R
                L = primary_L if (primary_L is not None and primary_L[3] > 0.35) else fallback_L
                return R, L
            Rf, Lf = limb_pair(RW, LW, RE, LE)
            s_fd, w_fd = depth_front(Rf, Lf)
            s_fp, w_fp = proj_front(Rf, Lf, pelvis, fwd)
            s_fore = (1.0 - FWD_BLEND) * s_fd + FWD_BLEND * s_fp
            s_forearm = s_fore
            w_fore = (w_fd + w_fp)
            score_total  += W_HANDS_eff * s_fore
            weight_total += W_HANDS_eff * w_fore

            # 3) Head yaw
            if face_res.multi_face_landmarks:
                s_head, w_head = head_yaw_score(face_res.multi_face_landmarks)
                s_head = s_head
                score_total  += W_HEAD_eff * s_head
                weight_total += W_HEAD_eff * w_head

        # stance vote + hysteresis
        if weight_total > 0:
            s_inst = score_total / max(1e-6, weight_total)
            votes.append(float(np.clip(s_inst, -1, 1)))
            wts.append(weight_total)
            stance_val = float(np.average(votes, weights=wts))
        else:
            stance_val = 0.0

        last_output_stance, last_change_frame = decide_with_hysteresis(
            last_output_stance, stance_val, frame, last_change_frame
        )
        stance_label = last_output_stance

        rows.append({
            "frame": frame,
            "time_s": t_s,
            "stance_val": stance_val,
            "stance_label": stance_label,
            "quality": quality,
            "blur_var": lv,
            "shoulder_s": s_shoulder,
            "forearm_s": s_forearm,
            "head_s": s_head,
            "weight_total": weight_total,
        })

    cap.release()

    # ---- Aggregates ----
    total_frames = len(rows)
    south = sum(1 for r in rows if r["stance_label"] == "SOUTHPAW")
    orth  = sum(1 for r in rows if r["stance_label"] == "ORTHODOX")
    unk   = sum(1 for r in rows if r["stance_label"] == "UNKNOWN")
    flips = 0
    prev_label = None
    for r in rows:
        lab = r["stance_label"]
        if prev_label is not None and lab != prev_label and "UNKNOWN" not in (lab, prev_label):
            flips += 1
        prev_label = lab

    dwell = dwell_stats([r["stance_label"] for r in rows], fps)

    # ---- Plots (embedded as base64) ----
    times = np.array([r["time_s"] for r in rows], dtype=float) if rows else np.array([])
    score = np.array([r["stance_val"] for r in rows], dtype=float) if rows else np.array([])
    qual  = np.array([r["quality"] for r in rows], dtype=float) if rows else np.array([])

    def plot_to_b64(x, y, title, yl):
        fig = plt.figure(figsize=(9, 3))
        if len(x): plt.plot(x, y, linewidth=1.5)
        if yl == "Score": plt.axhline(0, linestyle="--", linewidth=1)
        plt.title(title); plt.xlabel("Time (s)"); plt.ylabel(yl)
        return b64_png(fig)

    img1 = plot_to_b64(times, score, "Stance Score Over Time (+ = Southpaw, - = Orthodox)", "Score")
    img2 = plot_to_b64(times, qual,  "Frame Quality Over Time", "Quality (0–1)")

    fig3 = plt.figure(figsize=(6, 3.5))
    if dwell["segs"]:
        plt.hist(np.array(dwell["segs"])/fps, bins=15)
    plt.title("Dwell Duration Histogram"); plt.xlabel("Dwell (s)"); plt.ylabel("Count")
    img3 = b64_png(fig3)

    # ---- CSV output ----
    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_csv = f"{stem}_frames.csv"
    if rows and save_frames_csv:
        write_csv(rows, out_csv)

    # ---- HTML ----
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Muay Thai Stance Report – {stem}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }}
    h1 {{ margin-bottom: 0; }}
    .muted {{ color: #666; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(280px, 1fr)); gap: 18px; }}
    .card {{ border: 1px solid #eee; border-radius: 10px; padding: 14px; }}
    table.summary td {{ padding: 4px 12px 4px 0; vertical-align: top; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 6px; }}
    code {{ background: #f6f6f6; padding: 2px 6px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Muay Thai Stance Evaluation</h1>
  <div class="muted">{stem} — {W}×{H} @ {fps:.2f} FPS — duration {dur:.2f}s — frames {total_frames}</div>

  <div class="grid" style="margin-top:18px;">
    <div class="card">
      <h3>Summary</h3>
      <table class="summary">
        <tr><td>Southpaw time</td><td><strong>{percent(south, total_frames):.1f}%</strong> ({south} frames)</td></tr>
        <tr><td>Orthodox time</td><td><strong>{percent(orth,  total_frames):.1f}%</strong> ({orth} frames)</td></tr>
        <tr><td>Unknown time</td><td><strong>{percent(unk,   total_frames):.1f}%</strong> ({unk} frames)</td></tr>
        <tr><td>Stance flips</td><td><strong>{flips}</strong></td></tr>
        <tr><td>Dwell avg / med / max</td><td><strong>{dwell['avg_s']:.2f}s</strong> / {dwell['med_s']:.2f}s / {dwell['max_s']:.2f}s</td></tr>
        <tr><td>Frames CSV</td><td><code>{out_csv}</code></td></tr>
      </table>
    </div>

    <div class="card">
      <h3>Stance Score</h3>
      <img src="data:image/png;base64,{img1}">
    </div>

    <div class="card">
      <h3>Frame Quality</h3>
      <img src="data:image/png;base64,{img2}">
    </div>

    <div class="card">
      <h3>Dwell Histogram</h3>
      <img src="data:image/png;base64,{img3}">
    </div>
  </div>
</body>
</html>
"""
    out_html = f"{stem}_report.html"
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

    return {
        "video": video_path,
        "fps": fps, "W": W, "H": H, "N": N, "duration": dur,
        "frames_csv": out_csv if rows else None,
        "report_html": out_html,
        "southpaw_pct": percent(south, total_frames),
        "orthodox_pct": percent(orth, total_frames),
        "unknown_pct": percent(unk, total_frames),
        "flips": flips,
        "dwell": dwell,
    }

# ------------------------------- CLI -------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video")
    args = ap.parse_args()
    res = analyze_video(args.video)
    print("[OK] Report:", res["report_html"])
    if res["frames_csv"]:
        print("[OK] Per-frame metrics:", res["frames_csv"])

if __name__ == "__main__":
    main()
