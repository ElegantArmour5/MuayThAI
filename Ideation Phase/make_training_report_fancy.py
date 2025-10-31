# make_training_report_fancy.py
# Fancy, cohesive HTML report with robust thumbnail embedding and responsive grid layout.
# - Imports training_pace_adjusted.py (stance logic & helpers)
# - Detects strikes/feints, saves thumbnails as PNG files (absolute paths in CSV)
# - HTML embeds thumbnails as base64 (so they always show)
# - Responsive grid: visuals align & scale relative to each other
#
# Usage:
#   python make_training_report_fancy.py --video ".\pov.mp4"
#   python make_training_report_fancy.py --video ".\pov.mp4" --stride 2 --fast --downscale 0.85
#   python make_training_report_fancy.py --video ".\pov.mp4" --force

import os, io, sys, math, csv, time, base64, argparse
from collections import deque, Counter

# --- raise CSV field size limit (defensive) ---
try:
    import sys as _sys, csv as _csv
    _size = _sys.maxsize
    while True:
        try:
            _csv.field_size_limit(_size)
            break
        except OverflowError:
            _size = int(_size / 10)
except Exception:
    pass

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import mediapipe as mp

import training_pace_adjusted as tpa  # your stance/quality helpers

# --------------------------- Config from tpa (safe) -----------------------
def cfg(name, default):
    return getattr(tpa, name, default)

FAST_MODE_DEFAULT = True
W_SHOULDER = cfg("W_SHOULDER", 1.5 if FAST_MODE_DEFAULT else 1.2)
W_HANDS    = cfg("W_HANDS",    0.6 if FAST_MODE_DEFAULT else 0.9)
W_HEAD     = cfg("W_HEAD",     0.45 if FAST_MODE_DEFAULT else 0.5)
FWD_BLEND  = cfg("FWD_BLEND",  0.50 if FAST_MODE_DEFAULT else 0.35)
WIN        = cfg("WIN",        140 if FAST_MODE_DEFAULT else 200)
BLUR_FLOOR = cfg("BLUR_FLOOR", 4.0 if FAST_MODE_DEFAULT else 5.0)
BLUR_RATIO = cfg("BLUR_RATIO", 0.08 if FAST_MODE_DEFAULT else 0.10)
DOWNSCALE_FACTOR_DEFAULT = cfg("DOWNSCALE_FACTOR", 0.85 if FAST_MODE_DEFAULT else 1.0)

FLIP_UP   = cfg("FLIP_UP", 0.35)
FLIP_DOWN = cfg("FLIP_DOWN", 0.20)
MIN_DWELL_FRAMES = cfg("MIN_DWELL_FRAMES", 30 if FAST_MODE_DEFAULT else 20)

FULL2KEY = cfg("FULL2KEY", {
    "LEFT_SHOULDER": "LS", "RIGHT_SHOULDER": "RS",
    "LEFT_ELBOW": "LE", "RIGHT_ELBOW": "RE",
    "LEFT_WRIST": "LW", "RIGHT_WRIST": "RW",
    "LEFT_HIP": "LH", "RIGHT_HIP": "RH",
    "LEFT_ANKLE": "LA", "RIGHT_ANKLE": "RA",
    "LEFT_KNEE": "LK", "RIGHT_KNEE": "RK",
    "NOSE": "NOSE", "LEFT_EAR": "LEAR", "RIGHT_EAR": "REAR",
})

# helpers possibly provided by tpa
estimate_forward = cfg("estimate_forward",
    lambda LHIP,RHIP,LSH,RSH: (np.array([0,0,0],np.float32), np.array([0,0,1],np.float32)))
depth_front      = cfg("depth_front",   lambda r,l: (0.0, 0.0))
proj_front_func  = cfg("proj_front",    lambda r,l,o,f: (0.0, 0.0))
head_yaw_score   = cfg("head_yaw_score",lambda f: (0.0, 0.0))
all_present      = cfg("all_present",   lambda *vals: all(v is not None for v in vals))
decide_with_hysteresis = cfg("decide_with_hysteresis", None)

# smoothing filters
if hasattr(tpa, "LMFilter"):
    LMFilter = tpa.LMFilter
else:
    class OneEuro:
        def __init__(self, freq, min_cutoff=1.5, beta=0.06, d_cutoff=1.0):
            self.freq=freq; self.min_cutoff=min_cutoff; self.beta=beta; self.d_cutoff=d_cutoff
            self.x_prev=None; self.dx_prev=0.0
        def _alpha(self, cutoff):
            tau=1.0/(2*math.pi*cutoff); te=1.0/max(self.freq,1e-6)
            return 1.0/(1.0 + tau/te)
        def filter(self, x):
            dx=0.0 if self.x_prev is None else (x-self.x_prev)*self.freq
            a_d=self._alpha(self.d_cutoff); dx_hat=a_d*dx + (1-a_d)*self.dx_prev
            cutoff=self.min_cutoff + self.beta*abs(dx_hat); a=self._alpha(cutoff)
            x_hat=x if self.x_prev is None else a*x + (1-a)*self.x_prev
            self.x_prev, self.dx_prev = x_hat, dx_hat
            return x_hat
    class LMFilter:
        def __init__(self, fps):
            self.fx=OneEuro(fps,1.5,0.06); self.fy=OneEuro(fps,1.5,0.06); self.fz=OneEuro(fps,1.2,0.04)
        def apply(self, p):
            if p is None: return None
            return np.array([self.fx.filter(p[0]), self.fy.filter(p[1]), self.fz.filter(p[2]), p[3]], dtype=np.float32)

# --------------------------- Visual style (colors) ------------------------
plt.rcParams.update({
    "figure.facecolor": "#0e1116",
    "axes.facecolor":   "#0e1116",
    "savefig.facecolor":"#0e1116",
    "text.color":       "#d6e3ff",
    "axes.labelcolor":  "#d6e3ff",
    "axes.edgecolor":   "#2a2f3a",
    "xtick.color":      "#a9b5d9",
    "ytick.color":      "#a9b5d9",
    "grid.color":       "#2a2f3a",
})

PALETTE = {
    "accent":  "#66e2d5",
    "accent2": "#ff8f6b",
    "accent3": "#b088ff",
    "accent4": "#ffd166",
    "muted":   "#6c7a99",
    "good":    "#7bd389",
    "warn":    "#ffcc66",
    "bad":     "#ff6b6b",
}

CMAP_HEAT = LinearSegmentedColormap.from_list(
    "heatish", ["#0e1116", "#2f3b52", "#395a8a", "#6aa9ff", "#a9f8ff", "#ffffff"], N=256
)
CMAP_IMPACT = LinearSegmentedColormap.from_list(
    "impactish", ["#223", "#345", "#2aa", "#6df", "#fff"], N=256
)

# --------------------------- Utilities ------------------------------------
def laplacian_var(gray): return cv2.Laplacian(gray, cv2.CV_64F).var()

def b64_png(fig):
    bio = io.BytesIO()
    fig.savefig(bio, format="png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    return base64.b64encode(bio.getvalue()).decode("ascii")

def write_csv(rows, out_csv, fields):
    with open(out_csv,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k:r.get(k,"") for k in fields})

def read_csv(path):
    rows=[]
    with open(path,"r",encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows

def norm(v, eps=1e-6):
    n=np.linalg.norm(v); return v/(n+eps)

def joint_vec(a, b):
    if a is None or b is None: return None
    return b[:3] - a[:3]

def angle_between(u, v):
    if u is None or v is None: return None
    du, dv = norm(u), norm(v)
    return math.degrees(math.acos(float(np.clip(np.dot(du, dv), -1.0, 1.0))))

def end_effector_speed(prev_pt, curr_pt, fps, W, H):
    if prev_pt is None or curr_pt is None: return None
    dx = (curr_pt[0]-prev_pt[0])*W; dy=(curr_pt[1]-prev_pt[1])*H
    pix_per_s = math.hypot(dx, dy)*fps
    diag = math.hypot(W, H)
    return (pix_per_s / max(1e-6, diag))

def local_divergence(prev_gray, gray, cx, cy, win=24):
    if prev_gray is None or gray is None: return 0.0
    h,w = gray.shape[:2]
    x0=max(0,int(cx-win)); x1=min(w,int(cx+win))
    y0=max(0,int(cy-win)); y1=min(h,int(cy+win))
    if x1-x0<4 or y1-y0<4: return 0.0
    flow = cv2.calcOpticalFlowFarneback(prev_gray[y0:y1,x0:x1], gray[y0:y1,x0:x1], None,
                                        0.5, 2, 13, 2, 5, 1.2, 0)
    fx, fy = flow[...,0], flow[...,1]
    du_dx = cv2.Sobel(fx, cv2.CV_32F, 1, 0, ksize=3)
    dv_dy = cv2.Sobel(fy, cv2.CV_32F, 0, 1, ksize=3)
    div = du_dx + dv_dy
    return float(np.mean(np.abs(div)))

def save_thumb_file(frame_bgr, out_path, max_w=380):
    h, w = frame_bgr.shape[:2]
    if w > max_w:
        scale = max_w / float(w)
        frame_bgr = cv2.resize(frame_bgr, (max_w, int(h*scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_path, frame_bgr)

def read_thumb_b64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except Exception:
        return ""

def resolve_thumb_path(path, thumbs_dir, report_dir, script_dir):
    """Try multiple locations to find the thumbnail file."""
    cand = []
    if path:
        # if already absolute, try as-is
        if os.path.isabs(path):
            cand.append(path)
        # try relative forms
        cand.append(os.path.join(report_dir, os.path.basename(path)))
        cand.append(os.path.join(thumbs_dir, os.path.basename(path)))
        cand.append(os.path.join(script_dir, os.path.basename(path)))
        cand.append(path)  # original
    for p in cand:
        if os.path.exists(p):
            return os.path.abspath(p)
    return ""  # not found

# --------------------------- Simple classifiers ---------------------------
def classify_punch(shoulder, elbow, wrist, lateral_axis):
    if wrist is None or elbow is None or shoulder is None:
        return None, None
    side = "R" if (wrist[0] > shoulder[0]) else "L"
    u_se = joint_vec(shoulder, elbow)
    u_ew = joint_vec(elbow, wrist)
    if u_se is None or u_ew is None: return None, side
    elbow_angle = angle_between(-u_se, u_ew)
    if elbow_angle is None: return None, side
    horiz = abs(np.dot(norm(u_ew), norm(lateral_axis)))
    vertical = abs(np.dot(norm(u_ew), np.array([0,1,0],dtype=np.float32)))
    if elbow_angle > 160: label = "straight"
    elif horiz > 0.7:     label = "hook"
    elif vertical > 0.6:  label = "uppercut"
    else:                 label = "straight"
    return label, side

def classify_kick(hip, knee, ankle, fwd_axis):
    if ankle is None or knee is None or hip is None:
        return None, None
    side = "R" if ankle[0] > hip[0] else "L"
    u_hk = joint_vec(hip, knee)
    u_ka = joint_vec(knee, ankle)
    if u_hk is None or u_ka is None: return None, side
    forward_comp = abs(np.dot(norm(u_ka), norm(fwd_axis)))
    label = "teep" if forward_comp > 0.6 else "round"
    return label, side

# --------------------------- Analysis pass --------------------------------
def analyze(video_path, stride=2, downscale=None, fast=False, max_seconds=None, force=False,
            save_frames_csv=True, save_events_csv=True):
    stem = os.path.splitext(os.path.basename(video_path))[0]
    frames_csv = f"{stem}_frames.csv"
    events_csv = f"{stem}_events.csv"
    thumbs_dir = os.path.abspath(f"{stem}_thumbs")
    os.makedirs(thumbs_dir, exist_ok=True)

    reuse_frames = (os.path.exists(frames_csv) and not force)
    reuse_events = (os.path.exists(events_csv) and not force)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise RuntimeError(f"Could not open: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_to_do = N if max_seconds is None else min(N, int(max_seconds*fps))
    ds = (downscale if downscale is not None else DOWNSCALE_FACTOR_DEFAULT)

    # Fast path: reuse CSVs if present
    if reuse_frames and reuse_events:
        rows = []
        for r in read_csv(frames_csv):
            rows.append({
                "frame": int(float(r["frame"])),
                "time_s": float(r["time_s"]),
                "stance_val": float(r["stance_val"]),
                "stance_label": r["stance_label"],
                "quality": float(r["quality"]),
                "blur_var": float(r["blur_var"]),
                "shoulder_s": float(r.get("shoulder_s", 0.0) or 0.0),
                "forearm_s": float(r.get("forearm_s", 0.0) or 0.0),
                "head_s": float(r.get("head_s", 0.0) or 0.0),
                "weight_total": float(r.get("weight_total", 0.0) or 0.0)
            })
        evs = []
        for e in read_csv(events_csv):
            # Keep thumb_path as-is; embed during HTML build
            evs.append({
                "t": float(e["t"]),
                "frame": int(float(e["frame"])),
                "type": e["type"],
                "side": e["side"],
                "label": e["label"],
                "confidence": float(e["confidence"]),
                "impact_score": float(e["impact_score"]),
                "duration_s": float(e["duration_s"]),
                "max_speed": float(e["max_speed"]),
                "q": float(e["q"]),
                "blur": float(e["blur"]),
                "notes": e.get("notes",""),
                "thumb_path": e.get("thumb_path",""),
                "gx": float(e.get("gx","0") or 0.0),
                "gy": float(e.get("gy","0") or 0.0),
            })
        cap.release()
        return {"rows": rows, "events": evs, "fps": fps, "W": W, "H": H, "N": N, "stem": stem, "ds": ds, "fast": fast, "thumbs_dir": thumbs_dir}

    # Slow path: run detectors
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=(1 if fast else 2),
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )
    PLM = mp_pose.PoseLandmark

    # filters for key lms
    keys = set(FULL2KEY.values()) | set(["LK","RK"])
    filters = {k: LMFilter(fps) for k in keys}
    def get_lm(plms, name, vis_thresh):
        idx = PLM[name].value
        pt = plms.landmark[idx]
        if pt.visibility < vis_thresh: return None
        return np.array([pt.x, pt.y, pt.z, pt.visibility], dtype=np.float32)
    def LM(plms, name, vis=0.25):
        p = get_lm(plms, name, vis)
        if p is None: return None
        key = FULL2KEY.get(name, name[:2])
        f = filters.get(key, None)
        if f is None: filters[key]=LMFilter(fps); f=filters[key]
        return f.apply(p)

    # stance rolling vote
    votes = deque(maxlen=WIN); wts = deque(maxlen=WIN)
    last_output_stance = "UNKNOWN"; last_change_frame = -10**9
    blur_hist = deque(maxlen=120)

    rows, events = [], []
    prev_pts = {"RW": None, "LW": None, "RA": None, "LA": None}
    prev_gray = None

    class LimbFSM:
        IDLE, LOAD, EXTEND, CONTACT, RETRACT = range(5)
        def __init__(self, name): 
            self.name=name; self.state=self.IDLE; self.t0=None; self.max_speed=0.0; self.ext_frames=0
    fsm = {k: LimbFSM(k) for k in ["RW","LW","RA","LA"]}

    start_t = time.time()
    i = 0
    while True:
        pos = int(i*stride)
        if pos >= total_to_do: break
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame_bgr = cap.read()
        if not ok: break
        frame_idx = pos + 1
        t_s = frame_idx / fps
        i += 1

        if i % max(1,int(fps/stride)) == 0:
            print(f"[fancy] {frame_idx}/{total_to_do} frames | elapsed {time.time()-start_t:.1f}s", flush=True)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        lv = float(laplacian_var(gray))
        blur_hist.append(lv)
        med = float(np.median(blur_hist)) if blur_hist else lv
        dyn_thresh = max(BLUR_FLOOR, BLUR_RATIO*(med if med>1e-6 else lv))
        quality = 1.0 / (1.0 + np.exp(-(lv - dyn_thresh)/10.0))

        # pose run
        if ds < 1.0:
            Wd = max(2, int(W*ds)); Hd = max(2, int(H*ds))
            small_rgb = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                                   (Wd,Hd), interpolation=cv2.INTER_AREA)
            pose_res = pose.process(small_rgb)
        else:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pose_res = pose.process(frame_rgb)

        score_total = 0.0
        weight_total = 0.0
        s_shoulder = 0.0
        s_forearm  = 0.0
        s_head     = 0.0

        # key landmarks
        LSH=RSH=LE=RE=LW=RW=LH=RH=LA=RA=LK=RK=None
        pelvis = np.array([0,0,0],np.float32); fwd=np.array([0,0,1],np.float32); lateral=np.array([1,0,0],np.float32)

        if pose_res.pose_landmarks:
            plms = pose_res.pose_landmarks
            LSH = LM(plms, "LEFT_SHOULDER", 0.30);  RSH = LM(plms, "RIGHT_SHOULDER", 0.30)
            LE  = LM(plms, "LEFT_ELBOW",    0.20);  RE  = LM(plms, "RIGHT_ELBOW",    0.20)
            LW  = LM(plms, "LEFT_WRIST",    0.20);  RW  = LM(plms, "RIGHT_WRIST",    0.20)
            LH  = LM(plms, "LEFT_HIP",      0.20);  RH  = LM(plms, "RIGHT_HIP",      0.20)
            LK  = LM(plms, "LEFT_KNEE",     0.10);  RK  = LM(plms, "RIGHT_KNEE",     0.10)
            LA  = LM(plms, "LEFT_ANKLE",    0.10);  RA  = LM(plms, "RIGHT_ANKLE",    0.10)

            # axes
            if all_present(LSH,RSH,LH,RH):
                pelvis, fwd = estimate_forward(LH,RH,LSH,RSH)
                lateral = norm(LSH[:3]-RSH[:3])

            # quality-adjusted weights
            W_SHOULDER_eff = W_SHOULDER * (0.70 + 0.30*quality)
            W_HANDS_eff    = W_HANDS    * (0.20 + 0.80*quality)
            W_HEAD_eff     = W_HEAD     * (0.40 + 0.60*quality)

            # shoulders
            s_sd, w_sd = depth_front(RSH, LSH)
            s_shoulder = s_sd
            score_total  += W_SHOULDER_eff * s_sd
            weight_total += W_SHOULDER_eff * w_sd

            # forearms (wrists fallback to elbows)
            def limb_pair(primary_R, primary_L, fallback_R, fallback_L):
                R = primary_R if (primary_R is not None and primary_R[3] > 0.35) else fallback_R
                L = primary_L if (primary_L is not None and primary_L[3] > 0.35) else fallback_L
                return R, L
            Rf, Lf = limb_pair(RW,LW,RE,LE)
            s_fd, w_fd = depth_front(Rf, Lf)
            s_fp, w_fp = proj_front_func(Rf, Lf, pelvis, fwd)
            s_fore = (1.0 - FWD_BLEND) * s_fd + FWD_BLEND * s_fp
            s_forearm = s_fore
            w_fore = (w_fd + w_fp)
            score_total  += W_HANDS_eff * s_fore
            weight_total += W_HANDS_eff * w_fore

        # stance rollup
        if weight_total > 0:
            s_inst = score_total / max(1e-6, weight_total)
            votes.append(float(np.clip(s_inst, -1, 1))); wts.append(weight_total)
            stance_val = float(np.average(votes, weights=wts))
        else:
            stance_val = 0.0

        if decide_with_hysteresis:
            last_output_stance, last_change_frame = decide_with_hysteresis(
                last_output_stance, stance_val, frame_idx, last_change_frame
            )
        else:
            cand = "SOUTHPAW" if stance_val>0 else "ORTHODOX" if stance_val<0 else "UNKNOWN"
            if cand != "UNKNOWN": last_output_stance = cand
        stance_label = last_output_stance

        rows.append({
            "frame": frame_idx, "time_s": t_s,
            "stance_val": stance_val, "stance_label": stance_label,
            "quality": quality, "blur_var": lv,
            "shoulder_s": s_shoulder, "forearm_s": s_forearm, "head_s": s_head,
            "weight_total": weight_total,
        })

        # --------- events (strike/feint) ----------
        pts = {"RW": RW, "LW": LW, "RA": RA, "LA": LA}
        speeds = {k: end_effector_speed(prev_pts[k], pts[k], fps, W, H) for k in pts}
        prev_pts = dict(pts)
        divs = {}
        for limb, P in pts.items():
            if P is not None:
                cx, cy = int(P[0]*W), int(P[1]*H)
                divs[limb] = local_divergence(prev_gray, gray, cx, cy, win=24)
            else:
                divs[limb] = 0.0

        TH_ONSET = 0.35
        TH_CONTACT_DECEL = 0.30
        TH_DIV = 0.40
        MAX_FEINT_DUR = int(0.20 * fps)

        class LimbFSMObj: pass  # silence type checkers
        for limb in ["RW","LW","RA","LA"]:
            s = speeds[limb]; dv = divs[limb]; f = fsm[limb]
            if s is None: continue
            f.max_speed = max(f.max_speed, s)
            if f.state == LimbFSM.IDLE:
                if s > TH_ONSET:
                    f.state=LimbFSM.LOAD; f.t0=frame_idx; f.ext_frames=0
            elif f.state == LimbFSM.LOAD:
                if s > f.max_speed * 0.9:
                    f.state=LimbFSM.EXTEND
                elif s < TH_ONSET*0.5:
                    f.state=LimbFSM.IDLE; f.max_speed=0.0
            elif f.state == LimbFSM.EXTEND:
                f.ext_frames += 1
                speed_drop = (f.max_speed - s) > TH_CONTACT_DECEL
                contact_like = speed_drop or dv > TH_DIV
                if contact_like:
                    # classify + store coords
                    if limb in ("RW","LW"):
                        if limb=="RW":
                            label, side = classify_punch(RSH, RE, RW, norm(LSH[:3]-RSH[:3]) if all_present(LSH,RSH) else np.array([1,0,0]))
                            gx, gy = (RW[0], RW[1]) if RW is not None else (0.0, 0.0)
                        else:
                            label, side = classify_punch(LSH, LE, LW, norm(LSH[:3]-RSH[:3]) if all_present(LSH,RSH) else np.array([1,0,0]))
                            gx, gy = (LW[0], LW[1]) if LW is not None else (0.0, 0.0)
                    else:
                        if limb=="RA":
                            label, side = classify_kick(RH, RK, RA, fwd)
                            gx, gy = (RA[0], RA[1]) if RA is not None else (0.0, 0.0)
                        else:
                            label, side = classify_kick(LH, LK, LA, fwd)
                            gx, gy = (LA[0], LA[1]) if LA is not None else (0.0, 0.0)

                    side = side or ("R" if limb[0]=="R" else "L")
                    label = label or ("punch" if limb in ("RW","LW") else "kick")
                    contact_signals = (1.0 if speed_drop else 0.0) + (1.0 if dv>TH_DIV else 0.0)
                    confidence = 0.5 + 0.25*contact_signals
                    impact = float(0.6*min(1.0, f.max_speed) + 0.4*min(1.0, dv))

                    # save thumbnail as file (absolute path)
                    thumb_name = f"{stem}_f{frame_idx}_{limb}.png"
                    thumb_path = os.path.abspath(os.path.join(thumbs_dir, thumb_name))
                    save_thumb_file(frame_bgr, thumb_path, max_w=380)

                    events.append({
                        "t": round(t_s,3), "frame": frame_idx,
                        "type": "strike", "side": side, "label": label,
                        "confidence": round(confidence,2), "impact_score": round(impact,2),
                        "duration_s": round((frame_idx - (f.t0 or frame_idx))/fps,3),
                        "max_speed": round(f.max_speed,3),
                        "q": round(quality,2), "blur": round(lv,1),
                        "notes": f"{limb}: drop={speed_drop}, div={dv:.2f}",
                        "thumb_path": thumb_path, "gx": round(float(gx),4), "gy": round(float(gy),4),
                    })
                    f.state=LimbFSM.RETRACT
                else:
                    if f.ext_frames>2 and s < TH_ONSET*0.6:
                        dur = frame_idx - (f.t0 or frame_idx)
                        if dur <= MAX_FEINT_DUR:
                            thumb_name = f"{stem}_f{frame_idx}_{limb}_feint.png"
                            thumb_path = os.path.abspath(os.path.join(thumbs_dir, thumb_name))
                            save_thumb_file(frame_bgr, thumb_path, max_w=380)
                            events.append({
                                "t": round(t_s,3), "frame": frame_idx,
                                "type": "feint", "side": ("R" if limb[0]=="R" else "L"),
                                "label": ("hand" if limb in ("RW","LW") else "leg"),
                                "confidence": 0.6, "impact_score": 0.0,
                                "duration_s": round(dur/fps,3),
                                "max_speed": round(f.max_speed,3),
                                "q": round(quality,2), "blur": round(lv,1),
                                "notes": f"{limb}: abort", "thumb_path": thumb_path,
                                "gx": 0.0, "gy": 0.0,
                            })
                        f.state=LimbFSM.RETRACT
            elif f.state == LimbFSM.RETRACT:
                if s < TH_ONSET*0.4:
                    f.state=LimbFSM.IDLE; f.max_speed=0.0; f.ext_frames=0

        prev_gray = gray

    cap.release()

    # Save caches
    if save_frames_csv and rows:
        write_csv(rows, frames_csv, ["frame","time_s","stance_val","stance_label","quality","blur_var","shoulder_s","forearm_s","head_s","weight_total"])
    if save_events_csv and events:
        write_csv(events, events_csv, ["t","frame","type","side","label","confidence","impact_score","duration_s","max_speed","q","blur","notes","thumb_path","gx","gy"])

    return {"rows": rows, "events": events, "fps": fps, "W": W, "H": H, "N": N, "stem": stem, "ds": ds, "fast": fast, "thumbs_dir": thumbs_dir}

# --------------------------- Fancy plots ----------------------------------
def fig_to_b64(fig):
    bio = io.BytesIO()
    fig.savefig(bio, format="png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    return base64.b64encode(bio.getvalue()).decode("ascii")

def plot_multiaxis_stance_impact(times, stance_vals, impacts):
    fig, ax1 = plt.subplots(figsize=(10.5,3.4))
    ax1.grid(alpha=0.25)
    ax1.plot(times, stance_vals, color="#b088ff", lw=1.8, label="stance score")
    ax1.axhline(0, color="#445", lw=1, ls="--")
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Stance (+S / -O)", color="#b088ff")
    ax2 = ax1.twinx()
    if impacts[0]:
        ax2.plot(impacts[0], impacts[1], color="#ff8f6b", lw=1.5, alpha=0.9, label="impact")
    ax2.set_ylabel("Impact (0–1)", color="#ff8f6b")
    fig.suptitle("Stance vs Impact (Dual-Axis)", color="#e9f1ff")
    return fig_to_b64(fig)

def plot_3d_time_stance_quality(times, stance_vals, quals):
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(times, stance_vals, quals, color="#66e2d5", lw=1.5)
    ax.set_xlabel("t (s)"); ax.set_ylabel("stance"); ax.set_zlabel("quality")
    ax.set_title("3D Trajectory: time → stance → quality", color="#e9f1ff")
    return fig_to_b64(fig)

def plot_scatter_impact_speed_type(strikes):
    fig = plt.figure(figsize=(7.8,4.2))
    ax = fig.add_subplot(111)
    label_to_c = {"straight": "#ff8f6b", "hook": "#ffd166", "uppercut": "#7bd389", "round": "#b088ff", "teep": "#66e2d5"}
    side_to_m = {"R":"o","L":"s"}
    for e in strikes:
        c = label_to_c.get(e["label"], "#8ab")
        m = side_to_m.get(e["side"], "o")
        ax.scatter(e["max_speed"], e["impact_score"], c=c, marker=m, s=26, alpha=0.9)
    ax.set_xlabel("Max end-effector speed (norm)"); ax.set_ylabel("Impact (0–1)")
    ax.set_title("Impact vs Speed (colored by strike, shaped by side)", color="#e9f1ff")
    ax.grid(alpha=0.25)
    return fig_to_b64(fig)

def plot_hexbin_time_impact(strikes):
    if not strikes:
        fig=plt.figure(); return fig_to_b64(fig)
    t = np.array([e["t"] for e in strikes], float)
    imp = np.array([e["impact_score"] for e in strikes], float)
    fig, ax = plt.subplots(figsize=(8.6,3.6))
    hb = ax.hexbin(t, imp, gridsize=40, cmap=CMAP_IMPACT, mincnt=1)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Impact")
    ax.set_title("Hexbin Density: Time vs Impact", color="#e9f1ff")
    cb = fig.colorbar(hb, ax=ax); cb.set_label("Density")
    return fig_to_b64(fig)

def plot_contact_heatmap(strikes, bins=36):
    pts = [(e.get("gx",0.0), e.get("gy",0.0)) for e in strikes if (e.get("gx",0.0)>0 and e.get("gy",0.0)>0)]
    fig = plt.figure(figsize=(5.8,4.2))
    if pts:
        xy = np.array(pts, float)
        H2, xedges, yedges = np.histogram2d(xy[:,0], xy[:,1], bins=bins, range=[[0,1],[0,1]])
        plt.imshow(H2.T[::-1,:], cmap=CMAP_HEAT, interpolation="nearest", aspect="auto")
        plt.title("Contact Heatmap (normalized image coords)", color="#e9f1ff"); plt.axis("off")
    else:
        plt.title("Contact Heatmap (no coords captured)", color="#e9f1ff"); plt.axis("off")
    return fig_to_b64(fig)

def plot_type_side_matrix(strikes):
    types = ["straight","hook","uppercut","round","teep"]
    sides = ["L","R"]
    M = np.zeros((len(types), len(sides)), dtype=int)
    for e in strikes:
        if e["label"] in types and e["side"] in sides:
            M[types.index(e["label"]), sides.index(e["side"])] += 1
    fig, ax = plt.subplots(figsize=(4.6,3.8))
    im = ax.imshow(M, cmap=CMAP_HEAT)
    ax.set_xticks(range(len(sides))); ax.set_xticklabels(sides)
    ax.set_yticks(range(len(types))); ax.set_yticklabels(types)
    ax.set_title("Type × Side Matrix", color="#e9f1ff")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, str(M[i,j]), ha="center", va="center", color="#fff" if M[i,j]>0 else "#99a")
    return fig_to_b64(fig)

def plot_quality_vs_blur(rows):
    fig, ax = plt.subplots(figsize=(7.2,3.8))
    q = np.array([r["quality"] for r in rows], float)
    b = np.array([r["blur_var"] for r in rows], float)
    ax.scatter(b, q, s=9, c="#66e2d5", alpha=0.6)
    ax.set_xlabel("Laplacian variance (blur proxy)"); ax.set_ylabel("Quality (0–1)")
    ax.set_title("Frame Quality vs Blur", color="#e9f1ff")
    ax.grid(alpha=0.25)
    return fig_to_b64(fig)

def plot_small_hist(arr, title, xl, bins=18):
    fig, ax = plt.subplots(figsize=(5.4,3.2))
    if len(arr): ax.hist(arr, bins=bins, color="#ffd166")
    ax.set_title(title, color="#e9f1ff"); ax.set_xlabel(xl); ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    return fig_to_b64(fig)

def dwell_stats(labels, fps):
    segs=[]
    if not labels: return {"count":0,"avg_s":0,"med_s":0,"max_s":0,"segs":[]}
    cur=labels[0]; length=1
    for lab in labels[1:]:
        if lab==cur: length+=1
        else:
            if cur!="UNKNOWN": segs.append(length)
            cur=lab; length=1
    if cur!="UNKNOWN": segs.append(length)
    if not segs: return {"count":0,"avg_s":0,"med_s":0,"max_s":0,"segs":[]}
    arr=np.array(segs,dtype=float)
    return {"count":len(segs),"avg_s":float(arr.mean()/fps),
            "med_s":float(np.median(arr)/fps),"max_s":float(arr.max()/fps),"segs":segs}

# --------------------------- HTML builder ---------------------------------
def build_html(data):
    rows   = data["rows"]
    events = data["events"]
    fps, W, H, N, stem, ds, fast = data["fps"], data["W"], data["H"], data["N"], data["stem"], data["ds"], data["fast"]
    thumbs_dir = data["thumbs_dir"]
    report_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    times = np.array([r["time_s"] for r in rows], float) if rows else np.array([])
    stance_vals = np.array([r["stance_val"] for r in rows], float) if rows else np.array([])
    quals = np.array([r["quality"] for r in rows], float) if rows else np.array([])
    labels = [r["stance_label"] for r in rows]
    south = sum(1 for s in labels if s=="SOUTHPAW")
    orth  = sum(1 for s in labels if s=="ORTHODOX")
    unk   = sum(1 for s in labels if s=="UNKNOWN")
    total_frames = max(1, len(rows))

    # events split
    strikes = [e for e in events if e["type"]=="strike"]
    feints  = [e for e in events if e["type"]=="feint"]
    impact_times  = [e["t"] for e in strikes]
    impact_values = [e["impact_score"] for e in strikes]
    isi = []
    if len(impact_times) > 1:
        prev = impact_times[0]
        for t in impact_times[1:]:
            isi.append(max(0.0, t-prev)); prev = t

    # figures
    img_combo = plot_multiaxis_stance_impact(times, stance_vals, (impact_times, impact_values))
    img_3d    = plot_3d_time_stance_quality(times, stance_vals, quals)
    img_sc    = plot_scatter_impact_speed_type(strikes)
    img_hex   = plot_hexbin_time_impact(strikes)
    img_heat  = plot_contact_heatmap(strikes, bins=32)
    img_mat   = plot_type_side_matrix(strikes)
    img_qb    = plot_quality_vs_blur(rows)
    img_isi   = plot_small_hist(isi, "Inter-Strike Interval (s)", "seconds", bins=18)

    # Tables
    from html import escape as hesc

    def thumb_cell(e):
        raw = e.get("thumb_path","")
        if not raw:
            return ""
        resolved = resolve_thumb_path(raw, thumbs_dir, report_dir, script_dir)
        if not resolved:
            return ""
        b64 = read_thumb_b64(resolved)
        if not b64:
            return ""
        return f'<img src="data:image/png;base64,{b64}" style="max-width:200px;border:1px solid #223;border-radius:8px" />'

    ev_rows = []
    for e in events:
        ev_rows.append(
            f"<tr>"
            f"<td>{e['t']:.3f}</td><td>{e['frame']}</td>"
            f"<td>{hesc(e['type'])}</td><td>{hesc(e['label'])}</td><td>{e['side']}</td>"
            f"<td>{e['confidence']:.2f}</td><td>{e['impact_score']:.2f}</td>"
            f"<td>{e['duration_s']:.3f}</td><td>{e['max_speed']:.3f}</td>"
            f"<td>{e['q']:.2f}</td><td>{e['blur']:.1f}</td>"
            f"<td>{hesc(e.get('notes',''))}</td><td>{thumb_cell(e)}</td>"
            f"</tr>"
        )

    out_html = f"{stem}_report_fancy.html"
    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Muay Thai Fancy Analysis – {stem}</title>
<style>
:root {{
  --bg:#0e1116; --panel:#111622; --line:#222b3a; --text:#d6e3ff; --muted:#93a5c4;
}}
*{{box-sizing:border-box}}
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:16px;background:var(--bg);color:var(--text)}}
a{{color:#66e2d5;text-decoration:none}} a:hover{{text-decoration:underline}}
.topnav{{display:flex;gap:14px;flex-wrap:wrap;margin:8px 0 16px}}
.topnav a{{padding:6px 10px;border:1px solid var(--line);border-radius:10px;background:#0f1420}}
.card{{border:1px solid var(--line);border-radius:14px;padding:12px;background:var(--panel);box-shadow:0 1px 3px rgba(0,0,0,0.25);display:flex;flex-direction:column}}
.card h3{{margin:6px 0 10px 0}}
img.plot{{width:100%;height:auto;border:1px solid var(--line);border-radius:10px;display:block}}
.kpi{{font-size:22px;font-weight:800}}
.muted{{color:var(--muted)}}
table{{border-collapse:collapse;width:100%;background:#0f1420}}
th,td{{border:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}}
th{{background:#141a29;position:sticky;top:0;cursor:pointer}}
/* Responsive grid with named areas to keep panels aligned relative to each other */
.dashboard{{
  display:grid; gap:14px; align-items:stretch;
  grid-template-columns: 1.3fr 1fr 1fr;
  grid-template-areas:
    "kpi kpi kpi"
    "combo combo traj"
    "scat hex hex"
    "heat mat isi"
    "qb qb qb"
    "events events events";
}}
.sec-kpi{{grid-area:kpi}}
.sec-combo{{grid-area:combo}}
.sec-traj{{grid-area:traj}}
.sec-scat{{grid-area:scat}}
.sec-hex{{grid-area:hex}}
.sec-heat{{grid-area:heat}}
.sec-mat{{grid-area:mat}}
.sec-isi{{grid-area:isi}}
.sec-qb{{grid-area:qb}}
.sec-events{{grid-area:events}}

@media (max-width:1200px){{
  .dashboard{{
    grid-template-columns: 1fr 1fr;
    grid-template-areas:
      "kpi kpi"
      "combo traj"
      "scat hex"
      "heat mat"
      "qb qb"
      "isi isi"
      "events events";
  }}
}}
@media (max-width:760px){{
  .dashboard{{
    grid-template-columns: 1fr;
    grid-template-areas:
      "kpi"
      "combo"
      "traj"
      "scat"
      "hex"
      "heat"
      "mat"
      "qb"
      "isi"
      "events";
  }}
}}

</style>
</head>
<body>
<h1 id="top">Muay Thai Analysis (Fancy)</h1>
<div class="muted">{stem} — {W}×{H} @ {fps:.2f} FPS — frames {N} — downscale {ds:.2f} — fast {fast}</div>

<div class="topnav">
  <a href="#kpi">KPIs</a>
  <a href="#combo">Stance vs Impact</a>
  <a href="#traj">3D</a>
  <a href="#scatter">Impact×Speed</a>
  <a href="#hex">Hexbin</a>
  <a href="#heat">Contact Heatmap</a>
  <a href="#mat">Type×Side</a>
  <a href="#qb">Quality×Blur</a>
  <a href="#isi">Inter-Strike</a>
  <a href="#events">Events</a>
</div>

<div class="dashboard">
  <section class="sec-kpi" id="kpi">
    <div class="card" style="display:grid;grid-template-columns:repeat(3,minmax(180px,1fr));gap:12px">
      <div class="card">
        <div class="kpi" style="color:#7bd389">{100.0*south/total_frames:.1f}%</div>
        <div class="muted">Southpaw time</div>
      </div>
      <div class="card">
        <div class="kpi" style="color:#ffd166">{100.0*orth/total_frames:.1f}%</div>
        <div class="muted">Orthodox time</div>
      </div>
      <div class="card">
        <div class="kpi" style="color:#6c7a99">{100.0*unk/total_frames:.1f}%</div>
        <div class="muted">Unknown time</div>
      </div>
    </div>
  </section>

  <section class="sec-combo" id="combo">
    <div class="card"><h3>Stance vs Impact (Dual-Axis)</h3><img class="plot" src="data:image/png;base64,{img_combo}"></div>
  </section>

  <section class="sec-traj" id="traj">
    <div class="card"><h3>3D Trajectory</h3><img class="plot" src="data:image/png;base64,{img_3d}"></div>
  </section>

  <section class="sec-scat" id="scatter">
    <div class="card"><h3>Impact vs Speed (Type/Side)</h3><img class="plot" src="data:image/png;base64,{img_sc}"></div>
  </section>

  <section class="sec-hex" id="hex">
    <div class="card"><h3>Time × Impact (Hexbin)</h3><img class="plot" src="data:image/png;base64,{img_hex}"></div>
  </section>

  <section class="sec-heat" id="heat">
    <div class="card"><h3>Contact Heatmap</h3><img class="plot" src="data:image/png;base64,{img_heat}"></div>
  </section>

  <section class="sec-mat" id="mat">
    <div class="card"><h3>Type × Side Matrix</h3><img class="plot" src="data:image/png;base64,{img_mat}"></div>
  </section>

  <section class="sec-qb" id="qb">
    <div class="card"><h3>Quality vs Blur</h3><img class="plot" src="data:image/png;base64,{img_qb}"></div>
  </section>

  <section class="sec-isi" id="isi">
    <div class="card"><h3>Inter-Strike Interval</h3><img class="plot" src="data:image/png;base64,{img_isi}"></div>
  </section>

  <section class="sec-events" id="events">
    <div class="card">
      <h3>Events (click column headers to sort)</h3>
      <table id="ev"><thead>
        <tr>
          <th onclick="sortTable(0)">t (s)</th><th onclick="sortTable(1)">frame</th>
          <th>type</th><th>label</th><th>side</th>
          <th onclick="sortTable(5)">conf</th><th onclick="sortTable(6)">impact</th>
          <th onclick="sortTable(7)">dur</th><th onclick="sortTable(8)">max v</th>
          <th onclick="sortTable(9)">q</th><th>blur</th><th>notes</th><th>thumb</th>
        </tr>
      </thead><tbody>
        {''.join(ev_rows)}
      </tbody></table>
    </div>
  </section>
</div>

<script>
function sortTable(n) {{
  var table=document.getElementById("ev"), rows, switching=true, dir="asc", switchcount=0;
  while (switching) {{
    switching=false; rows=table.rows;
    for (var i=1;i<rows.length-1;i++) {{
      var x=rows[i].getElementsByTagName("TD")[n];
      var y=rows[i+1].getElementsByTagName("TD")[n];
      var xv=x.innerText || x.textContent; var yv=y.innerText || y.textContent;
      var a=parseFloat(xv), b=parseFloat(yv), cmp;
      if (!isNaN(a) && !isNaN(b)) cmp = a-b; else cmp = (xv>yv)?1:-1;
      if ((dir=="asc" && cmp>0) || (dir=="desc" && cmp<0)) {{
        rows[i].parentNode.insertBefore(rows[i+1], rows[i]); switching=true; switchcount++; break;
      }}
    }}
    if (!switching && switchcount==0) {{ dir = (dir=="asc"?"desc":"asc"); switching=true; }}
  }}
}}
</script>

</body></html>
"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html

# --------------------------- CLI -----------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--stride", type=int, default=2, help="Process every Nth frame (1=all)")
    ap.add_argument("--downscale", type=float, default=None, help="Resize factor (e.g., 0.85)")
    ap.add_argument("--fast", action="store_true", help="Pose complexity=1 (faster)")
    ap.add_argument("--max-seconds", type=float, default=None, help="Early stop after N seconds")
    ap.add_argument("--force", action="store_true", help="Ignore existing CSVs; recompute")
    args = ap.parse_args()

    data = analyze(args.video,
                   stride=max(1,args.stride),
                   downscale=args.downscale,
                   fast=args.fast,
                   max_seconds=args.max_seconds,
                   force=args.force,
                   save_frames_csv=True,
                   save_events_csv=True)
    out_html = build_html(data)
    print("[OK] Fancy report:", out_html)
    print("[OK] Frames CSV:", f"{data['stem']}_frames.csv")
    print("[OK] Events CSV:", f"{data['stem']}_events.csv")
    print("[OK] Thumbnails dir:", data["thumbs_dir"])

if __name__ == "__main__":
    main()
