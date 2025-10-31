# make_training_report_plus.py
# One-stop HTML report:
# - Reuses stance + quality pipeline from training_pace_adjusted.py
# - Adds Strike Taxonomy v1, Impact Proxy v1, Feint Detector v1
# - Outputs rich HTML with timelines, distributions, heatmaps, and clickable thumbnails
#
# Usage (examples):
#   python make_training_report_plus.py --video ".\pov.mp4"
#   python make_training_report_plus.py --video ".\pov.mp4" --stride 2 --fast --downscale 0.85
#   python make_training_report_plus.py --video ".\pov.mp4" --force  # ignore caches
#
# Notes:
# - No seaborn (pure matplotlib) to avoid extra deps.
# - Thumbnails are extracted around event frames and embedded as base64 images.
# - Single-camera, casual YouTube-like footage friendly.

import os, io, sys, math, csv, time, base64, argparse
from collections import deque, defaultdict, Counter

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

import training_pace_adjusted as tpa  # <-- your stance/quality module


# --------------------------- Safe config import ---------------------------
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

# Hysteresis defaults
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

# Reuse math helpers / logic if present
estimate_forward = cfg("estimate_forward", lambda LHIP,RHIP,LSH,RSH: (np.array([0,0,0],np.float32), np.array([0,0,1],np.float32)))
head_yaw_score   = cfg("head_yaw_score",   lambda face_lms: (0.0, 0.0))
depth_front      = cfg("depth_front",      lambda r,l: (0.0, 0.0))
proj_front_func  = cfg("proj_front",       lambda r,l,o,f: (0.0, 0.0))
all_present      = cfg("all_present",      lambda *vals: all(v is not None for v in vals))
decide_with_hysteresis = cfg("decide_with_hysteresis", None)

# Smoothing
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
            a_d=self._alpha(self.d_cutoff)
            dx_hat=a_d*dx + (1-a_d)*self.dx_prev
            cutoff=self.min_cutoff + self.beta*abs(dx_hat)
            a=self._alpha(cutoff)
            x_hat=x if self.x_prev is None else a*x + (1-a)*self.x_prev
            self.x_prev, self.dx_prev = x_hat, dx_hat
            return x_hat
    class LMFilter:
        def __init__(self, fps):
            self.fx=OneEuro(fps,1.5,0.06); self.fy=OneEuro(fps,1.5,0.06); self.fz=OneEuro(fps,1.2,0.04)
        def apply(self, p):
            if p is None: return None
            return np.array([self.fx.filter(p[0]), self.fy.filter(p[1]), self.fz.filter(p[2]), p[3]], dtype=np.float32)


# --------------------------- Lower-level utilities ------------------------
def laplacian_var(gray): return cv2.Laplacian(gray, cv2.CV_64F).var()
def tanh(x, k=1.0): return float(np.tanh(k*x))
def norm(v, eps=1e-6): 
    n=np.linalg.norm(v); return v/(n+eps)

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

def png_from_frame(frame_bgr, max_w=360):
    h, w = frame_bgr.shape[:2]
    if w > max_w:
        scale = max_w / float(w)
        frame_bgr = cv2.resize(frame_bgr, (max_w, int(h*scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", frame_bgr)
    if not ok: return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


# --------------------------- Strike/Feint classification -------------------
def classify_punch(shoulder, elbow, wrist, lateral_axis):
    if wrist is None or elbow is None or shoulder is None:
        return None, None
    # side: compare wrist to shoulder x (normalized image coords)
    side = "R" if (wrist[0] > shoulder[0]) else "L"
    u_se = joint_vec(shoulder, elbow)
    u_ew = joint_vec(elbow, wrist)
    if u_se is None or u_ew is None: return None, side
    elbow_angle = angle_between(-u_se, u_ew)
    if elbow_angle is None: return None, side
    horiz = abs(np.dot(norm(u_ew), norm(lateral_axis)))
    vertical = abs(np.dot(norm(u_ew), np.array([0,1,0],dtype=np.float32)))
    if elbow_angle > 160:
        label = "straight"  # jab/cross category; side adds the semantics
    elif horiz > 0.7:
        label = "hook"
    elif vertical > 0.6:
        label = "uppercut"
    else:
        label = "straight"
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


# --------------------------- Core analysis pass ---------------------------
def analyze(video_path, stride=1, downscale=None, fast=False, max_seconds=None, force=False,
            save_frames_csv=True, save_events_csv=True):
    stem = os.path.splitext(os.path.basename(video_path))[0]
    frames_csv = f"{stem}_frames.csv"
    events_csv = f"{stem}_events.csv"

    # If caches exist and not forcing, reuse to speed up HTML-only regen later
    reuse_frames = (os.path.exists(frames_csv) and not force)
    reuse_events = (os.path.exists(events_csv) and not force)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise RuntimeError(f"Could not open: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_to_do = N if max_seconds is None else min(N, int(max_seconds*fps))
    ds = downscale if downscale is not None else DOWNSCALE_FACTOR_DEFAULT

    # ---------- If reusing, just read CSVs and return data -----------
    if reuse_frames and reuse_events:
        rows = []
        for r in read_csv(frames_csv):
            # coerce
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
                "notes": e.get("notes","")
            })
        cap.release()
        return {"rows": rows, "events": evs, "fps": fps, "W": W, "H": H, "N": N, "stem": stem, "ds": ds, "fast": fast}

    # ---------- Otherwise, run full analysis in one pass -------------
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

    # Filters
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
        if f is None: filters[key] = LMFilter(fps); f = filters[key]
        return f.apply(p)

    # Rolling stance vote + hysteresis
    votes = deque(maxlen=WIN); wts = deque(maxlen=WIN)
    last_output_stance = "UNKNOWN"; last_change_frame = -10**9

    # Blur quality history
    blur_hist = deque(maxlen=120)

    # Output rows (per-frame) and events
    rows = []
    events = []

    # For thumbnails & spatial heatmaps
    prev_pts = {"RW": None, "LW": None, "RA": None, "LA": None}
    prev_gray = None
    glove_contacts_xy = []  # [(x,y) normalized] at strike contact
    midhip_xy = []          # ringcraft-ish heatmap from mid-hip trajectory

    # Limb finite-state machines
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
            print(f"[report+] {frame_idx}/{total_to_do} frames | elapsed {time.time()-start_t:.1f}s", flush=True)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        lv = float(laplacian_var(gray))
        blur_hist.append(lv)
        med = float(np.median(blur_hist)) if blur_hist else lv
        dyn_thresh = max(BLUR_FLOOR, BLUR_RATIO*(med if med>1e-6 else lv))
        quality = 1.0 / (1.0 + np.exp(-(lv - dyn_thresh)/10.0))

        # Downscale for processing if requested
        if (downscale if downscale is not None else DOWNSCALE_FACTOR_DEFAULT) < 1.0:
            Wd = max(2, int(W*ds)); Hd = max(2, int(H*ds))
            small_rgb = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), (Wd,Hd), interpolation=cv2.INTER_AREA)
            pose_res = pose.process(small_rgb)
        else:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pose_res = pose.process(frame_rgb)

        score_total = 0.0
        weight_total = 0.0
        s_shoulder = 0.0
        s_forearm  = 0.0
        s_head     = 0.0

        LSH=RSH=LE=RE=LW=RW=LH=RH=LA=RA=LK=RK=None
        midhip = None; fwd = np.array([0,0,1],np.float32); lateral=np.array([1,0,0],np.float32)

        if pose_res.pose_landmarks:
            plms = pose_res.pose_landmarks
            LSH = LM(plms, "LEFT_SHOULDER", 0.30);  RSH = LM(plms, "RIGHT_SHOULDER", 0.30)
            LE  = LM(plms, "LEFT_ELBOW",    0.20);  RE  = LM(plms, "RIGHT_ELBOW",    0.20)
            LW  = LM(plms, "LEFT_WRIST",    0.20);  RW  = LM(plms, "RIGHT_WRIST",    0.20)
            LH  = LM(plms, "LEFT_HIP",      0.20);  RH  = LM(plms, "RIGHT_HIP",      0.20)
            LK  = LM(plms, "LEFT_KNEE",     0.10);  RK  = LM(plms, "RIGHT_KNEE",     0.10)
            LA  = LM(plms, "LEFT_ANKLE",    0.10);  RA  = LM(plms, "RIGHT_ANKLE",    0.10)

            # Forward/lateral axes
            if all_present(LSH,RSH,LH,RH):
                pelvis, fwd = estimate_forward(LH,RH,LSH,RSH)
                lateral = norm(LSH[:3]-RSH[:3])
                midhip = 0.5*(LH[:2] + RH[:2])
                midhip_xy.append((float(midhip[0]), float(midhip[1])))

            # quality-adjusted weights
            W_SHOULDER_eff = W_SHOULDER * (0.70 + 0.30*quality)
            W_HANDS_eff    = W_HANDS    * (0.20 + 0.80*quality)
            W_HEAD_eff     = W_HEAD     * (0.40 + 0.60*quality)

            # 1) shoulders
            s_sd, w_sd = depth_front(RSH, LSH)
            s_shoulder = s_sd
            score_total  += W_SHOULDER_eff * s_sd
            weight_total += W_SHOULDER_eff * w_sd

            # 2) forearms (wrists fallback to elbows)
            def limb_pair(primary_R, primary_L, fallback_R, fallback_L):
                R = primary_R if (primary_R is not None and primary_R[3] > 0.35) else fallback_R
                L = primary_L if (primary_L is not None and primary_L[3] > 0.35) else fallback_L
                return R, L
            Rf, Lf = limb_pair(RW,LW,RE,LE)
            s_fd, w_fd = depth_front(Rf, Lf)
            s_fp, w_fp = proj_front_func(Rf, Lf, pelvis if 'pelvis' in locals() else np.array([0,0,0],np.float32), fwd)
            s_fore = (1.0 - FWD_BLEND) * s_fd + FWD_BLEND * s_fp
            s_forearm = s_fore
            w_fore = (w_fd + w_fp)
            score_total  += W_HANDS_eff * s_fore
            weight_total += W_HANDS_eff * w_fore

            # 3) head yaw (we skip FaceMesh for speed in this combined pass)
            # keep s_head=0

        # stance vote + hysteresis
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
            # fallback
            cand = "SOUTHPAW" if stance_val>0 else "ORTHODOX" if stance_val<0 else "UNKNOWN"
            if cand != "UNKNOWN": last_output_stance = cand
        stance_label = last_output_stance

        rows.append({
            "frame": frame_idx,
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

        # ---------------- strike/feint detection (with thumbnails) ---------------
        # compute end-effector speeds
        pts = {"RW": RW, "LW": LW, "RA": RA, "LA": LA}
        speeds = {k: end_effector_speed(prev_pts[k], pts[k], fps, W, H) for k in pts}
        for k in pts: prev_pts[k] = pts[k]
        # local divergence near gloves/shins
        divs = {}
        for limb in ["RW","LW","RA","LA"]:
            P = pts[limb]
            if P is not None:
                cx, cy = int(P[0]*W), int(P[1]*H)
                divs[limb] = local_divergence(prev_gray, gray, cx, cy, win=24)
            else:
                divs[limb] = 0.0

        # thresholds (normalized to typical 30fps)
        TH_ONSET = 0.35
        TH_CONTACT_DECEL = 0.30
        TH_DIV = 0.40
        MAX_FEINT_DUR = int(0.20 * fps)

        # classify & record
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
                    # STRIKE event
                    label, side = None, None
                    if limb in ("RW","LW"):
                        if limb=="RW":
                            label, side = classify_punch(RSH, RE, RW, lateral)
                        else:
                            label, side = classify_punch(LSH, LE, LW, lateral)
                    else:
                        if limb=="RA":
                            label, side = classify_kick(RH, RK, RA, fwd)
                        else:
                            label, side = classify_kick(LH, LK, LA, fwd)
                    side = side or ("R" if limb[0]=="R" else "L")
                    label = label or ("punch" if limb in ("RW","LW") else "kick")
                    contact_signals = (1.0 if speed_drop else 0.0) + (1.0 if dv>TH_DIV else 0.0)
                    confidence = 0.5 + 0.25*contact_signals
                    impact = float(0.6*min(1.0, f.max_speed) + 0.4*min(1.0, dv))

                    # store glove/shin sample for heatmap/scatter
                    P = pts[limb]
                    if P is not None: glove_contacts_xy.append((float(P[0]), float(P[1])))

                    # draw tiny overlay frame & save thumbnail
                    thumb_bgr = frame_bgr.copy()
                    if P is not None:
                        cv2.circle(thumb_bgr, (int(P[0]*W), int(P[1]*H)), 12, (0,255,255), 2)
                    cv2.putText(thumb_bgr, f"{label} {side}  imp={impact:.2f}",
                                (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                    thumb_png = png_from_frame(thumb_bgr, max_w=380)

                    events.append({
                        "t": round(t_s,3),
                        "frame": frame_idx,
                        "type": "strike",
                        "side": side,
                        "label": label,
                        "confidence": round(confidence,2),
                        "impact_score": round(impact,2),
                        "duration_s": round((frame_idx - (f.t0 or frame_idx))/fps,3),
                        "max_speed": round(f.max_speed,3),
                        "q": round(quality,2),
                        "blur": round(lv,1),
                        "notes": f"{limb}: drop={speed_drop}, div={dv:.2f}",
                        "thumb": thumb_png
                    })
                    f.state=LimbFSM.RETRACT
                else:
                    if f.ext_frames>2 and s < TH_ONSET*0.6:
                        dur = frame_idx - (f.t0 or frame_idx)
                        if dur <= MAX_FEINT_DUR:
                            thumb_png = png_from_frame(frame_bgr, max_w=380)
                            events.append({
                                "t": round(t_s,3), "frame": frame_idx,
                                "type": "feint", "side": ("R" if limb[0]=="R" else "L"),
                                "label": ("hand" if limb in ("RW","LW") else "leg"),
                                "confidence": 0.6, "impact_score": 0.0,
                                "duration_s": round(dur/fps,3),
                                "max_speed": round(f.max_speed,3),
                                "q": round(quality,2), "blur": round(lv,1),
                                "notes": f"{limb}: abort",
                                "thumb": thumb_png
                            })
                        f.state=LimbFSM.RETRACT
            elif f.state == LimbFSM.CONTACT:
                f.state = LimbFSM.RETRACT
            elif f.state == LimbFSM.RETRACT:
                if s < TH_ONSET*0.4:
                    f.state=LimbFSM.IDLE; f.max_speed=0.0; f.ext_frames=0

        prev_gray = gray

    cap.release()

    # Save CSVs if requested
    if save_frames_csv and rows:
        write_csv(rows, frames_csv, ["frame","time_s","stance_val","stance_label","quality","blur_var","shoulder_s","forearm_s","head_s","weight_total"])
    if save_events_csv and events:
        write_csv(events, events_csv, ["t","frame","type","side","label","confidence","impact_score","duration_s","max_speed","q","blur","notes","thumb"])

    return {"rows": rows, "events": events, "fps": fps, "W": W, "H": H, "N": N, "stem": stem, "ds": ds, "fast": fast}


# --------------------------- Visualization builders -----------------------
def plot_line(x, y, title, yl, hline0=False):
    fig = plt.figure(figsize=(9,3))
    if len(x): plt.plot(x, y, linewidth=1.8)
    if hline0: plt.axhline(0, linestyle="--", linewidth=1)
    plt.title(title); plt.xlabel("Time (s)"); plt.ylabel(yl)
    return b64_png(fig)

def plot_hist(data, title, xl, bins=20):
    fig = plt.figure(figsize=(6,3.5))
    if len(data): plt.hist(data, bins=bins)
    plt.title(title); plt.xlabel(xl); plt.ylabel("Count")
    return b64_png(fig)

def plot_heatmap_xy(points, title, W=640, H=360, bins=36):
    # points are normalized (x,y) in [0,1]
    if not points: 
        fig = plt.figure(figsize=(5,3)); plt.title(title); return b64_png(fig)
    xy = np.array(points, dtype=float)
    H2, xedges, yedges = np.histogram2d(xy[:,0], xy[:,1], bins=bins, range=[[0,1],[0,1]])
    # display with origin at top-left (image coordinates)
    fig = plt.figure(figsize=(5.5,4))
    plt.imshow(H2.T[::-1,:], aspect='auto', interpolation="nearest")
    plt.title(title); plt.axis("off")
    return b64_png(fig)

def plot_bar_from_counter(counter, title, xl="category"):
    labels, values = zip(*sorted(counter.items())) if counter else ([],[])
    fig = plt.figure(figsize=(7,3.2))
    if labels:
        xs = np.arange(len(labels))
        plt.bar(xs, values)
        plt.xticks(xs, labels, rotation=25, ha="right")
    plt.title(title); plt.ylabel("Count"); plt.xlabel(xl)
    return b64_png(fig)

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


# --------------------------- HTML report ----------------------------------
def build_html(data):
    rows   = data["rows"]
    events = data["events"]
    fps, W, H, N, stem, ds, fast = data["fps"], data["W"], data["H"], data["N"], data["stem"], data["ds"], data["fast"]

    times = np.array([r["time_s"] for r in rows], float) if rows else np.array([])
    stance_vals = np.array([r["stance_val"] for r in rows], float) if rows else np.array([])
    quals = np.array([r["quality"] for r in rows], float) if rows else np.array([])
    st_labels = [r["stance_label"] for r in rows]
    south = sum(1 for s in st_labels if s=="SOUTHPAW")
    orth  = sum(1 for s in st_labels if s=="ORTHODOX")
    unk   = sum(1 for s in st_labels if s=="UNKNOWN")
    total_frames = len(rows)

    # stance dwell stats
    dwell = dwell_stats(st_labels, fps)

    # event series and counters
    strikes = [e for e in events if e["type"]=="strike"]
    feints  = [e for e in events if e["type"]=="feint"]
    impact_times  = [e["t"] for e in strikes]
    impact_values = [e["impact_score"] for e in strikes]
    strike_labels = Counter([f'{e["label"]}:{e["side"]}' for e in strikes])
    strike_types  = Counter([e["label"] for e in strikes])
    side_counts   = Counter([e["side"] for e in strikes])
    feint_side    = Counter([e["side"] for e in feints])

    # inter-strike interval distribution
    isi = []
    if len(impact_times) > 1:
        prev = impact_times[0]
        for t in impact_times[1:]:
            isi.append(t - prev); prev = t

    # spatial heatmaps
    # glove contact XY heatmap already collected as normalized positions in events with "thumb" (during generation),
    # but for cache-reuse mode we don’t have them; we skip those heatmaps if absent.
    contact_xy = [(0.5,0.5)]  # default dummy
    if strikes and ("thumb" in strikes[0]):  # heuristic: thumbs exist => we generated in this run
        # we didn't persist glove XY in CSV; so reconstruct approximate scatter from max_speed note? Not available -> skip
        pass
    # we *can* approximate with times distribution only; instead show impact histogram & per-class bars

    # Build plots
    img_stance = plot_line(times, stance_vals, "Stance Score Over Time (+ = Southpaw, - = Orthodox)", "Score", hline0=True)
    img_quality = plot_line(times, quals, "Frame Quality Over Time", "Quality (0–1)")
    img_impacts = plot_line(impact_times, impact_values, "Impact Score Over Time", "Impact (0–1)")
    img_impact_hist = plot_hist(impact_values, "Impact Score Distribution", "Impact (0–1)", bins=16)
    img_isi_hist = plot_hist(isi, "Inter-Strike Interval (s)", "seconds", bins=18)
    img_types_bar = plot_bar_from_counter(strike_types, "Strike Types (total)")
    img_side_bar  = plot_bar_from_counter(side_counts,   "Strikes by Side")
    img_feint_bar = plot_bar_from_counter(feint_side,    "Feints by Side")

    # stance percentages
    def pct(n): return (100.0*n/max(1,total_frames))

    # Events table with thumbnails (if available)
    def html_escape(s):
        return (str(s).replace("&","&amp;").replace("<","&lt;")
                .replace(">","&gt;").replace('"',"&quot;"))

    ev_rows = []
    for e in events:
        thumb_html = ""
        if "thumb" in e and e["thumb"]:
            thumb_html = f'<img src="data:image/png;base64,{e["thumb"]}" style="max-width:220px;border:1px solid #ddd;border-radius:6px" />'
        ev_rows.append(
            f"<tr>"
            f"<td>{e['t']:.3f}</td><td>{e['frame']}</td>"
            f"<td>{html_escape(e['type'])}</td><td>{html_escape(e['label'])}</td><td>{e['side']}</td>"
            f"<td>{e['confidence']:.2f}</td><td>{e['impact_score']:.2f}</td>"
            f"<td>{e['duration_s']:.3f}</td><td>{e['max_speed']:.3f}</td>"
            f"<td>{e['q']:.2f}</td><td>{e['blur']:.1f}</td>"
            f"<td>{html_escape(e.get('notes',''))}</td>"
            f"<td>{thumb_html}</td>"
            f"</tr>"
        )

    out_html = f"{stem}_report_plus.html"
    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Muay Thai Analysis – {stem}</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:24px}}
.grid2{{display:grid;grid-template-columns:repeat(2,minmax(320px,1fr));gap:18px}}
.grid3{{display:grid;grid-template-columns:repeat(3,minmax(260px,1fr));gap:18px}}
.card{{border:1px solid #eee;border-radius:12px;padding:14px;box-shadow:0 1px 2px rgba(0,0,0,0.04)}}
.muted{{color:#666}} img.plot{{max-width:100%;height:auto;border:1px solid #eee;border-radius:8px}}
table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #eee;padding:6px 8px;text-align:left;vertical-align:top}}
th{{background:#fafafa;position:sticky;top:0;cursor:pointer}}
.badge{{display:inline-block;padding:2px 8px;border-radius:999px;background:#f0f0f0;margin-right:6px}}
.kpi{{font-size:22px;font-weight:700}}
</style>
</head>
<body>
<h1>Muay Thai Analysis</h1>
<div class="muted">{stem} — {W}×{H} @ {fps:.2f} FPS — frames {N} — downscale {ds:.2f} — fast {fast}</div>

<div class="grid3" style="margin-top:18px">
  <div class="card">
    <div class="kpi">{pct(south):.1f}%</div>
    <div class="muted">Southpaw time</div>
  </div>
  <div class="card">
    <div class="kpi">{pct(orth):.1f}%</div>
    <div class="muted">Orthodox time</div>
  </div>
  <div class="card">
    <div class="kpi">{pct(unk):.1f}%</div>
    <div class="muted">Unknown time</div>
  </div>
</div>

<div class="grid2" style="margin-top:18px">
  <div class="card"><h3>Stance Score</h3><img class="plot" src="data:image/png;base64,{img_stance}"></div>
  <div class="card"><h3>Frame Quality</h3><img class="plot" src="data:image/png;base64,{img_quality}"></div>
</div>

<div class="grid2" style="margin-top:18px">
  <div class="card"><h3>Impact Over Time</h3><img class="plot" src="data:image/png;base64,{img_impacts}"></div>
  <div class="card"><h3>Impact Distribution</h3><img class="plot" src="data:image/png;base64,{img_impact_hist}"></div>
</div>

<div class="grid2" style="margin-top:18px">
  <div class="card"><h3>Inter-Strike Interval</h3><img class="plot" src="data:image/png;base64,{img_isi_hist}"></div>
  <div class="card"><h3>Strike Types</h3><img class="plot" src="data:image/png;base64,{img_types_bar}"></div>
</div>

<div class="grid2" style="margin-top:18px">
  <div class="card"><h3>Strikes by Side</h3><img class="plot" src="data:image/png;base64,{img_side_bar}"></div>
  <div class="card"><h3>Feints by Side</h3><img class="plot" src="data:image/png;base64,{img_feint_bar}"></div>
</div>

<div class="card" style="margin-top:18px">
  <h3>Dwell (stance stability)</h3>
  <div class="muted">Segments (non-UNKNOWN): count={dwell['count']}, avg={dwell['avg_s']:.2f}s, med={dwell['med_s']:.2f}s, max={dwell['max_s']:.2f}s</div>
</div>

<div class="card" style="margin-top:18px">
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
    out_path = f"{stem}_report_plus.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


# --------------------------- CLI -----------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--stride", type=int, default=2, help="Process every Nth frame (1=all)")
    ap.add_argument("--downscale", type=float, default=None, help="Resize factor for inference (e.g., 0.85)")
    ap.add_argument("--fast", action="store_true", help="Pose complexity=1 (faster)")
    ap.add_argument("--max-seconds", type=float, default=None, help="Early stop after N seconds")
    ap.add_argument("--force", action="store_true", help="Ignore existing CSV caches and recompute")
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
    print("[OK] Report:", out_html)
    print("[OK] Frames CSV:", f"{data['stem']}_frames.csv")
    print("[OK] Events CSV:", f"{data['stem']}_events.csv")

if __name__ == "__main__":
    main()
