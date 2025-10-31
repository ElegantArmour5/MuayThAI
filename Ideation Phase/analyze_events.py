# analyze_events.py
# Add Strike Taxonomy v1 + Impact Proxy v1 + Feint Detector v1
# Builds on training_pace_adjusted.py (imports smoothing, weighting, hysteresis, etc.)
#
# Usage examples:
#   python analyze_events.py --video ".\pov.mp4" --show
#   python analyze_events.py --video ".\pov.mp4" --stride 2 --fast --downscale 0.85
#   python analyze_events.py --video ".\spar.mp4" --max-seconds 30 --save-csv

import os, io, sys, math, csv, time, base64, argparse
from collections import deque, defaultdict

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

import training_pace_adjusted as tpa  # your previous file

# ------------------------- Config & fallbacks -----------------------------
def cfg(name, default):
    return getattr(tpa, name, default)

# Reuse stance/quality weighting knobs where sensible
W_SHOULDER = cfg("W_SHOULDER", 1.5)
W_HANDS    = cfg("W_HANDS",    0.6)
W_HEAD     = cfg("W_HEAD",     0.45)
FWD_BLEND  = cfg("FWD_BLEND",  0.50)
BLUR_FLOOR = cfg("BLUR_FLOOR", 4.0)
BLUR_RATIO = cfg("BLUR_RATIO", 0.08)
DOWNSCALE_FACTOR_DEFAULT = cfg("DOWNSCALE_FACTOR", 0.85)
FULL2KEY = cfg("FULL2KEY", {
    "LEFT_SHOULDER": "LS", "RIGHT_SHOULDER": "RS",
    "LEFT_ELBOW": "LE", "RIGHT_ELBOW": "RE",
    "LEFT_WRIST": "LW", "RIGHT_WRIST": "RW",
    "LEFT_HIP": "LH", "RIGHT_HIP": "RH",
    "LEFT_ANKLE": "LA", "RIGHT_ANKLE": "RA",
    "LEFT_KNEE": "LK", "RIGHT_KNEE": "RK",
})

# Pose helpers (reuse from tpa if present)
estimate_forward = cfg("estimate_forward", lambda LHIP,RHIP,LSH,RSH: (np.array([0,0,0],np.float32), np.array([0,0,1],np.float32)))
all_present      = cfg("all_present",     lambda *vals: all(v is not None for v in vals))

# Smoothing (use tpa’s OneEuro/LMFilter if available)
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

# ------------------------- Utility / plotting ----------------------------
def laplacian_var(gray): return cv2.Laplacian(gray, cv2.CV_64F).var()
def tanh(x, k=1.0): return float(np.tanh(k*x))
def norm(v, eps=1e-6):
    n=np.linalg.norm(v); return v/(n+eps)

def b64_png(fig):
    import matplotlib
    bio = io.BytesIO()
    fig.savefig(bio, format="png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    return base64.b64encode(bio.getvalue()).decode("ascii")

def write_csv(rows, out_csv):
    keys = ["t","frame","type","side","label","confidence","impact_score","duration_s","max_speed",
            "q","blur","notes"]
    with open(out_csv,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k:r.get(k,"") for k in keys})

# ------------------------- Kinematics helpers ----------------------------
def joint_vec(a, b):
    if a is None or b is None: return None
    return b[:3] - a[:3]

def angle_between(u, v):
    if u is None or v is None: return None
    du, dv = norm(u), norm(v)
    return math.degrees(math.acos(float(np.clip(np.dot(du, dv), -1.0, 1.0))))

def end_effector_speed(prev_pt, curr_pt, fps, imgW, imgH):
    # speed in "image fractions per second" normalized by frame size
    if prev_pt is None or curr_pt is None: return None
    dx = (curr_pt[0]-prev_pt[0])*imgW; dy=(curr_pt[1]-prev_pt[1])*imgH
    pix_per_s = math.hypot(dx, dy)*fps
    # convert to unitless by dividing by diagonal
    diag = math.hypot(imgW, imgH)
    return (pix_per_s / max(1e-6, diag))

def proj_front(r, l, origin, fwd):
    if r is None or l is None: return 0.0, 0.0
    rp=float(np.dot(r[:3]-origin, fwd)); lp=float(np.dot(l[:3]-origin, fwd))
    return tanh(rp-lp, 6.0), 1.0

# ------------------------- Strike FSM & thresholds ------------------------
class LimbFSM:
    """Simple finite state machine per limb (R/L wrist & ankle) to segment onsets/contacts."""
    IDLE, LOAD, EXTEND, CONTACT, RETRACT = range(5)
    def __init__(self, name):
        self.name=name
        self.state=self.IDLE
        self.t0=self.t_last=None
        self.max_speed=0.0
        self.speed_hist=deque(maxlen=10)
        self.ext_frames=0

def classify_punch(shoulder, elbow, wrist, lateral_axis):
    # Coarse rules using elbow flexion and hand path relative to shoulder line.
    # Returns: "jab"/"cross"/"hook"/"uppercut" + side ("R"/"L")
    if wrist is None or elbow is None or shoulder is None:
        return None, None
    # side
    side = "R" if (shoulder is not None and shoulder[0] > 0.5) else "L"  # fallback if needed
    # vectors
    u_se = joint_vec(shoulder, elbow)
    u_ew = joint_vec(elbow, wrist)
    if u_se is None or u_ew is None: return None, None
    elbow_angle = angle_between(-u_se, u_ew)  # elbow extension ~180 when straight
    horiz = abs(np.dot(norm(u_ew), norm(lateral_axis)))  # hook-ish if horizontal
    vertical = abs(np.dot(norm(u_ew), np.array([0,1,0],dtype=np.float32)))  # upper-ish if vertical
    # heuristics
    if elbow_angle is None: return None, None
    if elbow_angle > 160:
        label = "cross" if side=="R" else "jab"
    elif horiz > 0.7:
        label = "hook"
    elif vertical > 0.6:
        label = "uppercut"
    else:
        label = "jab" if side=="L" else "cross"
    return label, side

def classify_kick(hip, knee, ankle, fwd_axis):
    if ankle is None or knee is None or hip is None:
        return None, None
    # side: ankle x vs midline
    side = "R" if ankle[0] > 0.5 else "L"
    u_hk = joint_vec(hip, knee)
    u_ka = joint_vec(knee, ankle)
    if u_hk is None or u_ka is None: return None, None
    knee_angle = angle_between(-u_hk, u_ka)
    # teep vs round: projection along forward vs lateral plane
    forward_comp = abs(np.dot(norm(u_ka), norm(fwd_axis)))
    label = "teep" if forward_comp > 0.6 else "round"
    return label, side

# ------------------------- Optical Flow divergence ------------------------
def local_divergence(prev_gray, gray, cx, cy, win=24):
    if prev_gray is None or gray is None: return 0.0
    h,w = gray.shape[:2]
    x0=max(0,int(cx-win)); x1=min(w,int(cx+win))
    y0=max(0,int(cy-win)); y1=min(h,int(cy+win))
    if x1-x0<4 or y1-y0<4: return 0.0
    flow = cv2.calcOpticalFlowFarneback(prev_gray[y0:y1,x0:x1], gray[y0:y1,x0:x1], None,
                                        0.5, 2, 13, 2, 5, 1.2, 0)
    fx, fy = flow[...,0], flow[...,1]
    # poor-man's divergence: ∂u/∂x + ∂v/∂y
    du_dx = cv2.Sobel(fx, cv2.CV_32F, 1, 0, ksize=3)
    dv_dy = cv2.Sobel(fy, cv2.CV_32F, 0, 1, ksize=3)
    div = du_dx + dv_dy
    return float(np.mean(np.abs(div)))

# ------------------------- Main analysis loop -----------------------------
def analyze(video_path, stride=1, downscale=None, fast=False, max_seconds=None,
            show=False, save_csv=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_to_do = N if max_seconds is None else min(N, int(max_seconds*fps))
    ds = downscale if downscale is not None else DOWNSCALE_FACTOR_DEFAULT

    # Pose (complexity: 1 if fast, else 2), Face not needed for v1 modules
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=(1 if fast else 2),
                        smooth_landmarks=True,
                        enable_segmentation=False,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.5)
    PLM = mp_pose.PoseLandmark

    # Filters for key landmarks
    keys = set(FULL2KEY.values()) | set(["LK","RK"])
    filters = {k: LMFilter(fps) for k in keys}

    def get_lm(plms, name, vis_thresh):
        idx = PLM[name].value
        pt = plms.landmark[idx]
        if pt.visibility < vis_thresh: return None
        return np.array([pt.x, pt.y, pt.z, pt.visibility], dtype=np.float32)

    def LM(plms, name, vis=0.20):
        p = get_lm(plms, name, vis)
        if p is None: return None
        key = FULL2KEY.get(name, name[:2])
        f = filters.get(key, None)
        if f is None:
            filters[key] = LMFilter(fps); f = filters[key]
        return f.apply(p)

    # FSMs for limbs
    fsm = {
        "RW": LimbFSM("RW"), "LW": LimbFSM("LW"),
        "RA": LimbFSM("RA"), "LA": LimbFSM("LA"),
    }

    prev_pts = {"RW": None, "LW": None, "RA": None, "LA": None}
    prev_gray = None

    events = []  # rows for CSV/HTML
    blur_hist = deque(maxlen=120)

    window_name = "Events (preview)"
    if show: cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

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

        # progress every ~1s
        if i % max(1, int(fps/stride)) == 0:
            print(f"[events] {frame_idx}/{total_to_do} ({100.0*frame_idx/total_to_do:.1f}%)  elapsed {time.time()-start_t:.1f}s", flush=True)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        lv = float(laplacian_var(gray))
        blur_hist.append(lv)
        med = float(np.median(blur_hist)) if blur_hist else lv
        dyn_thresh = max(BLUR_FLOOR, BLUR_RATIO * (med if med > 1e-6 else lv))
        quality = 1.0 / (1.0 + np.exp(-(lv - dyn_thresh) / 10.0))

        # Downscale for processing if requested
        if ds < 1.0:
            Wd = max(2, int(W*ds)); Hd = max(2, int(H*ds))
            small_rgb = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                                   (Wd,Hd), interpolation=cv2.INTER_AREA)
            pose_res = pose.process(small_rgb)
        else:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pose_res = pose.process(frame_rgb)

        annotated = frame_bgr.copy()

        if pose_res.pose_landmarks:
            plms = pose_res.pose_landmarks
            # Key points
            LSH = LM(plms, "LEFT_SHOULDER", 0.30);  RSH = LM(plms, "RIGHT_SHOULDER", 0.30)
            LE  = LM(plms, "LEFT_ELBOW",    0.20);  RE  = LM(plms, "RIGHT_ELBOW",    0.20)
            LW  = LM(plms, "LEFT_WRIST",    0.20);  RW  = LM(plms, "RIGHT_WRIST",    0.20)
            LH  = LM(plms, "LEFT_HIP",      0.20);  RH  = LM(plms, "RIGHT_HIP",      0.20)
            LA  = LM(plms, "LEFT_ANKLE",    0.10);  RA  = LM(plms, "RIGHT_ANKLE",    0.10)
            LK  = LM(plms, "LEFT_KNEE",     0.10);  RK  = LM(plms, "RIGHT_KNEE",     0.10)

            # Forward axis for kick classification
            if all_present(LSH, RSH, LH, RH):
                pelvis, fwd = estimate_forward(LH, RH, LSH, RSH)
            else:
                pelvis, fwd = np.array([0,0,0],np.float32), np.array([0,0,1],np.float32)
            lateral = norm(LSH[:3]-RSH[:3]) if all_present(LSH,RSH) else np.array([1,0,0],np.float32)

            # Compute speeds for end-effectors
            pts = {"RW": RW, "LW": LW, "RA": RA, "LA": LA}
            speeds = {}
            for limb in ["RW","LW","RA","LA"]:
                speeds[limb] = end_effector_speed(prev_pts[limb], pts[limb], fps, W, H)
                prev_pts[limb] = pts[limb]

            # Optical flow divergence around gloves/shins
            divs = {}
            for limb in ["RW","LW","RA","LA"]:
                P = pts[limb]
                if P is not None:
                    cx, cy = int(P[0]*W), int(P[1]*H)
                    divs[limb] = local_divergence(prev_gray, gray, cx, cy, win=24)
                else:
                    divs[limb] = 0.0

            # Update FSM per limb and detect events
            # thresholds tuned for 30fps 1080p-ish, normalized speed units (diag-normalized)
            TH_ONSET = 0.35   # start moving
            TH_CONTACT_DECEL = 0.30  # drop from peak speed
            TH_DIV = 0.40     # flow divergence near impact
            MAX_FEINT_DUR = int(0.20 * fps)  # <200ms from onset to abort

            for limb in ["RW","LW","RA","LA"]:
                s = speeds[limb]
                dv = divs[limb]
                f = fsm[limb]

                if s is None:
                    continue

                f.speed_hist.append(s)
                f.max_speed = max(f.max_speed, s)

                if f.state == LimbFSM.IDLE:
                    if s > TH_ONSET:
                        f.state = LimbFSM.LOAD
                        f.t0 = frame_idx
                        f.ext_frames = 0
                elif f.state == LimbFSM.LOAD:
                    # ramping up
                    if s > f.max_speed * 0.9:
                        f.state = LimbFSM.EXTEND
                    elif s < TH_ONSET * 0.5:
                        # aborted extremely early: feint micro-onset
                        f.state = LimbFSM.IDLE
                        f.max_speed = 0.0
                elif f.state == LimbFSM.EXTEND:
                    f.ext_frames += 1
                    # Contact condition: speed drop AND divergence spike
                    speed_drop = (f.max_speed - s) > TH_CONTACT_DECEL
                    contact_like = speed_drop or dv > TH_DIV
                    if contact_like:
                        f.state = LimbFSM.CONTACT
                        # Record STRIKE event
                        label, side = None, None
                        confidence = 0.5
                        if limb in ("RW","LW"):
                            # Punch classification
                            if limb=="RW":
                                label, side = classify_punch(RSH, RE, RW, lateral)
                            else:
                                label, side = classify_punch(LSH, LE, LW, lateral)
                        else:
                            # Kick classification
                            if limb=="RA":
                                label, side = classify_kick(RH, RK, RA, fwd)
                            else:
                                label, side = classify_kick(LH, LK, LA, fwd)

                        # Basic confidence: high if both speed_drop and divergence present
                        contact_signals = (1.0 if speed_drop else 0.0) + (1.0 if dv>TH_DIV else 0.0)
                        confidence = 0.5 + 0.25*contact_signals
                        impact_score = float(0.6*min(1.0, f.max_speed) + 0.4*min(1.0, dv))

                        events.append({
                            "t": round(t_s,3),
                            "frame": frame_idx,
                            "type": "strike",
                            "side": side or ("R" if limb[0]=="R" else "L"),
                            "label": (label or ("punch" if limb in ("RW","LW") else "kick")),
                            "confidence": round(confidence,2),
                            "impact_score": round(impact_score,2),
                            "duration_s": round((frame_idx - (f.t0 or frame_idx))/fps,3),
                            "max_speed": round(f.max_speed,3),
                            "q": round(quality,2),
                            "blur": round(lv,1),
                            "notes": f"{limb}: drop={speed_drop}, div={dv:.2f}"
                        })
                        f.state = LimbFSM.RETRACT
                    else:
                        # Possible FEINT: short extension then retract without contact cues
                        if f.ext_frames > 2 and s < TH_ONSET*0.6:
                            dur = frame_idx - (f.t0 or frame_idx)
                            if dur <= MAX_FEINT_DUR:
                                events.append({
                                    "t": round(t_s,3),
                                    "frame": frame_idx,
                                    "type": "feint",
                                    "side": ("R" if limb[0]=="R" else "L"),
                                    "label": ("hand" if limb in ("RW","LW") else "leg"),
                                    "confidence": 0.6,
                                    "impact_score": 0.0,
                                    "duration_s": round(dur/fps,3),
                                    "max_speed": round(f.max_speed,3),
                                    "q": round(quality,2),
                                    "blur": round(lv,1),
                                    "notes": f"{limb}: abort"
                                })
                            f.state = LimbFSM.RETRACT
                elif f.state == LimbFSM.CONTACT:
                    # brief state, move to retract quickly
                    f.state = LimbFSM.RETRACT
                elif f.state == LimbFSM.RETRACT:
                    if s < TH_ONSET*0.4:
                        f.state = LimbFSM.IDLE
                        f.max_speed = 0.0
                        f.ext_frames = 0

            # simple overlay for preview
            if show:
                for P in [LSH,RSH,LE,RE,LW,RW,LH,RH,LK,RK,LA,RA]:
                    if P is None: continue
                    cv2.circle(annotated, (int(P[0]*W), int(P[1]*H)), 4, (255,255,0), -1)

        # show
        if show:
            cv2.putText(annotated, f"q={quality:.2f} blur={lv:.1f}", (10,28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,255,50), 2)
            cv2.imshow(window_name, annotated)
            if (cv2.waitKey(1) & 0xFF) in (27, ord('q')): break

        prev_gray = gray

    cap.release()
    if show: cv2.destroyAllWindows()

    # -------------------------------- Reports --------------------------------
    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_csv = f"{stem}_events.csv"
    if events and save_csv:
        write_csv(events, out_csv)

    # HTML mini-report
    # Top counts
    count_by = defaultdict(int)
    for e in events:
        key = f"{e['type']}:{e['label']}"
        count_by[key] += 1

    # Impact distribution for strikes
    impacts = [e["impact_score"] for e in events if e["type"]=="strike"]
    times = [e["t"] for e in events if e["type"]=="strike"]
    fig1 = plt.figure(figsize=(9,3))
    if times and impacts:
        plt.plot(times, impacts, linewidth=1.5)
    plt.title("Impact Score Over Time"); plt.xlabel("Time (s)"); plt.ylabel("Impact (0–1)")
    img1 = b64_png(fig1)

    # Feints over time
    f_times = [e["t"] for e in events if e["type"]=="feint"]
    fig2 = plt.figure(figsize=(9,3))
    if f_times:
        y = np.ones(len(f_times))
        plt.stem(f_times, y, use_line_collection=True)
    plt.title("Feints Over Time"); plt.xlabel("Time (s)"); plt.ylabel("Feint")
    img2 = b64_png(fig2)

    # Build small sortable table
    def html_escape(s):
        return (str(s).replace("&","&amp;").replace("<","&lt;")
                .replace(">","&gt;").replace('"',"&quot;"))

    rows_html = []
    for e in events:
        rows_html.append(
            f"<tr>"
            f"<td>{e['t']:.3f}</td><td>{e['frame']}</td>"
            f"<td>{html_escape(e['type'])}</td><td>{html_escape(e['label'])}</td><td>{e['side']}</td>"
            f"<td>{e['confidence']:.2f}</td><td>{e['impact_score']:.2f}</td>"
            f"<td>{e['duration_s']:.3f}</td><td>{e['max_speed']:.3f}</td>"
            f"<td>{e['q']:.2f}</td><td>{e['blur']:.1f}</td><td>{html_escape(e['notes'])}</td>"
            f"</tr>"
        )

    out_html = f"{stem}_events.html"
    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Combat Events – {stem}</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:24px}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(280px,1fr));gap:18px}}
.card{{border:1px solid #eee;border-radius:10px;padding:14px}}
table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #eee;padding:6px 8px;text-align:left}}
th{{cursor:pointer}} .muted{{color:#666}}
</style>
</head><body>
<h1>Combat Events</h1>
<div class="muted">{stem} — stride {stride} — downscale {ds:.2f} — fast {fast}</div>

<div class="grid" style="margin-top:18px">
  <div class="card">
    <h3>Impact Score</h3>
    <img src="data:image/png;base64,{img1}">
  </div>
  <div class="card">
    <h3>Feints</h3>
    <img src="data:image/png;base64,{img2}">
  </div>
</div>

<div class="card" style="margin-top:18px">
  <h3>Counts</h3>
  <ul>
    {''.join(f"<li>{html_escape(k)}: <b>{v}</b></li>" for k,v in sorted(count_by.items()))}
  </ul>
</div>

<div class="card" style="margin-top:18px">
  <h3>Events</h3>
  <table id="ev"><thead>
    <tr><th onclick="sortTable(0)">t (s)</th><th onclick="sortTable(1)">frame</th>
        <th>type</th><th>label</th><th>side</th>
        <th onclick="sortTable(5)">conf</th><th onclick="sortTable(6)">impact</th>
        <th onclick="sortTable(7)">dur</th><th onclick="sortTable(8)">max v</th>
        <th onclick="sortTable(9)">q</th><th>blur</th><th>notes</th></tr></thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
</div>
<script>
function sortTable(n) {{
  var table=document.getElementById("ev"), rows, switching=true, dir="asc", switchcount=0;
  while (switching) {{
    switching=false; rows=table.rows;
    for (var i=1;i<rows.length-1;i++) {{
      var x=rows[i].getElementsByTagName("TD")[n];
      var y=rows[i+1].getElementsByTagName("TD")[n];
      var cmp = parseFloat(x.innerHTML)-parseFloat(y.innerHTML);
      if (isNaN(cmp)) cmp = (x.innerHTML > y.innerHTML) ? 1 : -1;
      if ((dir=="asc" && cmp>0) || (dir=="desc" && cmp<0)) {{
        rows[i].parentNode.insertBefore(rows[i+1], rows[i]);
        switching=true; switchcount++; break;
      }}
    }}
    if (!switching && switchcount==0) {{ dir = (dir=="asc"?"desc":"asc"); switching=true; }}
  }}
}}
</script>
</body></html>"""
    with open(out_html,"w",encoding="utf-8") as f:
        f.write(html)

    print(f"[OK] Events CSV: {out_csv}" if events and save_csv else "[INFO] CSV skipped")
    print(f"[OK] HTML: {out_html}")
    return {"csv": (out_csv if events and save_csv else None), "html": out_html, "events": events}

# --------------------------------- CLI -----------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--stride", type=int, default=2, help="Process every Nth frame (1=all)")
    ap.add_argument("--downscale", type=float, default=None, help="Resize factor for inference (e.g., 0.85)")
    ap.add_argument("--fast", action="store_true", help="Pose complexity=1 (faster), no face mesh")
    ap.add_argument("--max-seconds", type=float, default=None, help="Early stop after N seconds")
    ap.add_argument("--show", action="store_true", help="Live preview with keypoints")
    ap.add_argument("--save-csv", action="store_true", help="Write *_events.csv")
    args = ap.parse_args()

    analyze(args.video, stride=max(1,args.stride),
            downscale=args.downscale, fast=args.fast,
            max_seconds=args.max_seconds, show=args.show,
            save_csv=args.save_csv)

if __name__ == "__main__":
    main()
