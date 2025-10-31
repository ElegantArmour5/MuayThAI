# boxer_form_checker_waistup_capable.py
# Works for full body AND waist-up videos:
# - Primary scale: torso (mid-shoulders → mid-hips), real waist at hip_mid
# - Fallback scale (no torso): robust UPPER-BODY scale using shoulders/head/face
#   + virtual waist line derived from shoulders & upper-body scale
# - Normalized wrist–mouth distance (EMA) + elbow angle
# - Self-calibrated thresholds (per-hand near/far clustering)
# - Stabilized states: EMA, hysteresis, majority vote, cooldown
# - Guard only when other hand is punching
# - "Below waist ⇒ NOT punching" using real OR virtual waist
# - Outputs: annotated MP4 + CSV + HTML mini-report

import os, io, base64, math, argparse
from collections import deque
from datetime import timedelta
import cv2, numpy as np, pandas as pd

try:
    import mediapipe as mp
except Exception as e:
    raise SystemExit("Install mediapipe: pip install mediapipe==0.10.*") from e

# -------------------- Tunables --------------------
# Smoothing
# --- Tuned for your v2 (wrong) vs v8 (right) pair ---

# Smoothing
EMA_ALPHA_SCALE    = 0.25
EMA_ALPHA_FACE     = 0.35
EMA_ALPHA_DIST     = 0.35   # slightly more smoothing than v2

# State machine
STATE_VOTE_WINDOW  = 5
MIN_PUNCH_FRAMES   = 6
COOLDOWN_FRAMES    = 6

# Make punch start a bit harder, and discourage “slow drift” starts
VEL_MIN_PER_30FPS  = 0.020  # was 0.015
SLOW_EXT_FRAMES    = 6      # was 4

# Keep elbow strict enough to catch v2’s mistakes, but not overly picky
ELBOW_MIN_STRAIGHT = 152.0  # was 150.0

# Self-calibrated thresholds (fractions along [near, far])
# Harder to start punching, easier to keep guard
ENTER_FRACTION     = 0.52   # was 0.45
EXIT_FRACTION      = 0.33   # was 0.28
GUARD_FRACTION     = 0.27   # was 0.22  (more lenient guard)

MIN_CLUSTER_SEP    = 0.05

# Visibility
VIS_MIN            = 0.50
VIS_WRIST_MIN      = 0.45

# Waist (unchanged; still blocks “punching” if hand is below waist)
WAIST_MARGIN_TORSO = 0.06
UPPER_WAIST_FRAC   = 1.05

# Red overlay alpha (unchanged)
ALPHA_RED_OVERLAY  = 0.38

DRAW_SKELETON      = True

POSE = mp.solutions.pose
LM   = POSE.PoseLandmark

# -------------------- Helper functions --------------------
def to_xy(lm, w, h): return np.array([lm.x*w, lm.y*h], dtype=np.float32)
def dist(a,b): return float(np.linalg.norm(a-b))

def safe_xy_vis(lms, idx, w, h):
    try:
        p = lms[idx]; return to_xy(p,w,h), float(p.visibility)
    except Exception:
        return None, 0.0

def angle_deg(a,b,c):
    if a is None or b is None or c is None: return None
    ba=a-b; bc=c-b
    nba, nbc = np.linalg.norm(ba), np.linalg.norm(bc)
    if nba<1e-9 or nbc<1e-9: return None
    cosang = np.clip(np.dot(ba,bc)/(nba*nbc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def ema_series(vals, alpha):
    out=[]; prev=None
    for v in vals:
        if v is None: out.append(prev)
        else:
            prev = v if prev is None else (alpha*v + (1-alpha)*prev)
            out.append(prev)
    return out

def robust_near_far(series, min_sep=MIN_CLUSTER_SEP):
    xs = np.array([v for v in series if v is not None], dtype=np.float32)
    if xs.size < 20: return None
    q25, q50, q75, q85 = np.percentile(xs, [25,50,75,85])
    near = float(q25); far = float(max(q85, q75 + min_sep))
    if far - near < min_sep:
        far = near + min_sep
    enter = near + ENTER_FRACTION*(far-near)
    exit_ = near + EXIT_FRACTION *(far-near)
    guard = near + GUARD_FRACTION*(far-near)
    return {"near":near,"far":far,"enter":enter,"exit":exit_,"guard":guard}

# -------------------- Scale & Waist Estimator --------------------
class ScaleWaistEstimator:
    """
    Computes a per-frame scale and waist_y with smoothing.
    Prefers real torso (mid-shoulders ↔ mid-hips). If unavailable, uses an UPPER-BODY composite scale
    and a virtual waist derived from shoulders.
    """
    def __init__(self):
        self.scale_ema = None
        self.waist_ema = None
        self.src = "unknown"

    def _upper_body_scale(self, lsh, rsh, nose, mouth, left_ear, right_ear):
        comps = []
        if lsh is not None and rsh is not None:
            comps.append(dist(lsh, rsh))  # shoulder span
        if left_ear is not None and right_ear is not None:
            comps.append(1.8 * dist(left_ear, right_ear))  # head width scaled
        if nose is not None and lsh is not None and rsh is not None:
            sh_mid = 0.5*(lsh+rsh); comps.append(1.6 * dist(nose, sh_mid))
        if mouth is not None and lsh is not None and rsh is not None:
            sh_mid = 0.5*(lsh+rsh); comps.append(1.3 * dist(mouth, sh_mid))
        if not comps: return None
        # use median to reduce outliers
        return float(np.median(np.array(comps, dtype=np.float32)))

    def estimate(self, lms, W, H):
        # Landmarks we might use
        lsh,vlsh = safe_xy_vis(lms, LM.LEFT_SHOULDER,  W,H)
        rsh,vrsh = safe_xy_vis(lms, LM.RIGHT_SHOULDER, W,H)
        lhp,vlhp = safe_xy_vis(lms, LM.LEFT_HIP,       W,H)
        rhp,vrhp = safe_xy_vis(lms, LM.RIGHT_HIP,      W,H)
        nose,vn  = safe_xy_vis(lms, LM.NOSE,           W,H)
        mL,vmL   = safe_xy_vis(lms, LM.MOUTH_LEFT,     W,H)
        mR,vmR   = safe_xy_vis(lms, LM.MOUTH_RIGHT,    W,H)
        le, vle  = safe_xy_vis(lms, LM.LEFT_EAR,       W,H)
        re, vre  = safe_xy_vis(lms, LM.RIGHT_EAR,      W,H)

        mouth = 0.5*(mL+mR) if (mL is not None and mR is not None) else None

        # Prefer real torso if both shoulders & hips are reasonably visible
        torso_scale = None; waist_y = None; src = "upper"
        if (vlsh>=VIS_MIN and vrsh>=VIS_MIN and vlhp>=VIS_MIN and vrhp>=VIS_MIN
            and lsh is not None and rsh is not None and lhp is not None and rhp is not None):
            sh_mid = 0.5*(lsh+rsh)
            hip_mid= 0.5*(lhp+rhp)
            t = dist(sh_mid, hip_mid)
            if t >= 10.0:   # prevent degenerate cases
                torso_scale = t
                waist_y = hip_mid[1]
                src = "torso"

        # If no torso, compute upper-body scale & virtual waist
        if torso_scale is None:
            ub_scale = self._upper_body_scale(lsh, rsh, nose, mouth, le, re)
            if ub_scale is None:
                return None, None, "none"
            # virtual waist: shoulder_mid_y + fraction * ub_scale
            sh_mid = None
            if lsh is not None and rsh is not None: sh_mid = 0.5*(lsh+rsh)
            if sh_mid is None:
                return None, None, "none"
            waist_y = sh_mid[1] + UPPER_WAIST_FRAC * ub_scale
            scale = ub_scale
        else:
            scale = torso_scale

        # EMA smoothing for stability
        if self.scale_ema is None: self.scale_ema = scale
        else: self.scale_ema = EMA_ALPHA_SCALE*scale + (1-EMA_ALPHA_SCALE)*self.scale_ema

        if self.waist_ema is None: self.waist_ema = waist_y
        else: self.waist_ema = EMA_ALPHA_SCALE*waist_y + (1-EMA_ALPHA_SCALE)*self.waist_ema

        self.src = src
        return self.scale_ema, self.waist_ema, src

# -------------------- Pass 1: collect normalized series --------------------
def collect_series(video):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened(): raise SystemExit(f"Cannot open: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose = POSE.Pose(model_complexity=1, enable_segmentation=False, smooth_landmarks=True)
    scaler = ScaleWaistEstimator()

    # raw distances (before EMA)
    L_norm_raw, R_norm_raw = [], []
    scales, waists, srcs = [], [], []
    face_pts=[]

    while True:
        okr, frame = cap.read()
        if not okr: break
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            L_norm_raw.append(None); R_norm_raw.append(None)
            scales.append(None); waists.append(None); srcs.append("none"); face_pts.append(None); continue

        lms = res.pose_landmarks.landmark
        # face reference: prefer mouth-center, fallback to nose
        mL,vmL = safe_xy_vis(lms, LM.MOUTH_LEFT,  W,H)
        mR,vmR = safe_xy_vis(lms, LM.MOUTH_RIGHT, W,H)
        mouth = 0.5*(mL+mR) if (mL is not None and mR is not None) else None
        nose,vn= safe_xy_vis(lms, LM.NOSE,        W,H)
        face = mouth if mouth is not None else nose

        # wrists
        lwr,vlwr = safe_xy_vis(lms, LM.LEFT_WRIST,  W,H)
        rwr,vrwr = safe_xy_vis(lms, LM.RIGHT_WRIST, W,H)

        # per-frame scale & waist
        scale, waist_y, src = scaler.estimate(lms, W, H)

        if scale is None or face is None:
            L_norm_raw.append(None); R_norm_raw.append(None)
            scales.append(None); waists.append(None); srcs.append("none"); face_pts.append(None); continue

        Ld = dist(lwr, face)/scale if (lwr is not None and vlwr>=VIS_WRIST_MIN) else None
        Rd = dist(rwr, face)/scale if (rwr is not None and vrwr>=VIS_WRIST_MIN) else None

        L_norm_raw.append(Ld); R_norm_raw.append(Rd)
        scales.append(scale); waists.append(waist_y); srcs.append(src); face_pts.append(face)

    cap.release(); pose.close()

    # smooth series
    L_ema = ema_series(L_norm_raw, EMA_ALPHA_DIST)
    R_ema = ema_series(R_norm_raw, EMA_ALPHA_DIST)
    # keep face smoothed for drawing (not required for logic)
    face_sm=[]
    prev=None
    for p in face_pts:
        if p is None: face_sm.append(prev)
        else:
            prev = p if prev is None else (EMA_ALPHA_FACE*p + (1-EMA_ALPHA_FACE)*prev)
            face_sm.append(prev)

    return fps, W, H, L_ema, R_ema, scales, waists, srcs, face_sm

# -------------------- Pass 2: thresholds, states, annotate --------------------
def analyze(video, out_video, out_csv, out_html):
    fps, W, H, L_ema, R_ema, SCALE, WAIST, SRC, FACE = collect_series(video)
    vgate = VEL_MIN_PER_30FPS * (30.0/max(1e-6,fps))

    # velocities
    def vel(series):
        out=[None]
        for i in range(1,len(series)):
            a,b=series[i-1], series[i]
            out.append(None if (a is None or b is None) else (b-a))
        return out
    L_vel, R_vel = vel(L_ema), vel(R_ema)

    # thresholds per hand (near/far)
    L_th = robust_near_far(L_ema)
    R_th = robust_near_far(R_ema)

    pose = POSE.Pose(model_complexity=1, enable_segmentation=False, smooth_landmarks=True)
    cap  = cv2.VideoCapture(video)
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W,H))
    draw = mp.solutions.drawing_utils
    spec = draw.DrawingSpec(thickness=2, circle_radius=2)

    rows=[]; i=0
    L_state = {"is":False, "since":0, "cool":0, "above":0}
    R_state = {"is":False, "since":0, "cool":0, "above":0}
    L_vote = deque(maxlen=STATE_VOTE_WINDOW); R_vote = deque(maxlen=STATE_VOTE_WINDOW)

    while True:
        okr, frame = cap.read()
        if not okr: break
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        violations=[]
        reason=[]
        L_elb=R_elb=None
        L_block=R_block=False
        waist_y = WAIST[i] if i<len(WAIST) else None
        scale_i = SCALE[i] if i<len(SCALE) else None
        src_i   = SRC[i] if i<len(SRC) else "none"

        if not res.pose_landmarks:
            reason.append("no_pose")
        else:
            lms = res.pose_landmarks.landmark
            # elbows
            def elbow(side):
                if side=="L":
                    a=lms[LM.LEFT_SHOULDER]; b=lms[LM.LEFT_ELBOW]; c=lms[LM.LEFT_WRIST]
                else:
                    a=lms[LM.RIGHT_SHOULDER]; b=lms[LM.RIGHT_ELBOW]; c=lms[LM.RIGHT_WRIST]
                return angle_deg(to_xy(a,W,H), to_xy(b,W,H), to_xy(c,W,H))
            L_elb, R_elb = elbow("L"), elbow("R")

            # waist blocks (real or virtual waist)
            lwr,vlwr = safe_xy_vis(lms, LM.LEFT_WRIST,  W,H)
            rwr,vrwr = safe_xy_vis(lms, LM.RIGHT_WRIST, W,H)
            if waist_y is None or scale_i is None:
                reason.append("no_scale_waist")
            else:
                margin = WAIST_MARGIN_TORSO * scale_i
                if lwr is not None and vlwr>=VIS_WRIST_MIN and lwr[1] > waist_y + margin: L_block=True
                if rwr is not None and vrwr>=VIS_WRIST_MIN and rwr[1] > waist_y + margin: R_block=True
                if lwr is None or vlwr<VIS_WRIST_MIN: reason.append("low_vis_L")
                if rwr is None or vrwr<VIS_WRIST_MIN: reason.append("low_vis_R")

        # series
        Le = L_ema[i] if i<len(L_ema) else None
        Re = R_ema[i] if i<len(R_ema) else None
        Lv = L_vel[i] if i<len(L_vel) else None
        Rv = R_vel[i] if i<len(R_vel) else None

        # state update (no forwardness gate; rely on vel or slow-extend)
        def update(state, ema, vel, th, waist_block, vote):
            if th is None or ema is None:
                vote.append(False); return
            if ema > th["enter"]: state["above"] += 1
            else: state["above"] = 0

            if waist_block:
                prov=False
            else:
                if not state["is"]:
                    if state["cool"]>0: prov=False
                    else: prov = (ema>th["enter"] and ((vel or 0)>vgate or state["above"]>=SLOW_EXT_FRAMES))
                else:
                    if state["since"]<MIN_PUNCH_FRAMES: prov=True
                    else: prov = (ema>=th["exit"])

            vote.append(prov)
            decided = (sum(1 for x in vote if x) > len(vote)//2)
            if decided and not state["is"]:
                state["is"]=True; state["since"]=0
            elif (not decided) and state["is"]:
                state["is"]=False; state["since"]=0; state["cool"]=COOLDOWN_FRAMES
            else:
                state["since"]+=1
            if state["cool"]>0: state["cool"]-=1

        update(L_state, Le, Lv, L_th, L_block, L_vote)
        update(R_state, Re, Rv, R_th, R_block, R_vote)

        punching = {"L":L_state["is"], "R":R_state["is"]}

        # rules
        if punching["L"] and L_elb is not None and L_elb < ELBOW_MIN_STRAIGHT: violations.append("ElbowBent(L)")
        if punching["R"] and R_elb is not None and R_elb < ELBOW_MIN_STRAIGHT: violations.append("ElbowBent(R)")
        if punching["R"] and L_th is not None and Le is not None and Le > L_th["guard"]: violations.append("GuardAway(L)")
        if punching["L"] and R_th is not None and Re is not None and Re > R_th["guard"]: violations.append("GuardAway(R)")

        # annotate
        out = frame.copy()
        if violations:
            overlay = out.copy()
            cv2.rectangle(overlay, (0,0), (W,H), (0,0,255), -1)
            cv2.addWeighted(overlay, ALPHA_RED_OVERLAY, out, 1-ALPHA_RED_OVERLAY, 0, out)

        # draw waist line
        if waist_y is not None:
            cv2.line(out, (0, int(waist_y)), (W, int(waist_y)),
                     (180, 180, 180) if src_i=="torso" else (160, 200, 160), 1, cv2.LINE_AA)
        # skeleton
        if res.pose_landmarks and DRAW_SKELETON:
            draw.draw_landmarks(out, res.pose_landmarks, POSE.POSE_CONNECTIONS, spec, spec)

        # HUD
        tag = "OK" if not violations else "FORM ISSUE: " + ", ".join(violations[:3]) + ("..." if len(violations)>3 else "")
        color = (64,200,64) if not violations else (0,0,255)
        hud = f"Frame {i} | {str(timedelta(seconds=i/max(1,fps)))[:-3]} | scale:{src_i}"
        cv2.putText(out, hud, (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2,cv2.LINE_AA)
        cv2.putText(out, tag, (12,58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color,2,cv2.LINE_AA)
        writer.write(out)

        rows.append({
            "frame": i,
            "time_s": i/max(1,fps),
            "violations": ";".join(violations),
            "L_is_punch": int(punching["L"]),
            "R_is_punch": int(punching["R"]),
            "L_dist_norm_ema": Le, "R_dist_norm_ema": Re,
            "L_vel": Lv, "R_vel": Rv,
            "scale": scale_i, "waist_y": waist_y, "scale_src": src_i,
            "reasons": ",".join(reason) if reason else ""
        })
        i+=1

    cap.release(); writer.release(); pose.close()

    # outputs
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    red_pct = 100.0*df["violations"].str.len().gt(0).mean()
    html = io.StringIO()
    html.write(f"""<!doctype html><html><head><meta charset="utf-8"><title>Boxer Form Report (Waist-Up Capable)</title>
<style>body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:24px}}
.small{{color:#666}} .card{{border:1px solid #e5e7eb;border-radius:12px;padding:16px;margin:12px 0}}</style></head><body>
<h1>Boxer Form Report (Waist-Up Capable)</h1>
<div class="card"><b>Red frames</b>: {red_pct:.1f}%</div>
<div class="card"><pre>Left thresholds:  {L_th}\nRight thresholds: {R_th}</pre></div>
<div class="card"><div class="small">Tip: if detections are missing, check CSV columns <code>scale_src</code> (torso/upper) and <code>reasons</code> (e.g., low_vis).</div></div>
</body></html>""")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html.getvalue())

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--report_dir", default=None)
    args = ap.parse_args()

    stem,_ = os.path.splitext(args.video)
    out = args.out or (stem + "_annot_waistup.mp4")
    report_dir = args.report_dir or os.path.dirname(os.path.abspath(out)) or "."
    os.makedirs(report_dir, exist_ok=True)
    csv  = os.path.join(report_dir, os.path.splitext(os.path.basename(out))[0] + "_frames.csv")
    html = os.path.join(report_dir, os.path.splitext(os.path.basename(out))[0] + "_report.html")
    analyze(args.video, out, csv, html)
    print("Annotated video:", out)
    print("Frame CSV:", csv)
    print("HTML report:", html)

if __name__ == "__main__":
    main()
