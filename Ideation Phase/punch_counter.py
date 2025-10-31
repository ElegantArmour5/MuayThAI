# boxer_form_checker_robust.py
# Robust side/¾-view boxer form checker
# - Scale: torso length (mid-shoulders → mid-hips), not shoulder span
# - Face ref: visibility-weighted NOSE ⊕ mouth-center, EMA-smoothed
# - Self-calibration: per-hand 1D k-means (near vs far); derive ENTER/EXIT/GUARD_MAX
# - Visibility gating, forwardness gate, waist override
# - Stabilizers: EMA, hysteresis, majority vote, cooldown, FPS-aware velocity gate
# - Outputs: annotated MP4 + CSV + tiny HTML

import os, io, base64, math, argparse
from collections import deque, Counter
from datetime import timedelta

import cv2
import numpy as np
import pandas as pd

try:
    import mediapipe as mp
except Exception as e:
    raise SystemExit("Install mediapipe: pip install mediapipe==0.10.*") from e

# ---------- Tunables ----------
EMA_ALPHA_FACE         = 0.35
EMA_ALPHA_DIST         = 0.30
STATE_VOTE_WINDOW      = 5
MIN_PUNCH_FRAMES       = 6
COOLDOWN_FRAMES        = 6
VEL_MIN_PER_30FPS      = 0.03     # scaled by (30/fps)
FORWARD_MIN_COS        = 0.05     # wrist must be at least slightly toward the face direction
ELBOW_MIN_STRAIGHT     = 150.0    # deg (when punching)

WAIST_MARGIN_TORSO     = 0.06     # wrist below (waist + margin*torso) => not punching

# k-means derived thresholds (fractions along [near, far])
ENTER_FRACTION         = 0.60
EXIT_FRACTION          = 0.35
GUARD_FRACTION         = 0.25

# Fallbacks if clustering is degenerate
FALLBACK_ENTER_DELTA   = 0.55
FALLBACK_EXIT_DELTA    = 0.30
FALLBACK_GUARD_DELTA   = 0.20

MIN_CLUSTER_SEP        = 0.08     # require at least this separation (in torso units)
VIS_MIN                = 0.60     # min visibility to trust a landmark
ALPHA_RED_OVERLAY      = 0.38
DRAW_SKELETON          = True

POSE = mp.solutions.pose
LM   = POSE.PoseLandmark

def to_xy(lm, w, h): return np.array([lm.x*w, lm.y*h], dtype=np.float32)
def dist(a,b): return float(np.linalg.norm(a-b))

def safe_xy_vis(lms, idx, w, h):
    try:
        p = lms[idx]
        return to_xy(p,w,h), float(p.visibility)
    except Exception:
        return None, 0.0

def angle_deg(a, b, c):
    if a is None or b is None or c is None: return None
    ba = a - b; bc = c - b
    nba, nbc = np.linalg.norm(ba), np.linalg.norm(bc)
    if nba < 1e-9 or nbc < 1e-9: return None
    cosang = np.clip(np.dot(ba, bc) / (nba*nbc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def ema_series(vals, alpha):
    out=[]; prev=None
    for v in vals:
        if v is None:
            out.append(prev)
        else:
            prev = v if prev is None else (alpha*v + (1-alpha)*prev)
            out.append(prev)
    return out

def kmeans1d(vals, k=2, iters=25):
    """Simple 1D k-means for non-None values."""
    x = np.array([v for v in vals if v is not None], dtype=np.float32)
    if x.size < k: return None
    # init by percentiles
    c = np.percentile(x, [100*i/(k-1) for i in range(k)]).astype(np.float32)
    for _ in range(iters):
        dists = np.abs(x[:,None]-c[None,:])
        lab = np.argmin(dists, axis=1)
        c_new = np.array([x[lab==j].mean() if np.any(lab==j) else c[j] for j in range(k)], dtype=np.float32)
        if np.allclose(c, c_new, atol=1e-4): break
        c = c_new
    # map back to full array labels if needed
    c.sort()
    return float(c[0]), float(c[-1])

def forward_cos(wrist, shoulder_mid, face_ref):
    if wrist is None or shoulder_mid is None or face_ref is None: return None
    a = wrist - shoulder_mid
    b = face_ref - shoulder_mid
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na<1e-6 or nb<1e-6: return None
    return float(np.dot(a,b)/(na*nb))

def collect(video):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened(): raise SystemExit(f"Cannot open: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT)>0 else None

    pose = POSE.Pose(model_complexity=1, enable_segmentation=False, smooth_landmarks=True)

    L_dist_raw=[]; R_dist_raw=[]
    L_fwd=[]; R_fwd=[]
    torso_list=[]; ok=[]
    face_ref_pts=[]; face_ref_vis=[]

    while True:
        okr, frame = cap.read()
        if not okr: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if not res.pose_landmarks:
            L_dist_raw.append(None); R_dist_raw.append(None)
            L_fwd.append(None); R_fwd.append(None)
            torso_list.append(None); ok.append(False)
            face_ref_pts.append(None); face_ref_vis.append(0.0)
            continue

        lms = res.pose_landmarks.landmark

        lsh,vlsh = safe_xy_vis(lms, LM.LEFT_SHOULDER,  W,H)
        rsh,vrsh = safe_xy_vis(lms, LM.RIGHT_SHOULDER, W,H)
        lhp,vlhp = safe_xy_vis(lms, LM.LEFT_HIP,       W,H)
        rhp,vrhp = safe_xy_vis(lms, LM.RIGHT_HIP,      W,H)
        lwr,vlwr = safe_xy_vis(lms, LM.LEFT_WRIST,     W,H)
        rwr,vrwr = safe_xy_vis(lms, LM.RIGHT_WRIST,    W,H)
        mL,vmL   = safe_xy_vis(lms, LM.MOUTH_LEFT,     W,H)
        mR,vmR   = safe_xy_vis(lms, LM.MOUTH_RIGHT,    W,H)
        nose,vn  = safe_xy_vis(lms, LM.NOSE,           W,H)

        # visibility sanity
        vis_ok = all(v >= VIS_MIN for v in [vlsh,vrsh,vlhp,vrhp]) and ((vlwr>=VIS_MIN) or (vrwr>=VIS_MIN))
        if not vis_ok:
            L_dist_raw.append(None); R_dist_raw.append(None)
            L_fwd.append(None); R_fwd.append(None)
            torso_list.append(None); ok.append(False)
            face_ref_pts.append(None); face_ref_vis.append(0.0)
            continue

        sh_mid = None
        if lsh is not None and rsh is not None:
            sh_mid = 0.5*(lsh+rsh)
        hip_mid = None
        if lhp is not None and rhp is not None:
            hip_mid = 0.5*(lhp+rhp)
        torso = dist(sh_mid, hip_mid) if (sh_mid is not None and hip_mid is not None) else None
        torso_list.append(torso)

        # face reference (visibility-weighted nose & mouth-center)
        face = None; fvis = 0.0
        if nose is not None and (mL is not None and mR is not None):
            mouth = 0.5*(mL+mR)
            w_n = vn; w_m = 0.5*(vmL+vmR)
            if w_n<1e-6 and w_m<1e-6:
                face = None; fvis = 0.0
            else:
                face = (w_n*nose + w_m*mouth) / (w_n + w_m)
                fvis = (w_n + w_m)/2.0
        elif nose is not None:
            face = nose; fvis = vn
        elif (mL is not None and mR is not None):
            face = 0.5*(mL+mR); fvis = 0.5*(vmL+vmR)

        face_ref_pts.append(face); face_ref_vis.append(fvis)

        if torso is None or face is None:
            L_dist_raw.append(None); R_dist_raw.append(None)
            L_fwd.append(None); R_fwd.append(None); ok.append(False)
            continue

        # normalized distances by torso length
        Ld = dist(lwr, face)/ (torso+1e-9) if lwr is not None and vlwr>=VIS_MIN else None
        Rd = dist(rwr, face)/ (torso+1e-9) if rwr is not None and vrwr>=VIS_MIN else None
        L_dist_raw.append(Ld); R_dist_raw.append(Rd)

        # forwardness (cosine) relative to shoulder-mid → face vector
        fc = forward_cos(lwr, sh_mid, face); L_fwd.append(fc)
        fc = forward_cos(rwr, sh_mid, face); R_fwd.append(fc)
        ok.append(True)

    cap.release(); pose.close()

    # Smooth face ref (for drawing only)
    face_sm = ema_series(face_ref_pts, EMA_ALPHA_FACE)
    # Smooth distances
    L_ema = ema_series(L_dist_raw, EMA_ALPHA_DIST)
    R_ema = ema_series(R_dist_raw, EMA_ALPHA_DIST)

    return fps, W, H, total, L_ema, R_ema, L_fwd, R_fwd, torso_list, face_sm, ok

def derive_thresholds(series):
    """series: EMA distances (normalized by torso)"""
    res = kmeans1d(series, k=2, iters=25)
    if res is None:
        near, far = None, None
    else:
        near, far = res
    if (near is None) or (far is None) or ((far - near) < MIN_CLUSTER_SEP):
        # fallback: use robust percentiles
        xs = np.array([v for v in series if v is not None], dtype=np.float32)
        if xs.size < 16:  # too few points
            return None
        near = float(np.percentile(xs, 30))
        far  = float(np.percentile(xs, 85))
        if far <= near: far = near + FALLBACK_ENTER_DELTA

        enter = near + FALLBACK_ENTER_DELTA
        exit_ = near + FALLBACK_EXIT_DELTA
        guard = near + FALLBACK_GUARD_DELTA
    else:
        enter = near + ENTER_FRACTION*(far - near)
        exit_ = near + EXIT_FRACTION*(far - near)
        guard = near + GUARD_FRACTION*(far - near)

    return {"near":near, "far":far, "enter":enter, "exit":exit_, "guard":guard}

def analyze(video, out_video, out_csv, out_html):
    (fps, W, H, total, L_ema, R_ema, L_fwd, R_fwd, TORSO, FACE, ok_mask) = collect(video)
    vgate = VEL_MIN_PER_30FPS * (30.0/max(1e-6, fps))

    # Per-hand velocity of EMA
    def vel_of(series):
        v=[None]
        for i in range(1,len(series)):
            if series[i] is None or series[i-1] is None: v.append(None)
            else: v.append(series[i]-series[i-1])
        return v
    L_vel = vel_of(L_ema); R_vel = vel_of(R_ema)

    # Per-hand thresholds
    L_th = derive_thresholds(L_ema)
    R_th = derive_thresholds(R_ema)

    pose = POSE.Pose(model_complexity=2, enable_segmentation=False, smooth_landmarks=True)
    cap = cv2.VideoCapture(video)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (W,H))
    draw = mp.solutions.drawing_utils
    spec = draw.DrawingSpec(thickness=2, circle_radius=2)

    rows=[]; i=0
    L_state = {"is":False, "since":0, "cool":0}
    R_state = {"is":False, "since":0, "cool":0}
    L_vote = deque(maxlen=STATE_VOTE_WINDOW); R_vote = deque(maxlen=STATE_VOTE_WINDOW)

    while True:
        okr, frame = cap.read()
        if not okr: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        violations=[]
        waist_y = None; sh_mid=None
        L_waist_block = R_waist_block = False
        L_elb = R_elb = None

        if res.pose_landmarks:
            lms = res.pose_landmarks.landmark
            lsh,vlsh = safe_xy_vis(lms, LM.LEFT_SHOULDER,  W,H)
            rsh,vrsh = safe_xy_vis(lms, LM.RIGHT_SHOULDER, W,H)
            lhp,vlhp = safe_xy_vis(lms, LM.LEFT_HIP,       W,H)
            rhp,vrhp = safe_xy_vis(lms, LM.RIGHT_HIP,      W,H)
            lwr,vlwr = safe_xy_vis(lms, LM.LEFT_WRIST,     W,H)
            rwr,vrwr = safe_xy_vis(lms, LM.RIGHT_WRIST,    W,H)

            # basic vis gate
            if not (vlsh>=VIS_MIN and vrsh>=VIS_MIN and vlhp>=VIS_MIN and vrhp>=VIS_MIN):
                writer.write(frame); rows.append({"frame":i,"violations":"VisLow"}); i+=1; continue

            if lsh is not None and rsh is not None: sh_mid = 0.5*(lsh+rsh)
            if lhp is not None and rhp is not None: waist_y = 0.5*(lhp[1]+rhp[1])

            torso = TORSO[i] if i < len(TORSO) else None
            if torso is None: writer.write(frame); rows.append({"frame":i,"violations":"NoTorso"}); i+=1; continue
            margin_px = WAIST_MARGIN_TORSO * torso

            if lwr is not None and waist_y is not None and (lwr[1] > waist_y + margin_px):
                L_waist_block = True
            if rwr is not None and waist_y is not None and (rwr[1] > waist_y + margin_px):
                R_waist_block = True

            # elbow angles
            def elbow(side):
                if side=="L":
                    a=lms[LM.LEFT_SHOULDER]; b=lms[LM.LEFT_ELBOW]; c=lms[LM.LEFT_WRIST]
                else:
                    a=lms[LM.RIGHT_SHOULDER]; b=lms[LM.RIGHT_ELBOW]; c=lms[LM.RIGHT_WRIST]
                return angle_deg(to_xy(a,W,H), to_xy(b,W,H), to_xy(c,W,H))
            L_elb = elbow("L"); R_elb = elbow("R")

            # forwardness
            Lcos = L_fwd[i] if i<len(L_fwd) else None
            Rcos = R_fwd[i] if i<len(R_fwd) else None

            # helper to update a side’s punch state
            def update(side, state, ema, vel, th, cos_fwd, waist_block, vote):
                if th is None or ema is None:
                    vote.append(False)
                else:
                    # provisional state
                    if waist_block:
                        prov = False
                    else:
                        if not state["is"]:
                            if state["cool"]>0:
                                prov=False
                            else:
                                prov = (ema>th["enter"] and (vel or 0)>vgate and (cos_fwd or 0)>FORWARD_MIN_COS)
                        else:
                            if state["since"]<MIN_PUNCH_FRAMES:
                                prov=True
                            else:
                                prov = (ema>=th["exit"])
                    vote.append(prov)
                # majority vote
                decided = (sum(1 for x in vote if x) > len(vote)//2)
                # apply
                if decided and not state["is"]:
                    state["is"]=True; state["since"]=0
                elif (not decided) and state["is"]:
                    state["is"]=False; state["since"]=0; state["cool"]=COOLDOWN_FRAMES
                else:
                    state["since"]+=1
                if state["cool"]>0: state["cool"]-=1

            # per side data
            Le, Re = (L_ema[i] if i<len(L_ema) else None), (R_ema[i] if i<len(R_ema) else None)
            Lv, Rv = (L_vel[i] if i<len(L_vel) else None), (R_vel[i] if i<len(R_vel) else None)

            update("L", L_state, Le, Lv, L_th, Lcos, L_waist_block, L_vote)
            update("R", R_state, Re, Rv, R_th, Rcos, R_waist_block, R_vote)

            punching = {"L":L_state["is"], "R":R_state["is"]}

            # rules
            if punching["L"] and L_elb is not None and L_elb < ELBOW_MIN_STRAIGHT:
                violations.append("ElbowBent(L)")
            if punching["R"] and R_elb is not None and R_elb < ELBOW_MIN_STRAIGHT:
                violations.append("ElbowBent(R)")

            if punching["R"] and L_th is not None and Le is not None and Le > L_th["guard"]:
                violations.append("GuardAway(L)")
            if punching["L"] and R_th is not None and Re is not None and Re > R_th["guard"]:
                violations.append("GuardAway(R)")

        else:
            violations.append("NoPose")

        # annotate
        out = frame.copy()
        if violations:
            overlay = out.copy()
            cv2.rectangle(overlay, (0,0), (W,H), (0,0,255), -1)
            cv2.addWeighted(overlay, ALPHA_RED_OVERLAY, out, 1-ALPHA_RED_OVERLAY, 0, out)
        if res.pose_landmarks and DRAW_SKELETON:
            draw.draw_landmarks(out, res.pose_landmarks, POSE.POSE_CONNECTIONS, spec, spec)

        tag = "OK" if not violations else "FORM ISSUE: " + ", ".join(violations[:3]) + ("..." if len(violations)>3 else "")
        color = (64,200,64) if not violations else (0,0,255)
        hud = f"Frame {i} | {str(timedelta(seconds=i/max(1,fps)))[:-3]}"
        cv2.putText(out, hud, (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2,cv2.LINE_AA)
        cv2.putText(out, tag, (12,58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color,2,cv2.LINE_AA)

        writer.write(out)

        rows.append({
            "frame": i,
            "time_s": i/max(1,fps),
            "violations": ";".join(violations),
            "L_is_punch": int(L_state["is"]),
            "R_is_punch": int(R_state["is"]),
            "L_dist_torso_ema": L_ema[i] if i<len(L_ema) else None,
            "R_dist_torso_ema": R_ema[i] if i<len(R_ema) else None,
            "L_vel": L_vel[i] if i<len(L_vel) else None,
            "R_vel": R_vel[i] if i<len(R_vel) else None,
            "L_elbow_deg": L_elb, "R_elbow_deg": R_elb
        })
        i+=1

    cap.release(); writer.release(); pose.close()

    # outputs
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    df["red"] = df["violations"].apply(lambda s: bool(s) and len(s)>0)
    red_pct = 100.0*df["red"].mean()
    # simple HTML
    html = io.StringIO()
    html.write(f"""<!doctype html><html><head><meta charset="utf-8"><title>Boxer Form Report (Robust)</title>
<style>body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:24px}}
.small{{color:#666}} .card{{border:1px solid #e5e7eb;border-radius:12px;padding:16px;margin:12px 0}}</style></head><body>
<h1>Boxer Form Report (Robust)</h1>
<div class="small">Output: {os.path.basename(out_video)}</div>
<div class="card"><b>Red frames</b>: {red_pct:.1f}%</div>
<div class="card"><pre>L thresholds: {L_th}\nR thresholds: {R_th}</pre></div>
</body></html>""")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html.getvalue())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--report_dir", default=None)
    args = ap.parse_args()

    stem,_ = os.path.splitext(args.video)
    out = args.out or (stem + "_annot_robust.mp4")
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
