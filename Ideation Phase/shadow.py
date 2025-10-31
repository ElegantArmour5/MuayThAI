# punch_classifier_types.py — MediaPipe-only punch type classifier (jab/cross/hook/uppercut)
# ---- Python 3.11 protobuf shim + quiet logs (harmless) ----
import collections, collections.abc, os
for _n in ("Mapping","MutableMapping","Sequence"):
    if not hasattr(collections,_n): setattr(collections,_n,getattr(collections.abc,_n))
os.environ.setdefault("GLOG_minloglevel","2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","3")

import cv2, argparse, math, time, pathlib, csv
from datetime import datetime
import numpy as np
import mediapipe as mp

# ======================= Tunables =======================
MODEL_COMPLEXITY        = 2      # 0=lite, 1=full, 2=heavy (more accurate)
VIS_THRESH              = 0.60   # min visibility for wrist/elbow/shoulder

# Event detection (per hand)
START_SPEED             = 0.045  # px/frame normalized (w.r.t image width) to start
START_EXT_DELTA         = 0.020  # minimal extension increase vs baseline to start
END_SPEED               = 0.020  # drop below this to end
END_RETRACT_DELTA       = 0.010  # retract vs peak to end
MIN_FRAMES_EVENT        = 5      # minimum frames in an event
MIN_FRAMES_BETWEEN      = 6      # refractory frames between events

# Baselines / smoothing
BASELINE_ALPHA          = 0.08   # guard baseline (extension) EMA
EXT_SMOOTH_ALPHA        = 0.35   # extension smoothing (EMA)
CALM_SPEED              = 0.030  # only update baseline when calm

# Classification thresholds
# Direction: angles in degrees; image x→ right, y→ down (so "up" is negative dy)
STRAIGHT_MAX_TILT       = 25     # |trajectory angle| ≤ this → straight (jab/cross)
ELBOW_STRAIGHT_MIN      = 150    # elbow ≥ this → straight arm
ROTATION_FOR_CROSS      = 10     # torso rotation (°) at peak to call it a cross

HOOK_ELBOW_RANGE        = (70, 130)  # elbow angle at peak for a hook
FOREARM_VERT_MIN        = 50     # |forearm angle to horizontal| ≥ 50° → near vertical
LATERAL_DOM_RATIO       = 1.2    # lateral (|dx|) ≥ ratio * forward extension delta

UPP_MIN_UP_ANGLE        = -35    # trajectory angle ≤ -35° (upwards) for uppercut
UPP_ELBOW_MAX           = 150    # not fully straight
UPP_MIN_LIFT_PX         = 25     # wrist must rise by at least this many pixels

# Visuals
COLOR_MAP = {
    "jab":       (0, 255, 0),
    "cross":     (255, 200, 0),
    "hook":      (0, 200, 255),
    "uppercut":  (255, 0, 255),
    "unknown":   (200, 200, 200),
}
# ========================================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --------------- Geometry helpers ---------------
def angle(a,b,c):
    ax,ay=a; bx,by=b; cx,cy=c
    abx,aby=ax-bx,ay-by; cbx,cby=cx-bx,cy-by
    dot=abx*cbx+aby*cby
    mab=max(1e-9,(abx**2+aby**2)**0.5); mcb=max(1e-9,(cbx**2+cby**2)**0.5)
    cosv=max(-1.0,min(1.0,dot/(mab*mcb)))
    return math.degrees(math.acos(cosv))

def lmk_xy(lms, idx, w, h):
    lm=lms[idx]; return np.array([lm.x*w, lm.y*h], np.float32)

def vis_ok(lms, idx): return getattr(lms[idx], "visibility", 0.0) >= VIS_THRESH

def length(a,b): return float(np.hypot(*(a-b)))

def deg_line(a,b):
    dx,dy = (b-a)
    return math.degrees(math.atan2(dy, dx))  # 0=right, +90=down, -90=up

# --------------- IO helpers ---------------
def derive_paths(video_path):
    if video_path:
        p=pathlib.Path(video_path); base=str(p.with_suffix(""))
        return f"{base}_annotated{p.suffix}", f"{base}_events.csv"
    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"webcam_{ts}.mp4", f"webcam_{ts}_events.csv"

# --------------- Per-hand tracker ---------------
class HandEventTracker:
    def __init__(self, left=True):
        self.left = left
        self.prev_wrist = None
        self.ext_ema = None
        self.baseline = None
        self.state = 0  # 0=idle, 1=active
        self.frames_active = 0
        self.refractory = 0

        # event buffers
        self.start_frame = None
        self.start_wrist = None
        self.start_ext = None
        self.peak_frame = None
        self.peak_wrist = None
        self.peak_ext = None
        self.peak_metrics = None

    def _ema(self, old, new, a): return new if old is None else (a*new + (1-a)*old)

    def update(self, lms, w, h, dt, frame_idx, fps):
        WR, EL, SH = (int(mp_pose.PoseLandmark.LEFT_WRIST),
                      int(mp_pose.PoseLandmark.LEFT_ELBOW),
                      int(mp_pose.PoseLandmark.LEFT_SHOULDER)) if self.left else \
                     (int(mp_pose.PoseLandmark.RIGHT_WRIST),
                      int(mp_pose.PoseLandmark.RIGHT_ELBOW),
                      int(mp_pose.PoseLandmark.RIGHT_SHOULDER))
        LSH, RSH = int(mp_pose.PoseLandmark.LEFT_SHOULDER), int(mp_pose.PoseLandmark.RIGHT_SHOULDER)
        LHP, RHP = int(mp_pose.PoseLandmark.LEFT_HIP), int(mp_pose.PoseLandmark.RIGHT_HIP)

        if not (vis_ok(lms, WR) and vis_ok(lms, EL) and vis_ok(lms, SH)):
            return None  # skip this frame

        wrist = lmk_xy(lms, WR, w, h)
        elbow = lmk_xy(lms, EL, w, h)
        sh    = lmk_xy(lms, SH, w, h)
        l_sh  = lmk_xy(lms, LSH, w, h)
        r_sh  = lmk_xy(lms, RSH, w, h)
        l_hip = lmk_xy(lms, LHP, w, h)
        r_hip = lmk_xy(lms, RHP, w, h)

        # normalization: shoulder span
        sh_span = max(1e-6, length(l_sh, r_sh))

        # speed
        if self.prev_wrist is None:
            speed = 0.0
        else:
            v = (wrist - self.prev_wrist) / max(dt,1e-3)
            speed = float(np.hypot(v[0], v[1])) / max(1.0, w)  # normalize by width
        self.prev_wrist = wrist.copy()

        # extension (normalized)
        ext = length(wrist, sh) / sh_span
        if self.ext_ema is None or speed < CALM_SPEED:
            self.ext_ema = self._ema(self.ext_ema, ext, EXT_SMOOTH_ALPHA)

        # elbow angle, forearm orientation, torso rotation
        elbow_ang = angle(sh, elbow, wrist)
        forearm_deg = abs(deg_line(elbow, wrist))  # 0≈horizontal, 90≈vertical
        sh_deg  = deg_line(l_sh, r_sh)
        hip_deg = deg_line(l_hip, r_hip)
        torso_rot = abs((sh_deg - hip_deg + 540) % 360 - 180)

        # update baseline only when calm
        if speed < CALM_SPEED:
            self.baseline = self._ema(self.baseline, ext, BASELINE_ALPHA)

        # early frames need baseline; otherwise skip
        if self.baseline is None or self.ext_ema is None:
            return None

        event = None
        if self.refractory > 0:
            self.refractory -= 1

        if self.state == 0:  # IDLE
            if self.refractory == 0 and (speed >= START_SPEED) and ((self.ext_ema - self.baseline) >= START_EXT_DELTA):
                # start event
                self.state = 1
                self.frames_active = 0
                self.start_frame = frame_idx
                self.start_wrist = wrist.copy()
                self.start_ext = self.ext_ema
                self.peak_frame = frame_idx
                self.peak_wrist = wrist.copy()
                self.peak_ext = self.ext_ema
                self.peak_metrics = (elbow_ang, forearm_deg, torso_rot)
        else:  # ACTIVE
            self.frames_active += 1
            # track peak by extension; use speed as tie-breaker via timing
            if self.ext_ema > (self.peak_ext + 1e-6):
                self.peak_ext = self.ext_ema
                self.peak_frame = frame_idx
                self.peak_wrist = wrist.copy()
                self.peak_metrics = (elbow_ang, forearm_deg, torso_rot)

            retract = (self.peak_ext - self.ext_ema) >= END_RETRACT_DELTA
            slow    = speed <= END_SPEED
            long_enough = self.frames_active >= MIN_FRAMES_EVENT
            if long_enough and (retract or slow):
                # finalize event
                traj_deg = deg_line(self.start_wrist, self.peak_wrist)  # −90 up, +90 down
                dx = float(self.peak_wrist[0] - self.start_wrist[0])
                dy = float(self.peak_wrist[1] - self.start_wrist[1])
                elbow_p, forearm_p, rot_p = self.peak_metrics
                lift_px = self.start_wrist[1] - self.peak_wrist[1]  # positive if wrist went up

                # lateral versus forward proxy: compare |dx| vs change in extension * sh_span
                ext_delta_px = (self.peak_ext - self.start_ext) * sh_span
                lateral_dom = abs(dx) >= (LATERAL_DOM_RATIO * max(1.0, ext_delta_px))

                # classification
                label = "unknown"
                # Straight?
                if (abs(traj_deg) <= STRAIGHT_MAX_TILT) and (elbow_p >= ELBOW_STRAIGHT_MIN):
                    label = "cross" if rot_p >= ROTATION_FOR_CROSS else "jab"
                # Hook?
                elif (HOOK_ELBOW_RANGE[0] <= elbow_p <= HOOK_ELBOW_RANGE[1]) and (forearm_p >= FOREARM_VERT_MIN) and lateral_dom:
                    label = "hook"
                # Uppercut?
                elif (traj_deg <= UPP_MIN_UP_ANGLE) and (elbow_p <= UPP_ELBOW_MAX) and (lift_px >= UPP_MIN_LIFT_PX):
                    label = "uppercut"

                event = {
                    "frame_start": self.start_frame,
                    "frame_peak": self.peak_frame,
                    "time_sec": self.peak_frame / max(1.0, fps),
                    "side": "L" if self.left else "R",
                    "type": label,
                    "traj_deg": traj_deg,
                    "elbow_deg": elbow_p,
                    "forearm_deg": forearm_p,
                    "torso_rot_deg": rot_p,
                    "dx": dx, "dy": dy, "lift_px": lift_px
                }

                # reset
                self.state = 0
                self.frames_active = 0
                self.refractory = MIN_FRAMES_BETWEEN
                self.start_frame = self.start_wrist = None
                self.peak_frame = self.peak_wrist = None
                self.start_ext = self.peak_ext = None
                self.peak_metrics = None

        return event

# -------------------- Main --------------------
def main():
    ap=argparse.ArgumentParser()
    g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--video", type=str)
    g.add_argument("--webcam", type=int)
    ap.add_argument("--display", action="store_true")
    args=ap.parse_args()

    cap = cv2.VideoCapture(args.video if args.video else args.webcam)
    if not cap.isOpened(): raise SystemExit("Could not open input")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps<=1 or np.isnan(fps): fps=30.0

    ok, frame = cap.read()
    if not ok: raise SystemExit("Could not read first frame")
    h, w = frame.shape[:2]
    out_video, out_csv = derive_paths(args.video)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (w,h))

    pose = mp_pose.Pose(static_image_mode=False, model_complexity=MODEL_COMPLEXITY,
                        enable_segmentation=False, min_detection_confidence=0.7,
                        min_tracking_confidence=0.7)

    left  = HandEventTracker(left=True)
    right = HandEventTracker(left=False)

    frame_idx=0
    last_time=time.time()
    events=[]

    # For drawing last detection label for a few frames
    label_timer = 0
    last_label  = None
    last_color  = (200,200,200)

    while True:
        frame_vis = frame.copy()
        rgb = cv2.cvtColor(frame_vis, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        now=time.time(); dt = now - last_time; last_time = now

        if res.pose_landmarks:
            lms = res.pose_landmarks.landmark

            e1 = left.update(lms, w, h, dt, frame_idx, fps)
            e2 = right.update(lms, w, h, dt, frame_idx, fps)

            if e1:
                events.append(e1)
                last_label = f"{e1['side']} {e1['type']}"
                last_color = COLOR_MAP.get(e1["type"], (200,200,200))
                label_timer = 12
            if e2:
                events.append(e2)
                last_label = f"{e2['side']} {e2['type']}"
                last_color = COLOR_MAP.get(e2["type"], (200,200,200))
                label_timer = 12

            # draw pose
            mp_drawing.draw_landmarks(
                frame_vis, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(60,220,60), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(230,230,230), thickness=2)
            )

        # HUD: counts by type
        if events:
            counts = {"jab":0,"cross":0,"hook":0,"uppercut":0,"unknown":0}
            for ev in events: counts[ev["type"]] = counts.get(ev["type"],0)+1
            hud = " | ".join([f"{k}:{counts.get(k,0)}" for k in ("jab","cross","hook","uppercut")])
            cv2.putText(frame_vis, hud, (12, h-16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 2, cv2.LINE_AA)

        if label_timer>0 and last_label:
            (tw,th), _ = cv2.getTextSize(last_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame_vis, (10,10), (10+tw+16, 10+th+16), (0,0,0), -1)
            cv2.putText(frame_vis, last_label, (18, 10+th+6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, last_color, 2, cv2.LINE_AA)
            label_timer -= 1

        writer.write(frame_vis)
        if args.display:
            cv2.imshow("Punch Type Classifier", frame_vis)
            if (cv2.waitKey(1) & 0xFF) in (27, ord('q')): break

        ok, frame = cap.read()
        frame_idx += 1
        if not ok: break

    cap.release(); writer.release(); cv2.destroyAllWindows()

    # Save CSV
    with open(out_csv, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["time_sec","frame_peak","side","type","traj_deg","elbow_deg","forearm_deg","torso_rot_deg","dx","dy","lift_px"])
        for ev in events:
            wcsv.writerow([f"{ev['time_sec']:.3f}", ev["frame_peak"], ev["side"], ev["type"],
                           f"{ev['traj_deg']:.1f}", f"{ev['elbow_deg']:.1f}", f"{ev['forearm_deg']:.1f}",
                           f"{ev['torso_rot_deg']:.1f}", f"{ev['dx']:.1f}", f"{ev['dy']:.1f}", f"{ev['lift_px']:.1f}"])

    # Quick summary
    totals = {}
    for ev in events: totals[ev["type"]] = totals.get(ev["type"],0)+1
    print("Saved annotated video:", out_video)
    print("Saved events CSV:     ", out_csv)
    print("Counts:", totals)

if __name__=="__main__":
    main()
