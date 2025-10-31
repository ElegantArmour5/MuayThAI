# training_pace_adjusted.py
# Robust stance detection for fast, partial-body Muay Thai clips.
# - Torso-focused (shoulders + forearms + head yaw)
# - One-Euro smoothing per landmark
# - NO-SKIP blurred frames (quality-weighted fusion instead)
# - Optional live preview (--show) and saving via ffmpeg H.264 (--save out.mp4)

import os
import math
import time
import shlex
import argparse
import subprocess
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

# =========================== One-Euro Smoothing ===========================
class OneEuro:
    def __init__(self, freq, min_cutoff=1.2, beta=0.03, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / max(self.freq, 1e-6)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x):
        dx = 0.0 if self.x_prev is None else (x - self.x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = x if self.x_prev is None else a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat

class LMFilter:
    def __init__(self, fps):
        # Heavier damping on z to tame depth spikes
        self.fx = OneEuro(fps, 1.6, 0.035)
        self.fy = OneEuro(fps, 1.6, 0.035)
        self.fz = OneEuro(fps, 1.3, 0.025)

    def apply(self, p):  # p: np.array([x,y,z,vis])
        if p is None:
            return None
        return np.array([self.fx.filter(p[0]), self.fy.filter(p[1]), self.fz.filter(p[2]), p[3]], dtype=np.float32)

# =========================== Optical Flow Carry ===========================
class FlowCarry:
    """Carry last good pixel points forward for a few frames when landmarks drop."""
    def __init__(self, grace=5):
        self.prev_gray = None
        self.cache = {}  # name -> {'pt':(x,y), 'age':int}
        self.grace = grace

    def update(self, gray, pts_dict):
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            for k, p in pts_dict.items():
                if p is not None:
                    self.cache[k] = {'pt': (p[0], p[1]), 'age': 0}
            return pts_dict

        names, pts = [], []
        for k, v in pts_dict.items():
            if v is not None:
                names.append(k); pts.append([[v[0], v[1]]])
            elif k in self.cache and self.cache[k]['age'] < self.grace:
                names.append(k); pts.append([[self.cache[k]['pt'][0], self.cache[k]['pt'][1]]])

        if not pts:
            self.prev_gray = gray.copy()
            return pts_dict

        p0 = np.float32(pts)
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, p0, None,
            winSize=(21, 21), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )
        out = dict(pts_dict)
        for i, name in enumerate(names):
            if st[i][0] == 1:
                x, y = p1[i][0]
                if pts_dict.get(name) is None:
                    out[name] = (float(x), float(y))
                self.cache[name] = {'pt': (float(x), float(y)), 'age': 0}
            else:
                if name in self.cache:
                    self.cache[name]['age'] += 1
        self.prev_gray = gray.copy()
        return out

# =========================== Helpers ===========================
def laplacian_var(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def tanh(x, k=1.0):
    return float(np.tanh(k * x))

def norm(v, eps=1e-6):
    n = np.linalg.norm(v); return v / (n + eps)

def all_present(*vals):
    return all(v is not None for v in vals)

# Explicit short keys for filter bank (no 'name[:2]' bugs)
FULL2KEY = {
    "LEFT_SHOULDER": "LS", "RIGHT_SHOULDER": "RS",
    "LEFT_ELBOW": "LE", "RIGHT_ELBOW": "RE",
    "LEFT_WRIST": "LW", "RIGHT_WRIST": "RW",
    "LEFT_HIP": "LH", "RIGHT_HIP": "RH",
    "LEFT_ANKLE": "LA", "RIGHT_ANKLE": "RA",
    "NOSE": "NOSE", "LEFT_EAR": "LEAR", "RIGHT_EAR": "REAR",
}

def start_ffmpeg_writer(out_path, w, h, fps):
    cmd = (
        f'ffmpeg -y -f rawvideo -vcodec rawvideo -pix_fmt bgr24 '
        f'-s {w}x{h} -r {fps:.02f} -i - '
        f'-an -vcodec libx264 -preset veryfast -crf 20 -pix_fmt yuv420p "{out_path}"'
    )
    return subprocess.Popen(shlex.split(cmd), stdin=subprocess.PIPE)

# =========================== Stance Logic ===========================
def estimate_forward(LHIP, RHIP, LSH, RSH):
    mid_hip = 0.5 * (LHIP[:3] + RHIP[:3])
    mid_sh  = 0.5 * (LSH[:3] + RSH[:3])
    lateral = norm(LSH[:3] - RSH[:3])
    up = norm(mid_sh - mid_hip)
    fwd = norm(np.cross(lateral, up))
    return mid_hip, fwd

def depth_front(r, l):
    if r is None or l is None:
        return 0.0, 0.0
    # MediaPipe z: more negative => closer; >0 means RIGHT ahead
    dz = (l[2] - r[2])
    s = tanh(dz, 4.0)
    w = 0.5 * ((r[3] if r is not None else 0) + (l[3] if l is not None else 0))
    return s, w

def proj_front(r, l, origin, fwd):
    if r is None or l is None:
        return 0.0, 0.0
    rp = float(np.dot(r[:3] - origin, fwd))
    lp = float(np.dot(l[:3] - origin, fwd))
    s = tanh(rp - lp, 6.0)   # >0 ⇒ RIGHT ahead
    w = 0.5 * ((r[3] if r is not None else 0) + (l[3] if l is not None else 0))
    return s, w

def head_yaw_score(face_landmarks):
    """Nudge from head orientation using Face Mesh (nose vs ears)."""
    if not face_landmarks:
        return 0.0, 0.0
    lm = face_landmarks[0].landmark
    if len(lm) <= 454:
        return 0.0, 0.0
    idx_nose = 1; idx_le = 234; idx_re = 454
    nose = np.array([lm[idx_nose].x, lm[idx_nose].y], dtype=np.float32)
    le   = np.array([lm[idx_le].x,   lm[idx_le].y  ], dtype=np.float32)
    re   = np.array([lm[idx_re].x,   lm[idx_re].y  ], dtype=np.float32)
    v1 = le - re; v2 = nose - re
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    face_w = max(1e-6, np.linalg.norm((le - re)))
    s = tanh((cross / face_w), 3.0)  # >0 ⇒ facing toward RIGHT ⇒ right likely leading
    w = 1.0
    return s, w

def stance_from_score(score):
    # >0 RIGHT ahead ⇒ SOUTHPAW ; <0 LEFT ahead ⇒ ORTHODOX
    return "SOUTHPAW" if score > 0.02 else ("ORTHODOX" if score < -0.02 else "UNKNOWN")

# =========================== Main ===========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, help="Path to input video")
    ap.add_argument("--webcam", type=int, help="Webcam index, e.g., 0")
    ap.add_argument("--show", action="store_true", help="Show live annotated window")
    ap.add_argument("--save", type=str, default="", help='Optional output path, e.g., "pov_annotated.mp4" (ffmpeg H.264)')
    args = ap.parse_args()

    # Open source (graceful fallbacks)
    if args.video is not None:
        src = args.video
    elif args.webcam is not None:
        src = args.webcam
    else:
        src = "./pov.mp4" if os.path.exists("./pov.mp4") else 0

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open any video source.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Source: {'webcam' if isinstance(src, int) else src} | {W}x{H} @ {fps:.2f} fps")

    # MediaPipe solutions (Pose + FaceMesh) – simpler than Tasks for video
    mp_pose = mp.solutions.pose
    mp_face = mp.solutions.face_mesh
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,       # 0=lite, 1=full, 2=heavy
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )
    face = mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )
    POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS
    PLM = mp_pose.PoseLandmark

    # Filters
    def make_filters(fps_):
        keys = set(FULL2KEY.values())
        return {k: LMFilter(fps_) for k in keys}
    filters = make_filters(fps)

    def get_or_make_filter(key):
        if key not in filters:
            filters[key] = LMFilter(fps)
        return filters[key]

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
        f = get_or_make_filter(key)
        return f.apply(p)

    # Voting & weights
    WIN = 200
    votes = deque(maxlen=WIN)
    wts   = deque(maxlen=WIN)

    # Base weights and blend
    W_HANDS = 0.9
    W_SHOULDER = 1.2
    W_HEAD = 0.5
    FWD_BLEND = 0.35

    # Optical flow carry
    flow = FlowCarry(grace=5)

    # Optional save via ffmpeg pipe
    ff = None
    if args.save:
        ff = start_ffmpeg_writer(args.save, W, H, fps)
        print(f"[INFO] Saving to {args.save} (H.264 via ffmpeg)")

    # Live window
    if args.show:
        cv2.namedWindow("Stance (Upper-Body Robust)", cv2.WINDOW_NORMAL)
    delay = max(1, int(1000 / (fps if fps > 0 else 30)))

    # --- NO-SKIP Blur Quality (Option B: quality-weighted) ---
    BLUR_FLOOR = 5.0        # much lower floor; typical phone clips will pass
    BLUR_RATIO = 0.10       # softer adaptive threshold
    blur_hist = deque(maxlen=120)

    frame_idx = 0
    t0 = time.time()

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            print("[INFO] End of stream.")
            break
        frame_idx += 1

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        lv = laplacian_var(gray)
        blur_hist.append(lv)
        med = np.median(blur_hist) if blur_hist else lv
        dyn_thresh = max(BLUR_FLOOR, BLUR_RATIO * (med if med > 1e-6 else lv))
        # Logistic quality score in [0,1] (0=badly blurred)
        quality = 1.0 / (1.0 + np.exp(-(lv - dyn_thresh) / 10.0))
        # Visual hint; DO NOT skip
        if quality < 0.35:
            cv2.putText(frame_bgr, f"LOW QUALITY (q={quality:.2f})", (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pose_res = pose.process(frame_rgb)
        face_res = face.process(frame_rgb)

        annotated = frame_bgr.copy()

        score_total = 0.0
        weight_total = 0.0

        if pose_res.pose_landmarks:
            plms = pose_res.pose_landmarks

            LSH = LM(plms, "LEFT_SHOULDER", 0.30);  RSH = LM(plms, "RIGHT_SHOULDER", 0.30)
            LE  = LM(plms, "LEFT_ELBOW",    0.20);  RE  = LM(plms, "RIGHT_ELBOW",    0.20)
            LW  = LM(plms, "LEFT_WRIST",    0.20);  RW  = LM(plms, "RIGHT_WRIST",    0.20)
            LH  = LM(plms, "LEFT_HIP",      0.20);  RH  = LM(plms, "RIGHT_HIP",      0.20)

            # Forward axis (if torso visible)
            if all_present(LSH, RSH, LH, RH):
                pelvis, fwd = estimate_forward(LH, RH, LSH, RSH)
            else:
                pelvis, fwd = np.array([0, 0, 0], dtype=np.float32), np.array([0, 0, 1], dtype=np.float32)

            # Quality-adjusted effective weights
            W_SHOULDER_eff = W_SHOULDER * (0.6 + 0.4 * quality)   # 0.6..1.0
            W_HANDS_eff    = W_HANDS    * (0.25 + 0.75 * quality) # 0.25..1.0
            W_HEAD_eff     = W_HEAD     * (0.4 + 0.6 * quality)   # 0.4..1.0

            # 1) Shoulder depth (robust under glove blur)
            s_sd, w_sd = depth_front(RSH, LSH)  # >0 RIGHT ahead
            score_total  += W_SHOULDER_eff * s_sd
            weight_total += W_SHOULDER_eff * w_sd

            # 2) Forearm/frontness: prefer wrists; fallback to elbows
            Hh, Ww = H, W
            pts_px = {}
            for tag, P in [("RW", RW), ("LW", LW), ("RE", RE), ("LE", LE), ("RS", RSH), ("LS", LSH)]:
                if P is not None:
                    x = int(P[0] * Ww); y = int(P[1] * Hh)
                    pts_px[tag] = (x, y)
            _ = flow.update(gray, pts_px)  # currently just maintains continuity

            def limb_pair(primary_R, primary_L, fallback_R, fallback_L):
                R = primary_R if (primary_R is not None and primary_R[3] > 0.35) else fallback_R
                L = primary_L if (primary_L is not None and primary_L[3] > 0.35) else fallback_L
                return R, L

            Rf, Lf = limb_pair(RW, LW, RE, LE)
            s_fd, w_fd = depth_front(Rf, Lf)
            s_fp, w_fp = proj_front(Rf, Lf, pelvis, fwd)
            s_fore = (1.0 - FWD_BLEND) * s_fd + FWD_BLEND * s_fp
            w_fore = (w_fd + w_fp)
            score_total  += W_HANDS_eff * s_fore
            weight_total += W_HANDS_eff * w_fore

            # 3) Head yaw nudge (Face Mesh)
            if face_res.multi_face_landmarks:
                s_head, w_head = head_yaw_score(face_res.multi_face_landmarks)
                score_total  += W_HEAD_eff * s_head
                weight_total += W_HEAD_eff * w_head

            # Minimal upper-body overlay
            def draw_dot(pt, col):
                if pt is None: return
                cv2.circle(annotated, (int(pt[0] * Ww), int(pt[1] * Hh)), 4, col, -1)
            def draw_line(a, b, col):
                if a is None or b is None: return
                cv2.line(annotated, (int(a[0] * Ww), int(a[1] * Hh)),
                                   (int(b[0] * Ww), int(b[1] * Hh)), col, 2)

            draw_line(LSH, RSH, (0, 255, 0))
            draw_line(LSH, LE,  (0, 255, 255)); draw_line(LE,  LW, (0, 255, 255))
            draw_line(RSH, RE,  (0, 255, 255)); draw_line(RE,  RW, (0, 255, 255))
            for p in [LSH, RSH, LE, RE, LW, RW]: draw_dot(p, (255, 255, 0))

        # Accumulate stance vote
        if weight_total > 0:
            s = score_total / max(1e-6, weight_total)
            votes.append(s); wts.append(weight_total)
            stance_val = np.average(np.clip(votes, -1, 1), weights=wts)
            stance = stance_from_score(stance_val)
        else:
            stance = "UNKNOWN"; stance_val = 0.0

        # HUD
        cv2.putText(annotated, f"STANCE: {stance}", (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(annotated, f"score={stance_val:.2f}", (12, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Show / Save
        if args.show:
            cv2.imshow("Stance (Upper-Body Robust)", annotated)
            if (cv2.waitKey(delay) & 0xFF) in (27, ord('q')):
                break
        if ff:
            ff.stdin.write(annotated.tobytes())

        if frame_idx % 60 == 0:
            elapsed = time.time() - t0
            print(f"[INFO] processed {frame_idx} frames ({frame_idx/elapsed:.1f} FPS)")

    cap.release()
    if ff:
        try:
            ff.stdin.close()
            ff.wait()
        except Exception:
            pass
        print(f"[INFO] Saved: {args.save}")
    cv2.destroyAllWindows()
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
