#!/usr/bin/env python3
# Robust limb lengths (2D/3D fusion) + reconstructed joint angles from a single image.
# Saves annotated image and CSV. Auto-corrects height if passed in meters.

import argparse, math, csv, os
from typing import Optional, Tuple, Dict
import cv2, numpy as np, mediapipe as mp

PL = mp.solutions.pose.PoseLandmark

# ---------------- height normalization ----------------
def normalize_height_cm(h: Optional[float]) -> Optional[float]:
    """Accept height in cm, auto-fix common mistake of meters (e.g., 1.75 -> 175.0)."""
    if h is None:
        return None
    try:
        h = float(h)
    except:
        return None
    if 1.2 <= h <= 2.5:   # likely meters
        return h * 100.0
    return h

# ---------------- basic helpers ----------------
def _l2(p, q): return float(math.hypot(p[0]-q[0], p[1]-q[1]))
def _w2(a, b):
    dx, dy, dz = (a.x-b.x), (a.y-b.y), (a.z-b.z)
    return float(math.sqrt(dx*dx + dy*dy + dz*dz))
def _xy(lm, W, H): return (lm.x * W, lm.y * H)
def _safe(val): return val if val is not None and np.isfinite(val) else None

def _joint_angle_deg_3d(a, b, c) -> Optional[float]:
    if a is None or b is None or c is None: return None
    v1 = np.array([a.x-b.x, a.y-b.y, a.z-b.z], float)
    v2 = np.array([c.x-b.x, c.y-b.y, c.z-b.z], float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return None
    cosang = np.clip(np.dot(v1, v2)/(n1*n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def _cm_per_px_from_height(height_cm: Optional[float], img_lms, W, H,
                           head_extra_ratio=0.25, vis_thr=0.35) -> Optional[float]:
    if height_cm is None or img_lms is None: return None
    head_ids = [PL.NOSE.value, PL.LEFT_EYE.value, PL.RIGHT_EYE.value, PL.LEFT_EAR.value, PL.RIGHT_EAR.value]
    foot_ids = [PL.LEFT_ANKLE.value, PL.RIGHT_ANKLE.value, PL.LEFT_FOOT_INDEX.value, PL.RIGHT_FOOT_INDEX.value]
    head_pts = [img_lms[i] for i in head_ids if img_lms[i].visibility >= vis_thr]
    foot_pts = [img_lms[i] for i in foot_ids if img_lms[i].visibility >= vis_thr]
    if not head_pts or not foot_pts: return None
    head_min_y = min(p.y for p in head_pts)
    proxies = []
    if img_lms[PL.LEFT_EYE.value].visibility >= vis_thr and img_lms[PL.RIGHT_EYE.value].visibility >= vis_thr:
        proxies.append(abs(img_lms[PL.LEFT_EYE.value].x - img_lms[PL.RIGHT_EYE.value].x))
    if img_lms[PL.LEFT_EAR.value].visibility >= vis_thr and img_lms[PL.RIGHT_EAR.value].visibility >= vis_thr:
        proxies.append(abs(img_lms[PL.LEFT_EAR.value].x - img_lms[PL.RIGHT_EAR.value].x))
    head_norm = float(np.median(proxies)) if proxies else 0.03
    crown_y = max(0.0, head_min_y - head_extra_ratio * head_norm)
    foot_max_y = max(p.y for p in foot_pts)
    span_px = (foot_max_y - crown_y) * H
    return None if span_px <= 1 else (height_cm / span_px)

def _world_total_height_m(world_lms) -> Optional[float]:
    if world_lms is None: return None
    try:
        sh_y = min(world_lms[PL.LEFT_SHOULDER.value].y, world_lms[PL.RIGHT_SHOULDER.value].y)
        an_y = max(world_lms[PL.LEFT_ANKLE.value].y, world_lms[PL.RIGHT_ANKLE.value].y)
        sh_w = _w2(world_lms[PL.LEFT_SHOULDER.value], world_lms[PL.RIGHT_SHOULDER.value])
        span_m = (an_y - sh_y) + 0.25 * sh_w
        return abs(span_m)
    except Exception:
        return None

def _robust_segment_len(img_a, img_b, world_a, world_b, W, H,
                        cm_per_px: Optional[float],
                        world_to_cm: Optional[float],
                        mode="blend_max") -> Dict[str, Optional[float]]:
    len_px_2d = None
    if img_a is not None and img_b is not None:
        len_px_2d = _l2(_xy(img_a, W, H), _xy(img_b, W, H))
    cm_2d = (len_px_2d * cm_per_px) if (len_px_2d is not None and cm_per_px is not None) else None
    len_m_world = None
    if world_a is not None and world_b is not None:
        len_m_world = _w2(world_a, world_b)
    cm_world = (len_m_world * world_to_cm) if (len_m_world is not None and world_to_cm is not None) else None

    if mode == "2d": chosen = cm_2d
    elif mode == "world": chosen = cm_world
    elif mode == "blend_mean":
        chosen = 0.5*(cm_2d+cm_world) if (cm_2d is not None and cm_world is not None) else (cm_2d if cm_world is None else cm_world)
    else:  # blend_max
        chosen = max(cm_2d, cm_world) if (cm_2d is not None and cm_world is not None) else (cm_2d if cm_world is None else cm_world)

    return dict(len_px_2d=_safe(len_px_2d), cm_2d=_safe(cm_2d),
                len_m_world=_safe(len_m_world), cm_world=_safe(cm_world),
                chosen_cm=_safe(chosen))

# ---------------- priors & fusion ----------------
def _anthro_prior_cm(height_cm: float, segment: str) -> Optional[float]:
    """Forgiving priors (fractions of height) for combat athletes."""
    if height_cm is None: return None
    seg = segment.lower()
    if "upper arm" in seg: return 0.19 * height_cm
    if "forearm"   in seg: return 0.165 * height_cm
    if "thigh"     in seg: return 0.245 * height_cm
    if "shank"     in seg: return 0.246 * height_cm
    if "shoulder width" in seg: return 0.26 * height_cm
    if "hip width"       in seg: return 0.20 * height_cm
    return None

def _huber_pull(x, mu, delta=6.0):
    if x is None or mu is None: return 0.0
    r = abs(x - mu)
    return 1.0 / (1.0 + (r / max(delta, 1e-6)))

def _weighted_median(vals, weights):
    arr = [(v, w) for v, w in zip(vals, weights) if v is not None and w > 0]
    if not arr: return None
    arr.sort(key=lambda t: t[0])
    total = sum(w for _, w in arr)
    acc = 0.0
    for v, w in arr:
        acc += w
        if acc >= 0.5 * total:
            return v
    return arr[-1][0]

def _fuse_length(height_cm: float,
                 seg_name: str,
                 primary: Optional[float],
                 mirror: Optional[float],
                 base2d: Optional[float],
                 base3d: Optional[float]) -> Optional[float]:
    prior = _anthro_prior_cm(height_cm, seg_name)
    w_primary = 1.0
    w_mirror  = 0.8 if mirror is not None else 0.0
    w_prior   = 0.5 * _huber_pull(primary or mirror or 0, prior, delta=8.0) if prior is not None else 0.0
    ref = primary if primary is not None else mirror
    w_2d = 0.3 * (_huber_pull(base2d, ref, delta=10.0) if ref is not None else 0.3)
    w_3d = 0.3 * (_huber_pull(base3d, ref, delta=10.0) if ref is not None else 0.3)
    vals = [primary, mirror, prior, base2d, base3d]
    ws   = [w_primary, w_mirror, w_prior, w_2d, w_3d]
    return _weighted_median(vals, ws)

# ---------------- drawing ----------------
def _draw_label(img, text, org, scale=0.8, color=(255,230,120)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

def _draw_angle_arc(img, center, radius, start_deg, end_deg, color=(140,230,255), thick=2):
    pts = []
    for a in np.linspace(np.radians(start_deg), np.radians(end_deg), 24):
        x = int(center[0] + radius*np.cos(a)); y = int(center[1] + radius*np.sin(a))
        pts.append((x,y))
    for i in range(1, len(pts)):
        cv2.line(img, pts[i-1], pts[i], color, thick, cv2.LINE_AA)

# ---------------- main measurement ----------------
def measure_limbs_from_image(image_path: str,
                             height_cm: float,
                             out_image: str,
                             out_csv: str,
                             show: bool = False,
                             head_extra_ratio: float = 0.25,
                             label_scale: float = 0.85,
                             vis_thr: float = 0.35) -> None:
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None: raise RuntimeError(f"Could not read image: {image_path}")
    H, W = img_bgr.shape[:2]
    annot = img_bgr.copy()
    t = max(2, int(round(0.003 * (W + H))))
    fs = max(0.5, (W + H) / 1800.0) * float(label_scale)

    with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2,
                                enable_segmentation=False, min_detection_confidence=0.5) as pose:
        res = pose.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    img_lms = res.pose_landmarks.landmark if res.pose_landmarks else None
    world_lms = res.pose_world_landmarks.landmark if res.pose_world_landmarks else None

    cm_per_px = _cm_per_px_from_height(height_cm, img_lms, W, H, head_extra_ratio, vis_thr)
    world_to_cm = None
    if world_lms is not None and height_cm:
        tot_h_m = _world_total_height_m(world_lms)
        if tot_h_m and tot_h_m > 1e-6:
            world_to_cm = float(height_cm) / float(tot_h_m)

    writer = csv.DictWriter(open(out_csv, "w", newline="", encoding="utf-8"),
                            fieldnames=["segment","length_px","length_cm","length_cm_2d","length_cm_world","note"])
    writer.writeheader()

    def seg(name, ia, ib, color=(90, 230, 255)):
        img_a = img_lms[ia] if img_lms else None
        img_b = img_lms[ib] if img_lms else None
        w_a   = world_lms[ia] if world_lms else None
        w_b   = world_lms[ib] if world_lms else None
        d = _robust_segment_len(img_a, img_b, w_a, w_b, W, H, cm_per_px, world_to_cm, mode="blend_max")
        writer.writerow({
            "segment": name,
            "length_px": f"{d['len_px_2d']:.2f}" if d["len_px_2d"] is not None else "",
            "length_cm": f"{d['chosen_cm']:.2f}" if d["chosen_cm"] is not None else "",
            "length_cm_2d": f"{d['cm_2d']:.2f}" if d["cm_2d"] is not None else "",
            "length_cm_world": f"{d['cm_world']:.2f}" if d["cm_world"] is not None else "",
            "note": ""
        })
        if img_a is not None and img_b is not None:
            pa = _xy(img_a, W, H); pb = _xy(img_b, W, H)
            cv2.line(annot, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, t, cv2.LINE_AA)
            mid = (int((pa[0]+pb[0])//2), int((pa[1]+pb[1])//2))
            if d["chosen_cm"] is not None:
                _draw_label(annot, f"{name}: {d['chosen_cm']:.1f} cm", (mid[0]+6, mid[1]-6), fs)
        return d  # dict

    # ---- bones (per side) ----
    UA_L = seg("Left Upper Arm",  PL.LEFT_SHOULDER.value, PL.LEFT_ELBOW.value)
    FA_L = seg("Left Forearm",    PL.LEFT_ELBOW.value,    PL.LEFT_WRIST.value)
    UA_R = seg("Right Upper Arm", PL.RIGHT_SHOULDER.value,PL.RIGHT_ELBOW.value)
    FA_R = seg("Right Forearm",   PL.RIGHT_ELBOW.value,   PL.RIGHT_WRIST.value)

    TH_L = seg("Left Thigh",  PL.LEFT_HIP.value,   PL.LEFT_KNEE.value, (90,255,140))
    SH_L = seg("Left Shank",  PL.LEFT_KNEE.value,  PL.LEFT_ANKLE.value,(90,255,140))
    TH_R = seg("Right Thigh", PL.RIGHT_HIP.value,  PL.RIGHT_KNEE.value,(90,255,140))
    SH_R = seg("Right Shank", PL.RIGHT_KNEE.value, PL.RIGHT_ANKLE.value,(90,255,140))

    # ---- fuse lengths (mirror + prior + raw) ----
    def _fuse_pair(name_L, name_R, dL, dR):
        fL = _fuse_length(height_cm, name_L, dL["chosen_cm"], dR["chosen_cm"], dL["cm_2d"], dL["cm_world"])
        fR = _fuse_length(height_cm, name_R, dR["chosen_cm"], dL["chosen_cm"], dR["cm_2d"], dR["cm_world"])
        return fL, fR

    UA_L_f, UA_R_f = _fuse_pair("Left Upper Arm", "Right Upper Arm", UA_L, UA_R)
    FA_L_f, FA_R_f = _fuse_pair("Left Forearm",   "Right Forearm",   FA_L, FA_R)
    TH_L_f, TH_R_f = _fuse_pair("Left Thigh",     "Right Thigh",     TH_L, TH_R)
    SH_L_f, SH_R_f = _fuse_pair("Left Shank",     "Right Shank",     SH_L, SH_R)

    for k, v in [("Left Upper Arm_fused", UA_L_f), ("Right Upper Arm_fused", UA_R_f),
                 ("Left Forearm_fused",   FA_L_f), ("Right Forearm_fused",   FA_R_f),
                 ("Left Thigh_fused",     TH_L_f), ("Right Thigh_fused",     TH_R_f),
                 ("Left Shank_fused",     SH_L_f), ("Right Shank_fused",     SH_R_f)]:
        csv.DictWriter(open(out_csv, "a", newline="", encoding="utf-8"),
                       fieldnames=["segment","length_px","length_cm","length_cm_2d","length_cm_world","note"]).writerow(
            {"segment": k, "length_px":"", "length_cm": f"{v:.2f}" if v is not None else "", "length_cm_2d":"", "length_cm_world":"", "note":"fused"}
        )

    # Extended limbs (pose-invariant)
    ArmExt_L = (UA_L_f or 0) + (FA_L_f or 0) if (UA_L_f is not None and FA_L_f is not None) else None
    ArmExt_R = (UA_R_f or 0) + (FA_R_f or 0) if (UA_R_f is not None and FA_R_f is not None) else None
    LegExt_L = (TH_L_f or 0) + (SH_L_f or 0) if (TH_L_f is not None and SH_L_f is not None) else None
    LegExt_R = (TH_R_f or 0) + (SH_R_f or 0) if (TH_R_f is not None and SH_R_f is not None) else None

    for k, v in [("ArmExtended_cm_L", ArmExt_L), ("ArmExtended_cm_R", ArmExt_R),
                 ("LegExtended_cm_L", LegExt_L), ("LegExtended_cm_R", LegExt_R)]:
        csv.DictWriter(open(out_csv, "a", newline="", encoding="utf-8"),
                       fieldnames=["segment","length_px","length_cm","length_cm_2d","length_cm_world","note"]).writerow(
            {"segment": k, "length_px":"", "length_cm": f"{v:.2f}" if v is not None else "", "length_cm_2d":"", "length_cm_world":"", "note":"extended(sum fused)"}
        )

    # Shoulder / Hip widths
    def pair_len(name, ia, ib, color=(255,210,120)):
        d = seg(name, ia, ib, color)
        return d["chosen_cm"]

    ShoulderW = pair_len("Shoulder Width", PL.LEFT_SHOULDER.value, PL.RIGHT_SHOULDER.value, (255,210,120))
    HipW      = pair_len("Hip Width",      PL.LEFT_HIP.value,      PL.RIGHT_HIP.value,      (255,210,120))

    # Torso (mid-shoulders -> mid-hips), robust (max of 2D->cm, world->cm)
    Torso = None
    if img_lms is not None:
        sh_mid = ((_xy(img_lms[PL.LEFT_SHOULDER.value], W, H)[0] + _xy(img_lms[PL.RIGHT_SHOULDER.value], W, H)[0]) / 2.0,
                  (_xy(img_lms[PL.LEFT_SHOULDER.value], W, H)[1] + _xy(img_lms[PL.RIGHT_SHOULDER.value], W, H)[1]) / 2.0)
        hip_mid = ((_xy(img_lms[PL.LEFT_HIP.value], W, H)[0] + _xy(img_lms[PL.RIGHT_HIP.value], W, H)[0]) / 2.0,
                   (_xy(img_lms[PL.LEFT_HIP.value], W, H)[1] + _xy(img_lms[PL.RIGHT_HIP.value], W, H)[1]) / 2.0)
        torso_2d_cm = (_l2(sh_mid, hip_mid) * cm_per_px) if cm_per_px is not None else None
        torso_world_cm = None
        if world_lms is not None:
            sh_mid_w = ( (world_lms[PL.LEFT_SHOULDER.value].x + world_lms[PL.RIGHT_SHOULDER.value].x)/2.0,
                         (world_lms[PL.LEFT_SHOULDER.value].y + world_lms[PL.RIGHT_SHOULDER.value].y)/2.0,
                         (world_lms[PL.LEFT_SHOULDER.value].z + world_lms[PL.RIGHT_SHOULDER.value].z)/2.0 )
            hip_mid_w = ( (world_lms[PL.LEFT_HIP.value].x + world_lms[PL.RIGHT_HIP.value].x)/2.0,
                          (world_lms[PL.LEFT_HIP.value].y + world_lms[PL.RIGHT_HIP.value].y)/2.0,
                          (world_lms[PL.LEFT_HIP.value].z + world_lms[PL.RIGHT_HIP.value].z)/2.0 )
            torso_m = math.sqrt( (sh_mid_w[0]-hip_mid_w[0])**2 + (sh_mid_w[1]-hip_mid_w[1])**2 + (sh_mid_w[2]-hip_mid_w[2])**2 )
            torso_world_cm = torso_m * world_to_cm if world_to_cm is not None else None
        Torso = max([x for x in [torso_2d_cm, torso_world_cm] if x is not None], default=None)
        csv.DictWriter(open(out_csv, "a", newline="", encoding="utf-8"),
                       fieldnames=["segment","length_px","length_cm","length_cm_2d","length_cm_world","note"]).writerow(
            {"segment":"Torso (Shoulder→Hip Center)","length_px":"","length_cm": f"{Torso:.2f}" if Torso is not None else "","length_cm_2d":"","length_cm_world":"","note":"robust"}
        )
        cv2.line(annot, (int(sh_mid[0]), int(sh_mid[1])), (int(hip_mid[0]), int(hip_mid[1])), (255,210,120), t, cv2.LINE_AA)
        if Torso is not None: _draw_label(annot, f"Torso: {Torso:.1f} cm", (int(sh_mid[0])+6, int(sh_mid[1])-6), fs)

    # ---------- reconstruct angles (law of cosines) ----------
    def chord_cm(ia, ib):
        if img_lms is None or cm_per_px is None: return None
        a = _xy(img_lms[ia], W, H); b = _xy(img_lms[ib], W, H)
        return _l2(a, b) * cm_per_px

    HA_L = chord_cm(PL.LEFT_HIP.value,   PL.LEFT_ANKLE.value)
    HA_R = chord_cm(PL.RIGHT_HIP.value,  PL.RIGHT_ANKLE.value)
    SW_L = chord_cm(PL.LEFT_SHOULDER.value,  PL.LEFT_WRIST.value)
    SW_R = chord_cm(PL.RIGHT_SHOULDER.value, PL.RIGHT_WRIST.value)

    def law_of_cos(a, b, c):
        if a is None or b is None or c is None or a<=0 or b<=0: return None
        denom = max(2*a*b, 1e-6)
        cosx = np.clip((a*a + b*b - c*c)/denom, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosx)))

    Knee_L_recon = law_of_cos(TH_L_f, SH_L_f, HA_L)
    Knee_R_recon = law_of_cos(TH_R_f, SH_R_f, HA_R)
    Elbow_L_recon = law_of_cos(UA_L_f, FA_L_f, SW_L)
    Elbow_R_recon = law_of_cos(UA_R_f, FA_R_f, SW_R)

    def world_ang(ia, ib, ic):
        if world_lms is None: return None
        return _joint_angle_deg_3d(world_lms[ia], world_lms[ib], world_lms[ic])

    Elbow_L_raw = world_ang(PL.LEFT_SHOULDER.value, PL.LEFT_ELBOW.value, PL.LEFT_WRIST.value)
    Elbow_R_raw = world_ang(PL.RIGHT_SHOULDER.value,PL.RIGHT_ELBOW.value, PL.RIGHT_WRIST.value)
    Knee_L_raw  = world_ang(PL.LEFT_HIP.value,      PL.LEFT_KNEE.value,  PL.LEFT_ANKLE.value)
    Knee_R_raw  = world_ang(PL.RIGHT_HIP.value,     PL.RIGHT_KNEE.value, PL.RIGHT_ANKLE.value)

    wr = csv.DictWriter(open(out_csv, "a", newline="", encoding="utf-8"),
                        fieldnames=["segment","length_px","length_cm","length_cm_2d","length_cm_world","note"])
    for name, val, note in [
        ("KneeAngle_deg_L_raw",  Knee_L_raw,  "world"),
        ("KneeAngle_deg_R_raw",  Knee_R_raw,  "world"),
        ("ElbowAngle_deg_L_raw", Elbow_L_raw, "world"),
        ("ElbowAngle_deg_R_raw", Elbow_R_raw, "world"),
        ("KneeAngle_deg_L_recon",  Knee_L_recon,  "law-of-cos"),
        ("KneeAngle_deg_R_recon",  Knee_R_recon,  "law-of-cos"),
        ("ElbowAngle_deg_L_recon", Elbow_L_recon, "law-of-cos"),
        ("ElbowAngle_deg_R_recon", Elbow_R_recon, "law-of-cos"),
    ]:
        wr.writerow({"segment": name, "length_px":"", "length_cm": f"{val:.1f}" if val is not None else "",
                     "length_cm_2d":"", "length_cm_world":"", "note": note})

    # draw reconstructed angles
    if img_lms is not None:
        def draw_recon(px_joint, ang, label):
            if ang is None: return
            x, y = int(px_joint[0]), int(px_joint[1])
            _draw_angle_arc(annot, (x, y), radius=30, start_deg=-60, end_deg=60, color=(140,230,255), thick=2)
            _draw_label(annot, f"{label}{ang:.0f}°", (x+6, y-6), fs*0.8, (200,255,180))
        draw_recon(_xy(img_lms[PL.LEFT_KNEE.value],  W, H), Knee_L_recon,  "Knee L ")
        draw_recon(_xy(img_lms[PL.RIGHT_KNEE.value], W, H), Knee_R_recon,  "Knee R ")
        draw_recon(_xy(img_lms[PL.LEFT_ELBOW.value], W, H), Elbow_L_recon, "Elbow L ")
        draw_recon(_xy(img_lms[PL.RIGHT_ELBOW.value],W, H), Elbow_R_recon, "Elbow R ")

    cv2.imwrite(out_image, annot)
    if show:
        cv2.imshow("Annotated", annot); cv2.waitKey(0); cv2.destroyAllWindows()

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Robust limb lengths + reconstructed angles from a single image (auto-correct height units).")
    ap.add_argument("--image", required=True)
    ap.add_argument("--height-cm", type=float, required=True)
    ap.add_argument("--out-image", default=None)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--head-extra-ratio", type=float, default=0.25)
    ap.add_argument("--label-scale", type=float, default=0.85)
    ap.add_argument("--vis-thr", type=float, default=0.35)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    args.height_cm = normalize_height_cm(args.height_cm)
    if args.height_cm and args.height_cm < 50:
        print(f"[WARN] Height looked like meters; interpreted as {args.height_cm:.1f} cm.")

    stem = os.path.splitext(os.path.basename(args.image))[0]
    out_image = args.out_image or f"{stem}_annot.png"
    out_csv   = args.out_csv   or f"{stem}_limbs.csv"

    measure_limbs_from_image(args.image, args.height_cm, out_image, out_csv,
                             show=args.show,
                             head_extra_ratio=args.head_extra_ratio,
                             label_scale=args.label_scale,
                             vis_thr=args.vis_thr)

if __name__ == "__main__":
    main()
