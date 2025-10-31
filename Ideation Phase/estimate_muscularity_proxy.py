import argparse
import csv
import os
from typing import Tuple, Optional, Dict, List

import cv2
import numpy as np
import mediapipe as mp


# ------------------------- Drawing helpers -------------------------
def draw_label(img, text, org, font_scale=0.6, color=(0, 230, 255), thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness+3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def to_xy(lm, W, H) -> Tuple[int, int]:
    return int(lm.x * W), int(lm.y * H)

def dist_px(a: Tuple[int,int], b: Tuple[int,int]) -> float:
    return float(np.hypot(a[0]-b[0], a[1]-b[1]))

def unit_perp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    v = b - a
    n = np.linalg.norm(v) + 1e-6
    v = v / n
    p = np.array([-v[1], v[0]], dtype=np.float32)  # rotate 90°
    return p

def clamp_pt(pt, W, H):
    return int(np.clip(pt[0], 0, W-1)), int(np.clip(pt[1], 0, H-1))

# ------------------------- Mask scanning -------------------------
def half_width_from_bone(mask: np.ndarray, a: Tuple[int,int], b: Tuple[int,int],
                         scan_half_len: int = 80, step: int = 1, iso: int = 128) -> Optional[float]:
    """
    Estimate half-thickness around the segment a->b by scanning along the perpendicular
    from the midpoint until mask crosses background on both sides.
    Returns: half-width in pixels (average of left/right), or None.
    """
    H, W = mask.shape[:2]
    mid = np.array([(a[0]+b[0])/2.0, (a[1]+b[1])/2.0], dtype=np.float32)
    p = unit_perp(np.array(a, np.float32), np.array(b, np.float32))

    def scan_dir(sign: float) -> Optional[float]:
        for d in range(0, scan_half_len, step):
            q = mid + sign * d * p
            x, y = clamp_pt(q, W, H)
            if mask[y, x] < iso:  # hit background
                return float(d)
        return None

    left = scan_dir(-1.0)
    right = scan_dir(+1.0)
    if left is None or right is None:
        return None
    return 0.5 * (left + right)

def width_from_bone(mask: np.ndarray, a: Tuple[int,int], b: Tuple[int,int],
                    scan_half_len: int = 80, step: int = 1, iso: int = 128) -> Optional[float]:
    hw = half_width_from_bone(mask, a, b, scan_half_len, step, iso)
    return None if hw is None else 2.0 * hw

# ------------------------- Scaling -------------------------
def estimate_cm_per_px(height_cm: Optional[float], lms, W, H, vis_thr=0.35, head_extra_ratio=0.25) -> Optional[float]:
    """
    Estimate cm/px from total height if provided, using crown-to-sole span in the image.
    """
    if height_cm is None:
        return None

    PL = mp.solutions.pose.PoseLandmark
    head_ids = [PL.NOSE.value, PL.LEFT_EYE.value, PL.RIGHT_EYE.value, PL.LEFT_EAR.value, PL.RIGHT_EAR.value]
    foot_ids = [PL.LEFT_ANKLE.value, PL.RIGHT_ANKLE.value, PL.LEFT_HEEL.value, PL.RIGHT_HEEL.value,
                PL.LEFT_FOOT_INDEX.value, PL.RIGHT_FOOT_INDEX.value]

    head_pts = [lms[i] for i in head_ids if lms[i].visibility >= vis_thr]
    foot_pts = [lms[i] for i in foot_ids if lms[i].visibility >= vis_thr]
    if not head_pts or not foot_pts:
        return None

    head_min_y = min(p.y for p in head_pts)

    proxies = []
    def vis(i): return lms[i].visibility >= vis_thr
    if vis(PL.LEFT_EYE.value) and vis(PL.RIGHT_EYE.value):
        proxies.append(np.hypot(lms[PL.LEFT_EYE.value].x - lms[PL.RIGHT_EYE.value].x,
                                lms[PL.LEFT_EYE.value].y - lms[PL.RIGHT_EYE.value].y))
    if vis(PL.LEFT_EAR.value) and vis(PL.RIGHT_EAR.value):
        proxies.append(np.hypot(lms[PL.LEFT_EAR.value].x - lms[PL.RIGHT_EAR.value].x,
                                lms[PL.LEFT_EAR.value].y - lms[PL.RIGHT_EAR.value].y))
    head_size_norm = float(np.median(proxies)) if proxies else 0.03
    crown_y = head_min_y - head_extra_ratio * head_size_norm
    crown_y = max(0.0, crown_y)

    foot_max_y = max(p.y for p in foot_pts)
    span_px = (foot_max_y - crown_y) * H
    if span_px <= 1: return None

    return height_cm / span_px

# ------------------------- Main -------------------------
def analyze_muscularity_proxy(image_path: str,
                              height_cm: Optional[float],
                              head_extra_ratio: float = 0.25,
                              out_image: Optional[str] = None,
                              out_csv: Optional[str] = None,
                              show: bool = False):
    # Load
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    H, W = img.shape[:2]
    vis = img.copy()

    # Pose
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, model_complexity=2,
                      smooth_landmarks=True, enable_segmentation=False,
                      min_detection_confidence=0.6, min_tracking_confidence=0.5) as pose:
        res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.pose_landmarks:
        raise RuntimeError("Pose not detected. Try a clearer, full-body photo.")

    lms = res.pose_landmarks.landmark
    PL = mp_pose.PoseLandmark

    # Person mask (Selfie Segmentation)
    with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) as seg:
        seg_res = seg.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    mask = (np.clip(seg_res.segmentation_mask, 0, 1) * 255).astype(np.uint8)  # 0..255

    # Key joints
    def L(i): return lms[i]
    def xy(i): return to_xy(lms[i], W, H)

    j = {}
    for name, idx in [
        ("LS", PL.LEFT_SHOULDER.value), ("RS", PL.RIGHT_SHOULDER.value),
        ("LE", PL.LEFT_ELBOW.value),    ("RE", PL.RIGHT_ELBOW.value),
        ("LW", PL.LEFT_WRIST.value),    ("RW", PL.RIGHT_WRIST.value),
        ("LH", PL.LEFT_HIP.value),      ("RH", PL.RIGHT_HIP.value),
        ("LK", PL.LEFT_KNEE.value),     ("RK", PL.RIGHT_KNEE.value),
        ("LA", PL.LEFT_ANKLE.value),    ("RA", PL.RIGHT_ANKLE.value),
    ]:
        j[name] = xy(idx)

    # Scaling
    cm_per_px = estimate_cm_per_px(height_cm, lms, W, H, head_extra_ratio=head_extra_ratio)

    # Linear widths at key segments (thickness proxies)
    # Arms: midpoint of Shoulder-Elbow and Elbow-Wrist (left/right)
    def segment_mid(a: Tuple[int,int], b: Tuple[int,int]) -> Tuple[int,int]:
        return int((a[0]+b[0])/2), int((a[1]+b[1])/2)

    def scan_width_along(a_name, b_name, label, color):
        a, b = j[a_name], j[b_name]
        w_px = width_from_bone(mask, a, b, scan_half_len=int(0.15*(W+H)))
        if w_px is not None:
            # draw normal at midpoint for visualization
            aV = np.array(a, np.float32); bV = np.array(b, np.float32)
            mid = (aV + bV) / 2.0
            p = unit_perp(aV, bV)
            L1 = mid - 0.5 * w_px * p
            L2 = mid + 0.5 * w_px * p
            P1 = clamp_pt(L1, W, H); P2 = clamp_pt(L2, W, H)
            cv2.line(vis, P1, P2, color, 2, cv2.LINE_AA)
            draw_label(vis, f"{label}: {w_px:.1f}px", (int(mid[0])+6, int(mid[1])-6),
                       font_scale=0.55, color=color, thickness=2)
        return w_px

    # Colors (BGR)
    C_ARM = (86, 220, 255)
    C_FORE = (86, 255, 171)
    C_THIGH = (255, 170, 86)
    C_SHANK = (144, 128, 255)
    C_TORSO = (200, 200, 200)
    C_AXES = (40, 200, 255)

    L_upper_w = scan_width_along("LS", "LE", "Left upper arm", C_ARM)
    L_fore_w  = scan_width_along("LE", "LW", "Left forearm",  C_FORE)
    R_upper_w = scan_width_along("RS", "RE", "Right upper arm", C_ARM)
    R_fore_w  = scan_width_along("RE", "RW", "Right forearm",  C_FORE)

    L_thigh_w = scan_width_along("LH", "LK", "Left thigh", C_THIGH)
    R_thigh_w = scan_width_along("RH", "RK", "Right thigh", C_THIGH)
    L_shank_w = scan_width_along("LK", "LA", "Left shank", C_SHANK)
    R_shank_w = scan_width_along("RK", "RA", "Right shank", C_SHANK)

    # Shoulder width & waist width
    shoulder_width_px = dist_px(j["LS"], j["RS"])

    # Waist: midway between hips, scan perpendicular to pelvis axis
    pelvis_p1, pelvis_p2 = j["LH"], j["RH"]
    pelvis_mid = segment_mid(pelvis_p1, pelvis_p2)
    waist_w = width_from_bone(mask, pelvis_p1, pelvis_p2, scan_half_len=int(0.25*(W+H)))
    # Draw pelvis axis and waist line
    cv2.line(vis, pelvis_p1, pelvis_p2, C_TORSO, 2, cv2.LINE_AA)
    if waist_w is not None:
        p = unit_perp(np.array(pelvis_p1, np.float32), np.array(pelvis_p2, np.float32))
        L1 = np.array(pelvis_mid, np.float32) - 0.5 * waist_w * p
        L2 = np.array(pelvis_mid, np.float32) + 0.5 * waist_w * p
        cv2.line(vis, clamp_pt(L1, W, H), clamp_pt(L2, W, H), C_TORSO, 2, cv2.LINE_AA)
        draw_label(vis, f"Waist width: {waist_w:.1f}px", (pelvis_mid[0]+6, pelvis_mid[1]-8), 0.6, C_TORSO, 2)

    # Shoulder line label
    cv2.line(vis, j["LS"], j["RS"], C_AXES, 2, cv2.LINE_AA)
    draw_label(vis, f"Shoulder width: {shoulder_width_px:.1f}px", (j["RS"][0]+6, j["RS"][1]-8), 0.6, C_AXES, 2)

    # Normalize features
    # Prefer height-based normalization if cm_per_px is available; else use overall body height in px as scale.
    body_height_px = None
    cm_per_px = estimate_cm_per_px(height_cm, lms, W, H, head_extra_ratio=head_extra_ratio)
    if cm_per_px is not None:
        scale = cm_per_px
        scale_name = "cm"
    else:
        # approximate body span in px for normalization
        # Use crown-to-sole span (recompute in px even without height_cm)
        cm_tmp = estimate_cm_per_px(180.0, lms, W, H, head_extra_ratio=head_extra_ratio)  # dummy height
        if cm_tmp is not None:
            body_height_px = 180.0 / cm_tmp
        else:
            # fallback: shoulder->ankle vertical span
            sy = (j["LS"][1] + j["RS"][1]) * 0.5
            ay = max(j["LA"][1], j["RA"][1])
            body_height_px = max(1.0, ay - sy)
        scale = 1.0  # we’ll keep values in px and normalize by body_height_px
        scale_name = "px"

    def norm_feature(px_val: Optional[float]) -> Optional[float]:
        if px_val is None:
            return None
        if cm_per_px is not None:
            return px_val * cm_per_px  # -> cm
        else:
            return px_val / max(1.0, body_height_px)  # unitless fraction of body height

    feats = {
        "shoulder_width": norm_feature(shoulder_width_px),
        "waist_width": norm_feature(waist_w) if waist_w is not None else None,
        "upper_arm_L": norm_feature(L_upper_w),
        "upper_arm_R": norm_feature(R_upper_w),
        "forearm_L": norm_feature(L_fore_w),
        "forearm_R": norm_feature(R_fore_w),
        "thigh_L": norm_feature(L_thigh_w),
        "thigh_R": norm_feature(R_thigh_w),
        "shank_L": norm_feature(L_shank_w),
        "shank_R": norm_feature(R_shank_w),
    }

    # Derived ratios
    sw = feats["shoulder_width"]
    ww = feats["waist_width"]
    shoulder_to_waist = (sw / ww) if (sw is not None and ww not in (None, 0)) else None

    arm_avg = np.nanmean([x for x in [feats["upper_arm_L"], feats["upper_arm_R"]] if x is not None]) if any([feats["upper_arm_L"], feats["upper_arm_R"]]) else None
    thigh_avg = np.nanmean([x for x in [feats["thigh_L"], feats["thigh_R"]] if x is not None]) if any([feats["thigh_L"], feats["thigh_R"]]) else None

    # Muscularity Index (proxy): purely heuristic, 0–100 scale.
    # - higher shoulder/waist and thicker arms/thighs raise the score
    # - coefficients tuned so values roughly span 10–90 for typical photos
    def z(x, mu, sigma):  # clamp z-score-ish
        return 0.0 if x is None else float(np.clip((x - mu) / (sigma + 1e-6), -2.5, 2.5))

    if cm_per_px is not None:
        # Features in cm (very rough; clothing bias!)
        score = 50 \
                + 12*z(shoulder_to_waist, 1.25, 0.25) \
                + 10*z(arm_avg, 9.0, 2.0) \
                + 10*z(thigh_avg, 14.0, 3.0)
    else:
        # Unitless fractions of body height
        score = 50 \
                + 12*z(shoulder_to_waist, 1.25, 0.25) \
                + 10*z(arm_avg, 0.06, 0.02) \
                + 10*z(thigh_avg, 0.10, 0.03)

    score = float(np.clip(score, 0.0, 100.0))

    # Annotate score & disclaimer
    draw_label(vis, f"Muscularity Index (proxy): {score:.1f} / 100",
               (12, 28), 0.8, (86, 255, 171), 2)
    disclaimer = "Not body fat or muscle % — silhouette-based proxy; clothing can bias measurements."
    draw_label(vis, disclaimer, (12, 56), 0.55, (200, 200, 200), 2)

    # Outputs
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_image = out_image or f"{stem}_muscularity_proxy.png"
    out_csv   = out_csv   or f"{stem}_muscularity_proxy.csv"

    cv2.imwrite(out_image, vis)

    # CSV: include both raw px widths and normalized features/ratios
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        cols = [
            "metric", f"value_{scale_name}",
            "shoulder_to_waist_ratio",
            "muscularity_index_proxy_0_100",
            "cm_per_px"
        ]
        writer = csv.writer(f)
        writer.writerow(cols)

        def row(name, px_val, feat_val):
            writer.writerow([name,
                             f"{feat_val:.4f}" if feat_val is not None else "",
                             f"{shoulder_to_waist:.4f}" if shoulder_to_waist is not None else "",
                             f"{score:.2f}",
                             f"{cm_per_px:.6f}" if cm_per_px is not None else ""])

        row("shoulder_width", shoulder_width_px, feats["shoulder_width"])
        row("waist_width", waist_w, feats["waist_width"])
        row("upper_arm_L", L_upper_w, feats["upper_arm_L"])
        row("upper_arm_R", R_upper_w, feats["upper_arm_R"])
        row("forearm_L", L_fore_w, feats["forearm_L"])
        row("forearm_R", R_fore_w, feats["forearm_R"])
        row("thigh_L", L_thigh_w, feats["thigh_L"])
        row("thigh_R", R_thigh_w, feats["thigh_R"])
        row("shank_L", L_shank_w, feats["shank_L"])
        row("shank_R", R_shank_w, feats["shank_R"])

    print(f"[OK] Saved annotated image: {out_image}")
    print(f"[OK] Saved CSV: {out_csv}")
    print("[NOTE] Muscularity Index is a visual proxy only — not medical advice, not body fat/muscle percentage.")

# ------------------------- CLI -------------------------
def main():
    ap = argparse.ArgumentParser(description="Estimate a visual muscularity proxy from a single (clothed) image.")
    ap.add_argument("--image", required=True, help="Input image path")
    ap.add_argument("--height-cm", type=float, default=None,
                    help="Optional person height (cm). If provided, features are in cm; otherwise normalized by body height in pixels.")
    ap.add_argument("--head-extra-ratio", type=float, default=0.25,
                    help="Head crown adjustment fraction (default 0.25)")
    ap.add_argument("--out-image", default=None, help="Output annotated image path")
    ap.add_argument("--out-csv", default=None, help="Output CSV path")
    ap.add_argument("--show", action="store_true", help="Display the annotated image")
    args = ap.parse_args()

    analyze_muscularity_proxy(args.image, args.height_cm, args.head_extra_ratio,
                              args.out_image, args.out_csv, args.show)

if __name__ == "__main__":
    main()
