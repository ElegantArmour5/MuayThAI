# torso_side_metrics_report.py
import collections, collections.abc
for _n in ("Mapping","MutableMapping","Sequence"):
    if not hasattr(collections,_n): setattr(collections,_n,getattr(collections.abc,_n))

import cv2, argparse, math, csv, os, io, base64
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

try:
    import mediapipe as mp
except Exception as e:
    raise SystemExit("Please install mediapipe: pip install mediapipe==0.10.*") from e

POSE_LANDMARKS = mp.solutions.pose.PoseLandmark

def to_xy(lm, w, h):
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)

def angle_deg(a, b, c):
    ba = a - b; bc = c - b
    na = np.linalg.norm(ba) + 1e-9; nc = np.linalg.norm(bc) + 1e-9
    cosang = np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def line_point_distance(a, b, p):
    ab = b - a
    if np.allclose(ab, 0): return float(np.linalg.norm(p - a))
    return float(np.abs(np.cross(ab, p - a)) / (np.linalg.norm(ab) + 1e-9))

def vec_angle_deg(u, v):
    nu = np.linalg.norm(u) + 1e-9; nv = np.linalg.norm(v) + 1e-9
    cu = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
    return float(np.degrees(np.arccos(cu)))

def angle_vs_vertical(u):
    return vec_angle_deg(u, np.array([0.0, 1.0], dtype=np.float32))

def ema(prev, x, alpha): return alpha * x + (1 - alpha) * prev

@dataclass
class FrameMetrics:
    frame: int
    tsec: float
    S: float
    elbow_L: float
    elbow_R: float
    wristflex_L: float
    wristflex_R: float
    line_off_L: float
    line_off_R: float
    wristdrop_L: float
    wristdrop_R: float
    chintuck_deg: float
    torso_tilt_deg: float
    reach_L: float
    reach_R: float
    guard_ok_L: int
    guard_ok_R: int
    punch_arm: str  # 'L' or 'R'

THRESH = {
    "elbow_min": 165.0, "elbow_max": 175.0,
    "line_off_S": 0.06,
    "wrist_drop_S": 0.20,
    "wrist_flex_abs": 12.0,
    "chin_min": 10.0, "chin_max": 20.0,
    "torso_tilt_max": 10.0,
    "guard_eye_S": 0.05,
    "guard_horiz_S": 0.20,
    "impact_near_elbow": 5.0,
}

def compute_metrics(landmarks, w, h, frame_idx, fps) -> Optional[FrameMetrics]:
    get = lambda idx: landmarks[idx.value]
    try:
        LSH, RSH = to_xy(get(POSE_LANDMARKS.LEFT_SHOULDER), w, h), to_xy(get(POSE_LANDMARKS.RIGHT_SHOULDER), w, h)
        LEL, REL = to_xy(get(POSE_LANDMARKS.LEFT_ELBOW), w, h), to_xy(get(POSE_LANDMARKS.RIGHT_ELBOW), w, h)
        LWR, RWR = to_xy(get(POSE_LANDMARKS.LEFT_WRIST), w, h), to_xy(get(POSE_LANDMARKS.RIGHT_WRIST), w, h)
        LIN, RIN = to_xy(get(POSE_LANDMARKS.LEFT_INDEX), w, h), to_xy(get(POSE_LANDMARKS.RIGHT_INDEX), w, h)
        LEAR, REAR = to_xy(get(POSE_LANDMARKS.LEFT_EAR), w, h), to_xy(get(POSE_LANDMARKS.RIGHT_EAR), w, h)
        LEYE, REYE = to_xy(get(POSE_LANDMARKS.LEFT_EYE), w, h), to_xy(get(POSE_LANDMARKS.RIGHT_EYE), w, h)
        NOSE = to_xy(get(POSE_LANDMARKS.NOSE), w, h)
    except Exception:
        return None

    S = float(np.linalg.norm(LSH - RSH) + 1e-9)
    if S < 1e-3: return None

    elbow_L = angle_deg(LSH, LEL, LWR)
    elbow_R = angle_deg(RSH, REL, RWR)

    wristflex_L = vec_angle_deg(LWR - LEL, LIN - LWR)
    wristflex_R = vec_angle_deg(RWR - REL, RIN - RWR)

    line_off_L = line_point_distance(LSH, LWR, LEL) / S
    line_off_R = line_point_distance(RSH, RWR, REL) / S

    wristdrop_L = abs(LWR[1] - LSH[1]) / S
    wristdrop_R = abs(RWR[1] - RSH[1]) / S

    use_left_head = abs(LEAR[0] - NOSE[0]) < abs(REAR[0] - NOSE[0])
    ear, sh = (LEAR, LSH) if use_left_head else (REAR, RSH)
    chintuck_deg = angle_vs_vertical(ear - sh)

    mid_sh = 0.5 * (LSH + RSH)
    torso_tilt_deg = angle_vs_vertical(NOSE - mid_sh)

    reach_L = abs(LWR[0] - LSH[0])
    reach_R = abs(RWR[0] - RSH[0])

    eye_y = 0.5 * (LEYE[1] + REYE[1])
    horiz_L = abs(LWR[0] - LEAR[0])
    horiz_R = abs(RWR[0] - REAR[0])
    guard_ok_L = int((LWR[1] <= eye_y + THRESH["guard_eye_S"] * S) and (horiz_L <= THRESH["guard_horiz_S"] * S))
    guard_ok_R = int((RWR[1] <= eye_y + THRESH["guard_eye_S"] * S) and (horiz_R <= THRESH["guard_horiz_S"] * S))

    punch_arm = 'L' if reach_L >= reach_R else 'R'

    return FrameMetrics(
        frame=frame_idx,
        tsec=frame_idx / max(fps, 1e-6),
        S=S,
        elbow_L=elbow_L, elbow_R=elbow_R,
        wristflex_L=wristflex_L, wristflex_R=wristflex_R,
        line_off_L=line_off_L, line_off_R=line_off_R,
        wristdrop_L=wristdrop_L, wristdrop_R=wristdrop_R,
        chintuck_deg=chintuck_deg,
        torso_tilt_deg=torso_tilt_deg,
        reach_L=reach_L, reach_R=reach_R,
        guard_ok_L=guard_ok_L, guard_ok_R=guard_ok_R,
        punch_arm=punch_arm
    )

def detect_impacts(reach_hist: List[float], elbow_hist: List[float], fps: float) -> List[int]:
    impacts = []
    N = len(reach_hist)
    if N < 5: return impacts
    reach = np.asarray(reach_hist, dtype=np.float32)
    elbow = np.asarray(elbow_hist, dtype=np.float32)
    vel = np.gradient(reach)
    for i in range(2, N-2):
        is_peak = (reach[i-1] < reach[i] >= reach[i+1]) and (vel[i-1] > 0 >= vel[i+1])
        elbow_ok = abs(elbow[i] - 170.0) <= THRESH["impact_near_elbow"]
        if is_peak and elbow_ok:
            impacts.append(i)
    min_sep = max(2, int(0.12 * fps))
    dedup = []
    for idx in impacts:
        if not dedup or idx - dedup[-1] >= min_sep:
            dedup.append(idx)
    return dedup

def b64_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def plot_series(t, y, title, xlabel="Time (s)", ylabel=""):
    fig = plt.figure(figsize=(8,3))
    plt.plot(t, y)
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.grid(True, linewidth=0.4, alpha=0.6)
    return b64_png(fig)

def plot_series_with_impacts(t, y, impacts_idx, title, xlabel="Time (s)", ylabel=""):
    fig = plt.figure(figsize=(8,3))
    plt.plot(t, y)
    for idx in impacts_idx:
        if 0 <= idx < len(t):
            plt.axvline(t[idx], linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.grid(True, linewidth=0.4, alpha=0.6)
    return b64_png(fig)

def pass_badge(ok: bool) -> str:
    return f'<span class="badge {"ok" if ok else "bad"}">{"PASS" if ok else "FAIL"}</span>'

def extract_thumbnails(video_path: str, frame_indices: List[int], thumb_h: int = 220) -> List[str]:
    b64s = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return b64s
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for idx in frame_indices:
        idx = max(0, min(total-1, idx))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok: 
            b64s.append("")
            continue
        h, w = frame.shape[:2]
        new_w = int(w * (thumb_h / h))
        thumb = cv2.resize(frame, (new_w, thumb_h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", thumb)
        if not ok:
            b64s.append("")
            continue
        b64s.append(base64.b64encode(buf).decode("ascii"))
    cap.release()
    return b64s

def write_csvs(out_dir: str, rows: List[FrameMetrics], impacts_L: List[int], impacts_R: List[int]):
    cf = os.path.join(out_dir, "per_frame_metrics.csv")
    with open(cf, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow([
            "frame","tsec","S",
            "elbow_L","elbow_R","wristflex_L","wristflex_R",
            "line_off_L","line_off_R","wristdrop_L","wristdrop_R",
            "chintuck_deg","torso_tilt_deg",
            "reach_L","reach_R","guard_ok_L","guard_ok_R","punch_arm"
        ])
        for r in rows:
            wr.writerow([
                r.frame, f"{r.tsec:.4f}", f"{r.S:.3f}",
                f"{r.elbow_L:.2f}", f"{r.elbow_R:.2f}",
                f"{r.wristflex_L:.2f}", f"{r.wristflex_R:.2f}",
                f"{r.line_off_L:.4f}", f"{r.line_off_R:.4f}",
                f"{r.wristdrop_L:.4f}", f"{r.wristdrop_R:.4f}",
                f"{r.chintuck_deg:.2f}", f"{r.torso_tilt_deg:.2f}",
                f"{r.reach_L:.2f}", f"{r.reach_R:.2f}",
                r.guard_ok_L, r.guard_ok_R, r.punch_arm
            ])

    ci = os.path.join(out_dir, "per_impact_summary.csv")
    with open(ci, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow([
            "arm","impact_frame","tsec",
            "elbow_deg","elbow_ok",
            "line_off_over_S","line_off_ok",
            "wristflex_deg","wristflex_ok",
            "wristdrop_over_S","wristdrop_ok",
            "chintuck_deg","chintuck_ok",
            "torso_tilt_deg","torso_tilt_ok",
            "guard_ok_nonpunch"
        ])
        def eval_impact(idx, arm):
            r = rows[idx]
            elbow = r.elbow_L if arm=='L' else r.elbow_R
            line_off = r.line_off_L if arm=='L' else r.line_off_R
            wristflex = r.wristflex_L if arm=='L' else r.wristflex_R
            wristdrop = r.wristdrop_L if arm=='L' else r.wristdrop_R
            non_punch_guard = r.guard_ok_R if arm=='L' else r.guard_ok_L

            elbow_ok = (THRESH["elbow_min"] <= elbow <= THRESH["elbow_max"])
            line_ok = (line_off <= THRESH["line_off_S"])
            flex_ok = (abs(wristflex) <= THRESH["wrist_flex_abs"])
            drop_ok = (wristdrop <= THRESH["wrist_drop_S"])
            chin_ok = (THRESH["chin_min"] <= r.chintuck_deg <= THRESH["chin_max"])
            torso_ok = (r.torso_tilt_deg <= THRESH["torso_tilt_max"])

            wr.writerow([
                arm, idx, f"{r.tsec:.3f}",
                f"{elbow:.1f}", int(elbow_ok),
                f"{line_off:.3f}", int(line_ok),
                f"{wristflex:.1f}", int(flex_ok),
                f"{wristdrop:.3f}", int(drop_ok),
                f"{r.chintuck_deg:.1f}", int(chin_ok),
                f"{r.torso_tilt_deg:.1f}", int(torso_ok),
                int(non_punch_guard)
            ])

        for idx in impacts_L: eval_impact(idx, 'L')
        for idx in impacts_R: eval_impact(idx, 'R')

def build_html(out_dir: str,
               video_path: str,
               rows: List[FrameMetrics],
               impacts_L: List[int],
               impacts_R: List[int],
               fps: float):

    t = np.array([r.tsec for r in rows])
    elbow_L = np.array([r.elbow_L for r in rows]); elbow_R = np.array([r.elbow_R for r in rows])
    reach_L = np.array([r.reach_L for r in rows]); reach_R = np.array([r.reach_R for r in rows])
    wristflex_L = np.array([r.wristflex_L for r in rows]); wristflex_R = np.array([r.wristflex_R for r in rows])
    line_L = np.array([r.line_off_L for r in rows]); line_R = np.array([r.line_off_R for r in rows])
    drop_L = np.array([r.wristdrop_L for r in rows]); drop_R = np.array([r.wristdrop_R for r in rows])
    chin = np.array([r.chintuck_deg for r in rows]); torso = np.array([r.torso_tilt_deg for r in rows])

    imp_idx_L = impacts_L
    imp_idx_R = impacts_R
    imp_all = sorted(list(set(imp_idx_L + imp_idx_R)))

    elbowL_png = plot_series_with_impacts(t, elbow_L, imp_idx_L, "Elbow Angle (Left) with Impacts", ylabel="deg")
    elbowR_png = plot_series_with_impacts(t, elbow_R, imp_idx_R, "Elbow Angle (Right) with Impacts", ylabel="deg")
    reachL_png = plot_series_with_impacts(t, reach_L, imp_idx_L, "Reach (Left) with Impacts", ylabel="px")
    reachR_png = plot_series_with_impacts(t, reach_R, imp_idx_R, "Reach (Right) with Impacts", ylabel="px")
    wristL_png = plot_series_with_impacts(t, wristflex_L, imp_idx_L, "Wrist Flex (Left) with Impacts", ylabel="deg")
    wristR_png = plot_series_with_impacts(t, wristflex_R, imp_idx_R, "Wrist Flex (Right) with Impacts", ylabel="deg")
    lineL_png  = plot_series_with_impacts(t, line_L, imp_idx_L, "Line Offset / ShoulderSpan (Left)", ylabel="ratio")
    lineR_png  = plot_series_with_impacts(t, line_R, imp_idx_R, "Line Offset / ShoulderSpan (Right)", ylabel="ratio")
    dropL_png  = plot_series_with_impacts(t, drop_L, imp_idx_L, "Wrist Drop / ShoulderSpan (Left)", ylabel="ratio")
    dropR_png  = plot_series_with_impacts(t, drop_R, imp_idx_R, "Wrist Drop / ShoulderSpan (Right)", ylabel="ratio")
    chin_png   = plot_series_with_impacts(t, chin, imp_all, "Chin Tuck (Head Flexion) with Impacts", ylabel="deg")
    torso_png  = plot_series_with_impacts(t, torso, imp_all, "Upper Torso Tilt with Impacts", ylabel="deg")

    thumbs_b64 = extract_thumbnails(video_path, [rows[i].frame for i in imp_all], thumb_h=220)

    per_impact_rows = []
    for i in imp_all:
        r = rows[i]
        arm = 'L' if i in imp_idx_L else 'R'
        elbow = r.elbow_L if arm=='L' else r.elbow_R
        line = r.line_off_L if arm=='L' else r.line_off_R
        flex = r.wristflex_L if arm=='L' else r.wristflex_R
        drop = r.wristdrop_L if arm=='L' else r.wristdrop_R
        guard = r.guard_ok_R if arm=='L' else r.guard_ok_L

        elbow_ok = THRESH["elbow_min"] <= elbow <= THRESH["elbow_max"]
        line_ok  = line <= THRESH["line_off_S"]
        flex_ok  = abs(flex) <= THRESH["wrist_flex_abs"]
        drop_ok  = drop <= THRESH["wrist_drop_S"]
        chin_ok  = THRESH["chin_min"] <= r.chintuck_deg <= THRESH["chin_max"]
        torso_ok = r.torso_tilt_deg <= THRESH["torso_tilt_max"]

        per_impact_rows.append({
            "t": r.tsec, "frame": r.frame, "arm": arm,
            "elbow": elbow, "elbow_ok": elbow_ok,
            "line": line, "line_ok": line_ok,
            "flex": flex, "flex_ok": flex_ok,
            "drop": drop, "drop_ok": drop_ok,
            "chin": r.chintuck_deg, "chin_ok": chin_ok,
            "torso": r.torso_tilt_deg, "torso_ok": torso_ok,
            "guard": guard
        })

    css = """
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 20px; }
      h1, h2 { margin: 0.4em 0; }
      .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
      .card { border: 1px solid #e5e5e5; border-radius: 10px; padding: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
      .badge { padding: 2px 8px; border-radius: 12px; font-size: 12px; margin-left: 6px; }
      .ok  { background: #e6f7ed; color: #18794e; border: 1px solid #b7ebc6; }
      .bad { background: #fdecec; color: #a8071a; border: 1px solid #ffccc7; }
      table { width: 100%; border-collapse: collapse; }
      th, td { padding: 8px 10px; border-bottom: 1px solid #eee; text-align: left; font-size: 14px; }
      .thumb { width: auto; height: 220px; border-radius: 8px; display: block; }
      .kvs { display:flex; gap:18px; flex-wrap:wrap; margin: 10px 0 0 0; }
      .kv { font-size: 13px; color:#333; }
      .muted { color:#666; font-size:13px; }
      .thresholds { font-size: 13px; }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    </style>
    """

    html = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Torso Side Metrics Report</title>{css}</head><body>"]
    html += [
        "<h1>Torso-Up Side-View Jab/Cross Report</h1>",
        f"<p class='muted'>Video: <span class='mono'>{os.path.basename(video_path)}</span> &middot; Frames: {len(rows)} &middot; Duration: {t[-1]:.2f}s &middot; FPS: {fps:.2f}</p>",
        "<div class='thresholds'><strong>Thresholds</strong>: "
        f"Elbow 165–175°, LineOffset/S ≤ {THRESH['line_off_S']}, WristFlex ≤ {THRESH['wrist_flex_abs']}°, WristDrop/S ≤ {THRESH['wrist_drop_S']}, "
        f"Chin 10–20°, TorsoTilt ≤ {THRESH['torso_tilt_max']}°</div><br/>"
    ]

    html += ["<h2>Impact Thumbnails</h2><div class='grid'>"]
    for k, i in enumerate(imp_all):
        r = rows[i]; arm = 'L' if i in imp_idx_L else 'R'
        b64 = thumbs_b64[k] if k < len(thumbs_b64) else ""
        elbow = r.elbow_L if arm=='L' else r.elbow_R
        line  = r.line_off_L if arm=='L' else r.line_off_R
        flex  = r.wristflex_L if arm=='L' else r.wristflex_R
        drop  = r.wristdrop_L if arm=='L' else r.wristdrop_R
        guard = r.guard_ok_R if arm=='L' else r.guard_ok_L
        elbow_ok = THRESH["elbow_min"] <= elbow <= THRESH["elbow_max"]
        line_ok  = line <= THRESH["line_off_S"]
        flex_ok  = abs(flex) <= THRESH["wrist_flex_abs"]
        drop_ok  = drop <= THRESH["wrist_drop_S"]
        chin_ok  = THRESH["chin_min"] <= r.chintuck_deg <= THRESH["chin_max"]
        torso_ok = r.torso_tilt_deg <= THRESH["torso_tilt_max"]

        html += [
            "<div class='card'>",
            f"<div><strong>{'Jab (L)' if arm=='L' else 'Cross (R)'} @ t={r.tsec:.2f}s</strong> &middot; frame {r.frame}</div>",
            f"<img class='thumb' src='data:image/png;base64,{b64}'/>" if b64 else "<div class='muted'>[thumbnail unavailable]</div>",
            "<div class='kvs'>",
            f"<div class='kv'>Elbow {elbow:.1f}° {pass_badge(elbow_ok)}</div>",
            f"<div class='kv'>Line/S {line:.3f} {pass_badge(line_ok)}</div>",
            f"<div class='kv'>WristFlex {flex:.1f}° {pass_badge(flex_ok)}</div>",
            f"<div class='kv'>Drop/S {drop:.3f} {pass_badge(drop_ok)}</div>",
            f"<div class='kv'>Chin {r.chintuck_deg:.1f}° {pass_badge(chin_ok)}</div>",
            f"<div class='kv'>Torso {r.torso_tilt_deg:.1f}° {pass_badge(torso_ok)}</div>",
            f"<div class='kv'>Guard {pass_badge(bool(guard))}</div>",
            "</div></div>"
        ]
    html += ["</div><br/>"]

    html += ["<h2>Metric Timelines</h2>",
             "<div class='grid'>",
             f"<div class='card'><img src='data:image/png;base64,{elbowL_png}'/></div>",
             f"<div class='card'><img src='data:image/png;base64,{elbowR_png}'/></div>",
             f"<div class='card'><img src='data:image/png;base64,{reachL_png}'/></div>",
             f"<div class='card'><img src='data:image/png;base64,{reachR_png}'/></div>",
             f"<div class='card'><img src='data:image/png;base64,{wristL_png}'/></div>",
             f"<div class='card'><img src='data:image/png;base64,{wristR_png}'/></div>",
             f"<div class='card'><img src='data:image/png;base64,{lineL_png}'/></div>",
             f"<div class='card'><img src='data:image/png;base64,{lineR_png}'/></div>",
             f"<div class='card'><img src='data:image/png;base64,{dropL_png}'/></div>",
             f"<div class='card'><img src='data:image/png;base64,{dropR_png}'/></div>",
             f"<div class='card'><img src='data:image/png;base64,{chin_png}'/></div>",
             f"<div class='card'><img src='data:image/png;base64,{torso_png}'/></div>",
             "</div><br/>"]

    html += ["<h2>Per-Impact Table</h2>",
             "<div class='card'>",
             "<table><thead><tr><th>#</th><th>t (s)</th><th>Frame</th><th>Arm</th>"
             "<th>Elbow°</th><th>Line/S</th><th>WristFlex°</th><th>Drop/S</th><th>Chin°</th><th>Torso°</th><th>Guard</th></tr></thead><tbody>"]

    for k, row in enumerate(per_impact_rows, 1):
        html += [f"<tr><td>{k}</td><td>{row['t']:.2f}</td><td>{row['frame']}</td><td>{row['arm']}</td>"
                 f"<td>{row['elbow']:.1f} {pass_badge(row['elbow_ok'])}</td>"
                 f"<td>{row['line']:.3f} {pass_badge(row['line_ok'])}</td>"
                 f"<td>{row['flex']:.1f} {pass_badge(row['flex_ok'])}</td>"
                 f"<td>{row['drop']:.3f} {pass_badge(row['drop_ok'])}</td>"
                 f"<td>{row['chin']:.1f} {pass_badge(row['chin_ok'])}</td>"
                 f"<td>{row['torso']:.1f} {pass_badge(row['torso_ok'])}</td>"
                 f"<td>{pass_badge(bool(row['guard']))}</td></tr>"]

    html += ["</tbody></table></div>",
             "<p class='muted'>Notes: Impacts are detected at local maxima of reach near elbow ≈170° with velocity sign change. Thresholds are Muay-Thai-safe (no hard elbow lock, neutral wrist, moderate chin tuck, low torso lean).</p>",
             "</body></html>"]

    out_html = os.path.join(out_dir, "report.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("".join(html))
    return out_html

def run(video_path: str, out_dir: str, show: bool, max_frames: int):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)
    rows: List[FrameMetrics] = []
    reach_hist_L: List[float] = []; reach_hist_R: List[float] = []
    elbow_hist_L: List[float] = []; elbow_hist_R: List[float] = []

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if max_frames and frame_idx >= max_frames: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks is not None:
            lm = res.pose_landmarks.landmark
            fm = compute_metrics(lm, w, h, frame_idx, fps)
            if fm:
                rows.append(fm)
                reach_hist_L.append(fm.reach_L); reach_hist_R.append(fm.reach_R)
                elbow_hist_L.append(fm.elbow_L); elbow_hist_R.append(fm.elbow_R)

                if show:
                    txt = f"f={fm.frame} t={fm.tsec:.2f}s"
                    cv2.putText(frame, txt, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    arm = fm.punch_arm
                    elbow = fm.elbow_L if arm=='L' else fm.elbow_R
                    line_off = fm.line_off_L if arm=='L' else fm.line_off_R
                    wristflex = fm.wristflex_L if arm=='L' else fm.wristflex_R
                    wristdrop = fm.wristdrop_L if arm=='L' else fm.wristdrop_R
                    guard_ok = fm.guard_ok_R if arm=='L' else fm.guard_ok_L
                    y0 = 50
                    disp = [
                        f"Arm={arm} Elbow={elbow:.1f}° Line/S={line_off:.3f}",
                        f"WristFlex={wristflex:.1f}° Drop/S={wristdrop:.3f}",
                        f"Chin={fm.chintuck_deg:.1f}° Torso={fm.torso_tilt_deg:.1f}° Guard(non-punch)={guard_ok}"
                    ]
                    for i,s in enumerate(disp):
                        cv2.putText(frame, s, (12, y0+22*i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50,220,50), 2)

        if show:
            cv2.imshow("Torso Side Metrics (preview)", frame)
            if cv2.waitKey(1) & 0xFF == 27: break

        frame_idx += 1

    cap.release()
    if show: cv2.destroyAllWindows()
    if not rows: raise SystemExit("No pose detected—check framing/lighting or try model_complexity=2.")

    impacts_L = detect_impacts(reach_hist_L, elbow_hist_L, fps)
    impacts_R = detect_impacts(reach_hist_R, elbow_hist_R, fps)

    write_csvs(out_dir, rows, impacts_L, impacts_R)
    html_path = build_html(out_dir, video_path, rows, impacts_L, impacts_R, fps)

    print(f"[OK] Frames processed: {len(rows)}  FPS≈{fps:.2f}")
    print(f"[OK] Impacts: L={len(impacts_L)}  R={len(impacts_R)}")
    print(f"[OK] Per-frame CSV: {os.path.join(out_dir,'per_frame_metrics.csv')}")
    print(f"[OK] Per-impact CSV: {os.path.join(out_dir,'per_impact_summary.csv')}")
    print(f"[OK] HTML report: {html_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--out_dir", default="out_metrics", help="Output folder")
    ap.add_argument("--show", action="store_true", help="Preview overlay window")
    ap.add_argument("--max_frames", type=int, default=0, help="Process at most N frames (0=all)")
    args = ap.parse_args()
    run(args.video, args.out_dir, args.show, args.max_frames)

if __name__ == "__main__":
    main()
