#!/usr/bin/env python3
# MMA limb proportions report (annotated image, extended limbs, fused lengths,
# reconstructed angles, radar scores, diagnostics). Auto-corrects height units.

import argparse, base64, csv, json, os, inspect
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- height normalization ----------------
def normalize_height_cm(h: Optional[float]) -> Optional[float]:
    if h is None:
        return None
    try:
        h = float(h)
    except:
        return None
    if 1.2 <= h <= 2.5:   # likely meters
        return h * 100.0
    return h

# ---------- utils ----------
def read_measure_csv(path: str) -> Tuple[Dict[str, float], bool, List[str]]:
    expect = {
        "Left Upper Arm","Right Upper Arm","Left Forearm","Right Forearm",
        "Left Thigh","Right Thigh","Left Shank","Right Shank",
        "Shoulder Width","Hip Width","Torso (Shoulder→Hip Center)",
        "ArmExtended_cm_L","ArmExtended_cm_R","LegExtended_cm_L","LegExtended_cm_R",
        "ElbowAngle_deg_L","ElbowAngle_deg_R","KneeAngle_deg_L","KneeAngle_deg_R",
        "KneeAngle_deg_L_recon","KneeAngle_deg_R_recon","ElbowAngle_deg_L_recon","ElbowAngle_deg_R_recon"
    }
    out, present = {}, set()
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    any_cm = any((r.get("length_cm") or "").strip() for r in rows)
    for r in rows:
        seg = (r.get("segment") or "").strip()
        if not seg: continue
        present.add(seg)
        vcm = (r.get("length_cm") or "").strip()
        vpx = (r.get("length_px") or "").strip()
        use = vcm if any_cm and vcm != "" else vpx
        if use != "":
            try: out[seg] = float(use)
            except: pass
    missing = sorted(list(expect - present))
    return out, any_cm, missing

def avg_ignore_none(vals):
    arr = [float(v) for v in vals if v is not None]
    return float(np.mean(arr)) if arr else None

def score_band(x, lo, hi, hard_lo, hard_hi):
    if x is None: return None
    if x <= hard_lo: return 0.0
    if x >= hard_hi: return 100.0
    if x < lo:  return 100.0*(x-hard_lo)/(lo-hard_lo+1e-9)
    if x > hi:  return 100.0*(x-hi)/(hard_hi-hi+1e-9)+100.0
    return 100.0

def fmt_val(x, unit="cm", d=1): return "—" if x is None else (f"{x:.{d}f} {unit}" if unit else f"{x:.{d}f}")
def fmt_ratio(x, d=3):         return "—" if x is None else f"{x:.{d}f}"

def qualitative(s):
    if s is None: return "Unavailable"
    if s>=85: return "Excellent"
    if s>=70: return "Strong"
    if s>=50: return "Good"
    if s>=30: return "Fair"
    return "Needs work"

def to_b64(path):
    if not path or not os.path.exists(path): return None
    with open(path,"rb") as f: return base64.b64encode(f.read()).decode("ascii")

# ---------- radar ----------
def radar_chart_png(metrics: Dict[str, Optional[float]], labels: List[str], out_png: str):
    vals = [metrics.get(k) for k in labels]
    vals = [0.0 if v is None else float(v) for v in vals]
    N = len(vals)
    ang = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    vals += vals[:1]; ang += ang[:1]
    fig = plt.figure(figsize=(4.6,4.6))
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor("#0e1116"); fig.patch.set_facecolor("#0e1116")
    ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
    ax.set_rlabel_position(0); ax.set_ylim(0,100)
    ax.set_yticks([20,40,60,80,100]); ax.set_yticklabels(["20","40","60","80","100"], color="#a9b5d9")
    ax.plot(ang, vals, linewidth=2, color="#66e2d5"); ax.fill(ang, vals, alpha=0.25, color="#66e2d5")
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels, color="#d6e3ff")
    plt.tight_layout(); fig.savefig(out_png, dpi=180, bbox_inches="tight"); plt.close(fig)

# ---------- analysis ----------
def analyze(csv_path: str, height_cm: Optional[float], hand_estimate_cm=19.0) -> Dict:
    cm, used_cm, missing = read_measure_csv(csv_path)
    unit = "cm" if used_cm else "px"

    # Prefer extended lengths if present (pose-invariant)
    arm_ext_L = cm.get("ArmExtended_cm_L"); arm_ext_R = cm.get("ArmExtended_cm_R")
    leg_ext_L = cm.get("LegExtended_cm_L"); leg_ext_R = cm.get("LegExtended_cm_R")

    ua_L = cm.get("Left Upper Arm");  ua_R = cm.get("Right Upper Arm")
    fa_L = cm.get("Left Forearm");    fa_R = cm.get("Right Forearm")
    th_L = cm.get("Left Thigh");      th_R = cm.get("Right Thigh")
    sh_L = cm.get("Left Shank");      sh_R = cm.get("Right Shank")

    arm_len_L = arm_ext_L if arm_ext_L is not None else ((ua_L or 0)+(fa_L or 0) if (ua_L is not None and fa_L is not None) else None)
    arm_len_R = arm_ext_R if arm_ext_R is not None else ((ua_R or 0)+(fa_R or 0) if (ua_R is not None and fa_R is not None) else None)
    leg_len_L = leg_ext_L if leg_ext_L is not None else ((th_L or 0)+(sh_L or 0) if (th_L is not None and sh_L is not None) else None)
    leg_len_R = leg_ext_R if leg_ext_R is not None else ((th_R or 0)+(sh_R or 0) if (th_R is not None and sh_R is not None) else None)

    arm_len_mean = avg_ignore_none([arm_len_L, arm_len_R])
    leg_len_mean = avg_ignore_none([leg_len_L, leg_len_R])

    # Angles (prefer reconstructed if present)
    elbow_L = cm.get("ElbowAngle_deg_L_recon") or cm.get("ElbowAngle_deg_L")
    elbow_R = cm.get("ElbowAngle_deg_R_recon") or cm.get("ElbowAngle_deg_R")
    knee_L  = cm.get("KneeAngle_deg_L_recon")  or cm.get("KneeAngle_deg_L")
    knee_R  = cm.get("KneeAngle_deg_R_recon")  or cm.get("KneeAngle_deg_R")

    shoulder_w = cm.get("Shoulder Width"); hip_w = cm.get("Hip Width")

    # Wingspan proxy uses extended arms
    hands_included = 0
    wingspan_proxy = None
    if shoulder_w is not None:
        span = shoulder_w
        if arm_len_L is not None: span += arm_len_L; hands_included += 1
        if arm_len_R is not None: span += arm_len_R; hands_included += 1
        if hands_included > 0: span += hands_included * hand_estimate_cm
        if hands_included > 0: wingspan_proxy = span

    # Compute ape index ONLY if measurements are in cm and height provided
    ape_index = None
    if used_cm and height_cm and wingspan_proxy:
        ape_index = wingspan_proxy / float(height_cm)

    # Ratios (bone means)
    ua_mean = avg_ignore_none([ua_L, ua_R])
    fa_mean = avg_ignore_none([fa_L, fa_R])
    th_mean = avg_ignore_none([th_L, th_R])
    sh_mean = avg_ignore_none([sh_L, sh_R])

    fa_ua_ratio   = (fa_mean/ua_mean) if (fa_mean and ua_mean) else None
    tib_fem_ratio = (sh_mean/th_mean) if (sh_mean and th_mean) else None
    v_taper       = (shoulder_w/hip_w) if (shoulder_w and hip_w) else None

    # Scoring (friendlier bands)
    s_ape    = score_band(ape_index, 0.98, 1.06, 0.92, 1.12) if (height_cm and ape_index is not None) else None
    s_fa_ua  = score_band(fa_ua_ratio, 0.85, 1.05, 0.70, 1.20)
    s_tib_fem= score_band(tib_fem_ratio, 0.85, 1.00, 0.70, 1.10)
    s_vtaper = score_band(v_taper, 1.25, 1.40, 1.00, 1.50)
    scores = {"Ape index": s_ape, "Forearm/Humerus": s_fa_ua, "Tibia/Femur": s_tib_fem, "V-taper": s_vtaper}
    subs = [v for v in scores.values() if v is not None]; scores["Overall"] = float(np.mean(subs)) if subs else None

    required_upper = ["Left Upper Arm","Right Upper Arm","Left Forearm","Right Forearm","Shoulder Width"]
    missing_upper = [m for m in required_upper if m in missing or cm.get(m) is None]

    return {
        "unit": "cm" if used_cm else "px",
        "used_cm": used_cm,
        "segments": cm,
        "missing": missing, "missing_upper": missing_upper,
        "arm_len_L": arm_len_L, "arm_len_R": arm_len_R, "arm_len_mean": arm_len_mean,
        "leg_len_L": leg_len_L, "leg_len_R": leg_len_R, "leg_len_mean": leg_len_mean,
        "elbow_L": elbow_L, "elbow_R": elbow_R, "knee_L": knee_L, "knee_R": knee_R,
        "shoulder_w": shoulder_w, "hip_w": hip_w,
        "wingspan_proxy": wingspan_proxy, "ape_index": ape_index,
        "fa_ua_ratio": fa_ua_ratio, "tib_fem_ratio": tib_fem_ratio, "v_taper": v_taper,
        "scores": scores, "hands_included": hands_included
    }

# ---------- html ----------
def build_html(out_path: str, title_stem: str, A: Dict,
               radar_png_path: str, annot_image_path: Optional[str],
               height_cm: Optional[float], annot_width_px: int):
    labels = ["Ape index","Forearm/Humerus","Tibia/Femur","V-taper"]
    radar_b64 = to_b64(radar_png_path)
    img_b64   = to_b64(annot_image_path) if annot_image_path else None
    s = A["scores"]
    raw_map = {"Ape index": A["ape_index"], "Forearm/Humerus": A["fa_ua_ratio"], "Tibia/Femur": A["tib_fem_ratio"], "V-taper": A["v_taper"]}
    tooltips = {
        "Ape index": "Target 0.98–1.06 (soft). 0.92–1.12 clipped. (Only shown if heights/lengths are in cm.)",
        "Forearm/Humerus": "Target 0.85–1.05 (soft). 0.70–1.20 clipped.",
        "Tibia/Femur": "Target 0.85–1.00 (soft). 0.70–1.10 clipped.",
        "V-taper": "Target ≥1.25 (soft). 1.00–1.50 clipped."
    }
    def tile(name: str) -> str:
        v = s.get(name); val = "—" if v is None else f"{v:.1f}"
        q = qualitative(v)
        color = "#9fb0d1" if v is None else ("#7bd389" if v>=70 else "#ffd166" if v>=50 else "#ff8f6b")
        raw = "—" if raw_map[name] is None else f"{raw_map[name]:.3f}"
        tip = tooltips[name]
        return f'''<div class="score" title="{tip}">
          <div class="label">{name}</div>
          <div class="val" style="color:{color}">{val}</div>
          <div class="raw">raw: {raw}</div>
          <div class="q">{q}</div>
        </div>'''
    unit = "cm" if A["used_cm"] else "px"
    missing_upper_txt = "None" if len(A["missing_upper"])==0 else ", ".join(A["missing_upper"])
    missing_all_txt   = "None" if len(A["missing"])==0 else ", ".join(A["missing"])

    ws_lines = []
    ws_lines.append(f"Shoulder width: {fmt_val(A['shoulder_w'], unit)}")
    ws_lines.append(f"Arms included: {A['hands_included']} (extended) + hand est per included side")
    if height_cm: ws_lines.append(f"Height (cm): {height_cm:.1f}")
    ws_lines.append(f"Wingspan proxy: {fmt_val(A['wingspan_proxy'], unit)}")
    ws_lines.append(f"Ape index: {fmt_ratio(A['ape_index'])}")
    ws_diag = "\\n".join(ws_lines)

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>MMA Limb Report – {title_stem}</title>
<style>
body {{ font-family: system-ui,-apple-system,Segoe UI,Roboto,sans-serif; background:#0e1116; color:#d6e3ff; margin:18px }}
h1,h2 {{ margin:8px 0 6px 0 }} .muted {{ color:#9fb0d1 }}
.grid {{ display:grid; grid-template-columns: 1.1fr 1fr; gap:16px }}
.card {{ background:#111625; border:1px solid #222a3a; border-radius:14px; padding:12px }}
.scores {{ display:flex; gap:12px; flex-wrap:wrap }}
.score {{ background:#0f1420; border:1px solid #20283a; padding:10px 12px; border-radius:10px; min-width:160px }}
.score .label {{ font-size:12px; color:#a9b5d9 }}
.score .val {{ font-size:24px; font-weight:800 }}
.score .raw {{ font-size:12px; color:#9fb0d1; margin-top:2px }}
.score .q {{ font-size:12px; color:#a9b5d9 }}
.kv td {{ padding:4px 6px; border-bottom:1px solid #1d2433 }}
.footer {{ color:#9fb0d1; font-size:12px; margin-top:12px }}
img.radar {{ width:100%; max-width:520px; border:1px solid #222a3a; border-radius:12px }}
img.annot {{ width:{annot_width_px}px; max-width:100%; border:1px solid #222a3a; border-radius:12px }}
pre.note {{ white-space:pre-wrap; background:#0f1420; border:1px dashed #253049; color:#9fb0d1; padding:10px; border-radius:8px }}
.badge {{ display:inline-block; padding:3px 8px; border-radius:999px; background:#0f1420; border:1px solid #20283a; color:#9fb0d1; font-size:12px }}
</style>
</head><body>
<h1>MMA Limb Proportions Report</h1>
<div class="muted">{title_stem} • Units: {unit}{' • Height: ' + str(height_cm) + ' cm' if height_cm else ''}</div>

<div class="grid">
  <div class="card">
    <h2>Annotated Measurement Image</h2>
    {"<img class='annot' src='data:image/png;base64," + (to_b64(annot_image_path) or "") + "'/>" if annot_image_path and os.path.exists(annot_image_path) else "<div class='muted'>Annotated image not found.</div>"}
  </div>

  <div class="card">
    <h2>Profile Scores</h2>
    <div class="scores">
      {tile('Ape index')}
      {tile('Forearm/Humerus')}
      {tile('Tibia/Femur')}
      {tile('V-taper')}
    </div>
    <div style="margin-top:10px">
      {"<img class='radar' src='data:image/png;base64," + (to_b64(radar_png_path) or "") + "' alt='Radar Chart'/>" if radar_png_path and os.path.exists(radar_png_path) else "<div class='muted'>Radar not available.</div>"}
    </div>
  </div>

  <div class="card">
    <h2>Key Ratios & Measurements</h2>
    <table class="kv">
      <tr><td>Ape index (wingspan ÷ height)</td><td><b>{fmt_ratio(A['ape_index'])}</b></td></tr>
      <tr><td>Forearm / Humerus</td><td><b>{fmt_ratio(A['fa_ua_ratio'])}</b></td></tr>
      <tr><td>Tibia / Femur</td><td><b>{fmt_ratio(A['tib_fem_ratio'])}</b></td></tr>
      <tr><td>V-taper (shoulder ÷ hip width)</td><td><b>{fmt_ratio(A['v_taper'])}</b></td></tr>
      <tr><td>Shoulder width</td><td><b>{fmt_val(A['shoulder_w'], unit)}</b></td></tr>
      <tr><td>Hip width</td><td><b>{fmt_val(A['hip_w'], unit)}</b></td></tr>
      <tr><td>Avg arm length (extended)</td><td><b>{fmt_val(A['arm_len_mean'], unit)}</b></td></tr>
      <tr><td>Avg leg length (extended)</td><td><b>{fmt_val(A['leg_len_mean'], unit)}</b></td></tr>
      <tr><td>Wingspan proxy</td><td><b>{fmt_val(A['wingspan_proxy'], unit)}</b> <span class="badge">uses extended arms + hands</span></td></tr>
    </table>
  </div>

  <div class="card">
    <h2>Angles & Pose (per side)</h2>
    <table class="kv">
      <tr><td>Elbow L / R (recon)</td><td><b>{fmt_ratio(A['elbow_L'], 1)}° / {fmt_ratio(A['elbow_R'], 1)}°</b></td></tr>
      <tr><td>Knee L / R (recon)</td><td><b>{fmt_ratio(A['knee_L'], 1)}° / {fmt_ratio(A['knee_R'], 1)}°</b></td></tr>
      <tr><td>Arm length (extended) L / R</td><td><b>{fmt_val(A['arm_len_L'], unit)} / {fmt_val(A['arm_len_R'], unit)}</b></td></tr>
      <tr><td>Leg length (extended) L / R</td><td><b>{fmt_val(A['leg_len_L'], unit)} / {fmt_val(A['leg_len_R'], unit)}</b></td></tr>
    </table>
  </div>

  <div class="card">
    <h2>Band Legend</h2>
    <pre class="note">Ape index: soft 0.98–1.06, hard 0.92–1.12
Forearm/Humerus: soft 0.85–1.05, hard 0.70–1.20
Tibia/Femur: soft 0.85–1.00, hard 0.70–1.10
V-taper: soft ≥1.25 (≈1.25–1.40), hard 1.00–1.50
Scores map ratio → 0–100 within these bands (tiles show raw ratio too).
Ape index shown only when measurements are in centimeters and a valid height is provided.</pre>
  </div>

  <div class="card">
    <h2>Diagnostics</h2>
    <pre class="note">Wingspan proxy uses: shoulder width + extended arms present + hands
{ws_diag}

Upper-body segments missing: {missing_upper_txt}
All missing segments: {missing_all_txt}</pre>
  </div>
</div>

<div class="footer">Heuristic bands tuned for MMA leverage. Not medical advice or a body-composition assessment.</div>
</body></html>"""
    with open(out_path, "w", encoding="utf-8") as f: f.write(html)

# ---------- build ----------
def build_everything(csv_path: str, out_html: str, height_cm: Optional[float],
                     hand_estimate_cm: float, annot_image_path: Optional[str],
                     annot_width_px: int) -> str:
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    A = analyze(csv_path, height_cm, hand_estimate_cm)
    labels = ["Ape index","Forearm/Humerus","Tibia/Femur","V-taper"]
    radar_png = f"{stem}_mma_radar.png"
    radar_chart_png(A["scores"], labels, radar_png)
    build_html(out_html, stem, A, radar_png, annot_image_path, height_cm, annot_width_px)
    with open(f"{stem}_mma_metrics.json","w",encoding="utf-8") as jf: json.dump(A, jf, indent=2)
    return out_html

# ---------- measurer bridge ----------
def ensure_csv_from_image(image_path: str, height_cm: float,
                          measurer_module: str = "measure_limbs_from_image",
                          head_extra_ratio: float = 0.25,
                          label_scale: float = 0.85) -> Tuple[str, Optional[str]]:
    mod = __import__(measurer_module)
    fn = getattr(mod, "measure_limbs_from_image", None) or getattr(mod, "measure_limbs_on_image", None)
    if fn is None: raise RuntimeError(f"Could not import measure_limbs_on_image from {measurer_module}.py")
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_img, out_csv = f"{stem}_annot.png", f"{stem}_limbs.csv"
    sig = inspect.signature(fn); params = list(sig.parameters.keys())
    if "label_scale" in params and "head_extra_ratio" in params:
        fn(image_path, float(height_cm), out_img, out_csv, False, head_extra_ratio, label_scale)
    elif "head_extra_ratio" in params:
        fn(image_path, float(height_cm), out_img, out_csv, False, head_extra_ratio)
    else:
        fn(image_path, float(height_cm), out_img, out_csv, False)
    if not os.path.exists(out_csv): raise RuntimeError("Measurement CSV was not created.")
    return out_csv, out_img if os.path.exists(out_img) else None

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Generate MMA limb insights report (annotated image embedded; angle-aware; auto-correct height units).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="CSV produced by measure_limbs_from_image.py")
    src.add_argument("--image", help="Image path (will run measure_limbs_from_image first)")
    ap.add_argument("--height-cm", type=float, default=None, help="Height (cm). Needed for ape index; required with --image.")
    ap.add_argument("--head-extra-ratio", type=float, default=0.25, help="Pass-through to measurer (extra crown height fraction).")
    ap.add_argument("--label-scale", type=float, default=0.85, help="Pass-through: multiplies label font on annotated image.")
    ap.add_argument("--hand-estimate-cm", type=float, default=19.0, help="Estimated hand length per included side for wingspan proxy.")
    ap.add_argument("--measurer-module", default="measure_limbs_from_image", help="Module providing measure_limbs_on_image().")
    ap.add_argument("--annot-image", default=None, help="Annotated image path to embed (default: <stem>_annot.png).")
    ap.add_argument("--annot-width-px", type=int, default=520, help="Display width of annotated image in the HTML.")
    ap.add_argument("--out-html", default=None, help="Output HTML path (default: <stem>_mma_report.html)")
    args = ap.parse_args()

    args.height_cm = normalize_height_cm(args.height_cm)
    if args.height_cm and args.height_cm < 50:
        print(f"[WARN] Height looked like meters; interpreted as {args.height_cm:.1f} cm.")

    if args.image:
        if args.height_cm is None: ap.error("--height-cm is required when using --image")
        csv_path, auto_annot = ensure_csv_from_image(args.image, args.height_cm, args.measurer_module,
                                                     args.head_extra_ratio, args.label_scale)
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        out_html = args.out_html or f"{stem}_mma_report.html"
        annot_path = args.annot_image or auto_annot
        path = build_everything(csv_path, out_html, args.height_cm, args.hand_estimate_cm,
                                annot_path, args.annot_width_px)
        print("[OK] Report:", path)
    else:
        csv_path = args.csv
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        out_html = args.out_html or f"{stem}_mma_report.html"
        default_annot = args.annot_image or (f"{stem}_annot.png" if os.path.exists(f"{stem}_annot.png") else None)
        path = build_everything(csv_path, out_html, args.height_cm, args.hand_estimate_cm,
                                default_annot, args.annot_width_px)
        print("[OK] Report:", path)
        if args.height_cm is None:
            print("[NOTE] Ape index requires --height-cm in centimeters; report includes other ratios without it.")

if __name__ == "__main__":
    main()
