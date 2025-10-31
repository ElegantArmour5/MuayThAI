# punch_checker_mediapipe.py  (MediaPipe-only)
# --- Optional shim for older protobuf on Python 3.11 ---
import collections, collections.abc
for _n in ("Mapping","MutableMapping","Sequence"):
    if not hasattr(collections,_n): setattr(collections,_n,getattr(collections.abc,_n))

import cv2, argparse, math, time, os, pathlib, csv
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

# ---------------- Tunables ----------------
ELBOW_MIN_STRAIGHT = 160.0     # deg
HORIZ_MIN_RATIO    = 1.6       # |vx|/|vy|
WRIST_MAX_DROP_PX  = 60        # px below shoulder
FORWARD_MIN_EXTEND = 0.08      # wrist-shoulder / shoulder-span
CROSS_MIN_ROT_DEG  = 8.0       # torso rotation (deg) for right cross
VEL_MIN            = 0.015     # min wrist speed to gate a punch
DEBOUNCE_FRAMES    = 6
SMOOTH_ALPHA       = 0.6

# Flash red overlay for this many seconds after a BAD detection
RED_FLASH_SECONDS  = 1.0

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def angle(a,b,c):
    ax,ay=a; bx,by=b; cx,cy=c
    abx,aby=ax-bx,ay-by; cbx,cby=cx-bx,cy-by
    dot=abx*cbx+aby*cby
    mab=max(1e-9,(abx**2+aby**2)**0.5); mcb=max(1e-9,(cbx**2+cby**2)**0.5)
    cosv=max(-1,min(1,dot/(mab*mcb)))
    return math.degrees(math.acos(cosv))

def lmk_xy(lms, idx, w, h):
    lm=lms[idx]; return np.array([lm.x*w, lm.y*h], np.float32)

def length(a,b): return float(np.hypot(*(a-b)))
def moving_avg(prev, curr, a=SMOOTH_ALPHA): return curr if prev is None else a*curr+(1-a)*prev

def side_indices(is_left):
    if is_left:
        return int(mp_pose.PoseLandmark.LEFT_WRIST), int(mp_pose.PoseLandmark.LEFT_ELBOW), int(mp_pose.PoseLandmark.LEFT_SHOULDER)
    else:
        return int(mp_pose.PoseLandmark.RIGHT_WRIST), int(mp_pose.PoseLandmark.RIGHT_ELBOW), int(mp_pose.PoseLandmark.RIGHT_SHOULDER)

def shoulder_mid(lms, w, h):
    l=lmk_xy(lms, int(mp_pose.PoseLandmark.LEFT_SHOULDER), w, h)
    r=lmk_xy(lms, int(mp_pose.PoseLandmark.RIGHT_SHOULDER), w, h)
    return (l+r)/2.0, l, r

def hip_line(lms, w, h):
    l=lmk_xy(lms, int(mp_pose.PoseLandmark.LEFT_HIP), w, h)
    r=lmk_xy(lms, int(mp_pose.PoseLandmark.RIGHT_HIP), w, h)
    return l, r

def line_angle_deg(a,b):
    dx,dy=b-a; return math.degrees(math.atan2(dy,dx))

def evaluate_punch(lms, w, h, is_left, prev_wrist, dt):
    wrist_i, elbow_i, shoulder_i = side_indices(is_left)
    wrist=lmk_xy(lms, wrist_i, w, h); elbow=lmk_xy(lms, elbow_i, w, h); shoulder=lmk_xy(lms, shoulder_i, w, h)
    _,l_sh,r_sh=shoulder_mid(lms,w,h)

    # velocity
    if prev_wrist is None: vx=vy=0.0
    else:
        v=(wrist-prev_wrist)/max(dt,1e-3); vx=float(v[0]); vy=float(v[1])
    speed=(vx**2+vy**2)**0.5

    elb_ang=angle(shoulder,elbow,wrist)
    horiz=max(1e-6,abs(vx)); vert=max(1e-6,abs(vy))
    horiz_ratio=horiz/vert
    shoulder_span=length(l_sh,r_sh)
    extend=length(wrist,shoulder)/max(1e-6,shoulder_span)
    wrist_drop=wrist[1]-shoulder[1]

    l_hip,r_hip=hip_line(lms,w,h)
    sh_angle=line_angle_deg(l_sh,r_sh); hip_angle=line_angle_deg(l_hip,r_hip)
    torso_rot=abs((sh_angle-hip_angle+540)%360-180)

    is_punch_motion = speed>VEL_MIN and extend>FORWARD_MIN_EXTEND*0.5 and horiz_ratio>1.0
    if not is_punch_motion:
        return False, False, [], {"elb":elb_ang,"hr":horiz_ratio,"ext":extend,"drop":wrist_drop,"rot":torso_rot}, wrist

    reasons=[]
    if elb_ang < ELBOW_MIN_STRAIGHT: reasons.append("not_straight")
    if horiz_ratio < HORIZ_MIN_RATIO: reasons.append("wobbly_direction")
    if wrist_drop > WRIST_MAX_DROP_PX: reasons.append("wrist_drop")
    if (not is_left) and (torso_rot < CROSS_MIN_ROT_DEG): reasons.append("no_torso_rot_cross")
    bad = len(reasons)>0

    return True, bad, reasons, {"elb":elb_ang,"hr":horiz_ratio,"ext":extend,"drop":wrist_drop,"rot":torso_rot}, wrist

def tint_red(frame, strength=0.55):
    overlay=np.full_like(frame, (0,0,255))
    return cv2.addWeighted(overlay, strength, frame, 1-strength, 0)

def derive_paths(video_path):
    if video_path:
        p=pathlib.Path(video_path)
        base=str(p.with_suffix(""))
        return f"{base}_annotated{p.suffix}", f"{base}_punch_log.csv", f"{base}_report.html", str(p.parent)
    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"webcam_%s.mp4"%ts, f"webcam_{ts}_punch_log.csv", f"webcam_{ts}_report.html", os.getcwd()

def save_report(logs, out_csv, out_html, out_dir):
    # CSV
    with open(out_csv, "w", newline="") as f:
        w=csv.writer(f)
        w.writerow(["t_frame","t_sec","side","bad","reasons","elbow_deg","horiz_ratio","extend","wrist_drop_px","torso_rot_deg"])
        for r in logs:
            w.writerow([r["frame"], f"{r['time_sec']:.3f}", r["side"], int(r["bad"]),
                        "|".join(r["reasons"]), f"{r['metrics']['elb']:.2f}",
                        f"{r['metrics']['hr']:.3f}", f"{r['metrics']['ext']:.3f}",
                        f"{r['metrics']['drop']:.1f}", f"{r['metrics']['rot']:.1f}"])

    # Aggregates
    import collections as C
    reason_counts=C.Counter()
    times=[r["time_sec"] for r in logs]
    elbows=[r["metrics"]["elb"] for r in logs]
    hrs=[r["metrics"]["hr"] for r in logs]
    drops=[r["metrics"]["drop"] for r in logs]
    rots=[r["metrics"]["rot"] for r in logs]
    sides=[r["side"] for r in logs]
    bads=[r["bad"] for r in logs]
    for r in logs:
        if r["bad"]:
            for reason in r["reasons"]:
                reason_counts[reason]+=1

    def save_plot(x,y,hline=None,ylabel="",fname="plot.png",xlabel="Time (s)"):
        plt.figure()
        plt.plot(x,y)
        if hline is not None: plt.axhline(hline, linestyle="--")
        plt.xlabel(xlabel); plt.ylabel(ylabel); plt.tight_layout()
        path=os.path.join(out_dir,fname); plt.savefig(path,dpi=150); plt.close()
        return fname

    def save_bar(counter,fname="reasons.png"):
        labels=list(counter.keys()) or ["no_bad_punches"]
        values=[counter[k] for k in labels] if counter else [0]
        plt.figure()
        plt.bar(labels,values)
        plt.ylabel("Count"); plt.xticks(rotation=20, ha="right")
        plt.tight_layout(); path=os.path.join(out_dir,fname); plt.savefig(path,dpi=150); plt.close()
        return fname

    elbow_png=save_plot(times, elbows, ELBOW_MIN_STRAIGHT, "Elbow angle (°)", "elbow_angle.png")
    dir_png  =save_plot(times, hrs,    HORIZ_MIN_RATIO,    "Direction ratio (|vx|/|vy|)", "direction_ratio.png")
    drop_png =save_plot(times, drops,  WRIST_MAX_DROP_PX,  "Wrist drop (px)", "wrist_drop.png")
    rot_png  =save_plot(times, rots,   CROSS_MIN_ROT_DEG,  "Torso rotation (°)", "torso_rotation.png")
    reasons_png=save_bar(reason_counts, "bad_reasons.png")

    total=len(logs); total_bad=sum(bads)
    bad_rate=f"{(100.0*total_bad/max(1,total)):.1f}%"
    left_bad=sum(1 for i in range(total) if bads[i] and sides[i]=="L")
    right_bad=sum(1 for i in range(total) if bads[i] and sides[i]=="R")

    html=f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Punch Accuracy Report</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:24px;background:#0b0e12;color:#e8eef5;line-height:1.35}}
.card{{background:#121721;border:1px solid #1e2633;border-radius:12px;padding:16px;margin:16px 0}}
.grid{{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(280px,1fr))}}
.kpi{{font-size:28px;font-weight:700}} .label{{font-size:12px;color:#9db0c4}} img{{width:100%;border-radius:8px;border:1px solid #1e2633}}
</style></head><body>
<h1>Punch Accuracy Report</h1>
<div class="card">
  <div class="grid">
    <div><div class="kpi">{total}</div><div class="label">Total punches detected</div></div>
    <div><div class="kpi">{total_bad}</div><div class="label">Inaccurate punches</div></div>
    <div><div class="kpi">{bad_rate}</div><div class="label">% inaccurate</div></div>
    <div><div class="kpi">{left_bad}</div><div class="label">Left (jab) inaccuracies</div></div>
    <div><div class="kpi">{right_bad}</div><div class="label">Right (cross) inaccuracies</div></div>
  </div>
  <p class="label">Thresholds — elbow ≥ {ELBOW_MIN_STRAIGHT}°, direction ratio ≥ {HORIZ_MIN_RATIO}, wrist drop ≤ {WRIST_MAX_DROP_PX}px, cross rotation ≥ {CROSS_MIN_ROT_DEG}°.</p>
  <p class="label">CSV: {os.path.basename(out_csv)}</p>
</div>
<h2>Top Problems</h2>
<div class="card"><img src="{os.path.basename(reasons_png)}" alt="Reasons bar chart"/></div>
<h2>Metric Timelines</h2>
<div class="grid">
  <div class="card"><h3>Elbow angle (°)</h3><img src="{os.path.basename(elbow_png)}" alt=""></div>
  <div class="card"><h3>Direction ratio</h3><img src="{os.path.basename(dir_png)}" alt=""></div>
  <div class="card"><h3>Wrist drop (px)</h3><img src="{os.path.basename(drop_png)}" alt=""></div>
  <div class="card"><h3>Torso rotation (°)</h3><img src="{os.path.basename(rot_png)}" alt=""></div>
</div>
</body></html>"""
    with open(out_html,"w",encoding="utf-8") as f: f.write(html)

def main():
    ap=argparse.ArgumentParser()
    g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--video", type=str)
    g.add_argument("--webcam", type=int)
    ap.add_argument("--display", action="store_true")
    args=ap.parse_args()

    cap=cv2.VideoCapture(args.video if args.video else args.webcam)
    if not cap.isOpened(): raise SystemExit("Could not open input")

    fps=cap.get(cv2.CAP_PROP_FPS)
    if fps<=1 or np.isnan(fps): fps=30.0

    ok, frame = cap.read()
    if not ok: raise SystemExit("Could not read first frame")
    h, w = frame.shape[:2]
    out_video, out_csv, out_html, out_dir = derive_paths(args.video)
    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    writer=cv2.VideoWriter(out_video, fourcc, fps, (w,h))

    pose=mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5)

    prev_wrist_L=prev_wrist_R=sm_wrist_L=sm_wrist_R=None
    last_time=time.time()
    debounce_L=debounce_R=0
    frame_idx=0
    logs=[]

    # Red flash counter (in frames)
    red_flash_frames = 0
    red_flash_total  = int(max(1, round(RED_FLASH_SECONDS * fps)))

    while True:
        frame_in = frame.copy()
        rgb=cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB)
        res=pose.process(rgb)
        now=time.time(); dt=now-last_time; last_time=now
        bad_detected_this_frame=False
        info_text=""

        if res.pose_landmarks:
            lms=res.pose_landmarks.landmark
            for is_left in (True, False):
                wrist_i,_,_=side_indices(is_left)
                cur=lmk_xy(lms, wrist_i, w, h)
                if is_left:
                    sm_wrist_L=moving_avg(sm_wrist_L, cur)
                    punch,bad,reasons,metrics,prev_wrist_L=evaluate_punch(
                        lms,w,h,True,prev_wrist_L if sm_wrist_L is None else sm_wrist_L,dt)
                    if punch and debounce_L==0:
                        info_text+=f" L:{'BAD' if bad else 'OK'}"
                        logs.append({"frame":frame_idx,"time_sec":frame_idx/max(1.0,fps),"side":"L","bad":bad,"reasons":reasons,"metrics":metrics})
                        if bad: bad_detected_this_frame=True
                        debounce_L=DEBOUNCE_FRAMES
                else:
                    sm_wrist_R=moving_avg(sm_wrist_R, cur)
                    punch,bad,reasons,metrics,prev_wrist_R=evaluate_punch(
                        lms,w,h,False,prev_wrist_R if sm_wrist_R is None else sm_wrist_R,dt)
                    if punch and debounce_R==0:
                        info_text+=f" R:{'BAD' if bad else 'OK'}"
                        logs.append({"frame":frame_idx,"time_sec":frame_idx/max(1.0,fps),"side":"R","bad":bad,"reasons":reasons,"metrics":metrics})
                        if bad: bad_detected_this_frame=True
                        debounce_R=DEBOUNCE_FRAMES

            if debounce_L>0: debounce_L-=1
            if debounce_R>0: debounce_R-=1

            mp_drawing.draw_landmarks(
                frame_in, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2)
            )

        # Start a 1-second flash ONLY when a BAD is detected this frame
        if bad_detected_this_frame:
            red_flash_frames = red_flash_total

        # Apply red overlay while the flash timer runs
        frame_out = tint_red(frame_in, 0.55) if red_flash_frames > 0 else frame_in
        if red_flash_frames > 0:
            red_flash_frames -= 1

        if info_text:
            cv2.rectangle(frame_out,(10,10),(10+320,10+44),(0,0,0),-1)
            cv2.putText(frame_out,f"Punch: {info_text.strip()}",
                        (18,38),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,255),2,cv2.LINE_AA)

        writer.write(frame_out)
        if args.display:
            cv2.imshow("Jab/Cross Accuracy Checker", frame_out)
            if (cv2.waitKey(1)&0xFF) in (27, ord('q')): break

        ok, frame = cap.read()
        frame_idx += 1
        if not ok: break

    cap.release(); writer.release(); cv2.destroyAllWindows()

    save_report(logs, out_csv, out_html, out_dir)
    print(f"Saved annotated video to: {out_video}")
    print(f"Saved CSV log to:        {out_csv}")
    print(f"Saved HTML report to:    {out_html}")

if __name__=="__main__":
    main()
