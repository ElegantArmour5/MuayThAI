import collections, collections.abc
for _n in ("Mapping","MutableMapping","Sequence"):
    if not hasattr(collections,_n): setattr(collections,_n,getattr(collections.abc,_n))

import cv2, argparse, numpy as np

try:
    import mediapipe as mp
except Exception as e:
    raise SystemExit("Please install mediapipe: pip install mediapipe==0.10.*") from e

POSE = mp.solutions.pose
LM   = POSE.PoseLandmark
CONN = mp.solutions.pose.POSE_CONNECTIONS

def hex_to_bgr(h):
    h = h.strip().lstrip("#")
    if len(h)==3: h = "".join([c*2 for c in h])
    r = int(h[0:2],16); g = int(h[2:4],16); b = int(h[4:6],16)
    return (b,g,r)

def draw_pose_skeleton(canvas, landmarks, w, h, line_color, joint_color, line_thick, joint_radius, vis_thresh=0.5):
    pts = []
    vis = []
    for lm in landmarks:
        pts.append((int(lm.x*w), int(lm.y*h)))
        vis.append(lm.visibility)
    pts = np.array(pts, dtype=np.int32)
    vis = np.array(vis, dtype=np.float32)

    for a,b in CONN:
        a, b = int(a), int(b)
        if vis[a] > vis_thresh and vis[b] > vis_thresh:
            pa, pb = tuple(pts[a]), tuple(pts[b])
            cv2.line(canvas, pa, pb, line_color, line_thick, cv2.LINE_AA)

    for i, p in enumerate(pts):
        if vis[i] > vis_thresh:
            cv2.circle(canvas, (int(p[0]), int(p[1])), joint_radius, joint_color, -1, lineType=cv2.LINE_AA)

def main():
    ap = argparse.ArgumentParser(description="Replay MediaPipe 2D skeleton from video")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["overlay","skeleton","side"], default="skeleton")
    ap.add_argument("--bg", default="#000000", help="Background color for skeleton mode (hex)")
    ap.add_argument("--line", type=int, default=3, help="Bone thickness")
    ap.add_argument("--radius", type=int, default=4, help="Joint radius")
    ap.add_argument("--line_color", default="#B4E0FF")
    ap.add_argument("--joint_color", default="#FFFFFF")
    ap.add_argument("--vis", type=float, default=0.5, help="Visibility threshold")
    ap.add_argument("--model_complexity", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open: {args.video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if args.mode == "side":
        out_size = (W*2, H)
    else:
        out_size = (W, H)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, FPS, out_size)

    bg_bgr = hex_to_bgr(args.bg)
    line_bgr  = hex_to_bgr(args.line_color)
    joint_bgr = hex_to_bgr(args.joint_color)

    pose = POSE.Pose(static_image_mode=False, model_complexity=args.model_complexity,
                     enable_segmentation=False)

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if args.max_frames and n >= args.max_frames: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        skeleton_frame = np.full((H, W, 3), bg_bgr, np.uint8)
        if res.pose_landmarks:
            draw_pose_skeleton(
                skeleton_frame,
                res.pose_landmarks.landmark,
                W, H,
                line_bgr, joint_bgr,
                args.line, args.radius,
                vis_thresh=args.vis
            )

        if args.mode == "overlay":
            out_frame = frame.copy()
            mask = (skeleton_frame != np.array(bg_bgr, dtype=np.uint8)).any(axis=2)
            out_frame[mask] = skeleton_frame[mask]
        elif args.mode == "skeleton":
            out_frame = skeleton_frame
        else:  # side
            out_frame = np.zeros((H, W*2, 3), dtype=np.uint8)
            out_frame[:, :W] = frame
            out_frame[:, W:] = skeleton_frame
            cv2.putText(out_frame, "Original", (16,32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(out_frame, "Skeleton", (W+16,32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230,230,230), 2, cv2.LINE_AA)

        writer.write(out_frame)
        n += 1

    writer.release()
    cap.release()
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
