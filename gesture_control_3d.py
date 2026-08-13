"""
GestureCanvas 3D
---------------------------
Control a 3D mesh using hand gestures captured from your webcam.

  * PINCH (thumb + index finger together/apart)   -> Zoom / Scale
  * TWIST (roll your wrist)                       -> Rotate about Z (roll)
  * TILT hand up/down (using MediaPipe's z-depth)  -> Rotate about X (pitch)
  * TURN hand left/right (using z-depth)           -> Rotate about Y (yaw)
  * SECOND HAND position                           -> Pan the model
  * FIST                                           -> Toggle wireframe
  * OPEN PALM (held)                               -> Reset view

Built with OpenCV + MediaPipe (hand landmark tracking, same as GestureCanvas 2D)
and Open3D (mesh rendering + transforms).

Controls (in the webcam window):
  q       - quit
  r       - reset scale / rotation / pan
  w       - toggle wireframe manually
  e       - export current mesh (with transform baked in) to disk
  SPACE   - freeze/unfreeze rotation tracking

Usage:
  python gesture_control_3d.py                     # uses a built-in placeholder mesh
  python gesture_control_3d.py --model path/to.obj  # uses your own mesh
"""

import argparse
import math
import os
import time
import urllib.request

import cv2
import numpy as np
import open3d as o3d
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

# --------------------------------------------------------------------------- #
# Model download (same approach as the 2D version)
# --------------------------------------------------------------------------- #
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base
]

# --------------------------------------------------------------------------- #
# Tunable parameters
# --------------------------------------------------------------------------- #
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6

PINCH_RATIO_MIN = 0.25
PINCH_RATIO_MAX = 1.4
SCALE_MIN = 0.4
SCALE_MAX = 2.5

# How strongly wrist->middle-knuckle depth maps to pitch/yaw (degrees per unit z)
DEPTH_ANGLE_GAIN = 220.0

# Fist detection: fingertip-to-wrist distance (normalized by hand size) below
# this counts as "curled" for that finger.
FIST_CURL_RATIO = 0.55
# Open palm: all fingertips extended beyond this ratio, held for RESET_HOLD_SEC
PALM_OPEN_RATIO = 0.9
RESET_HOLD_SEC = 1.0

SMOOTHING = 0.75  # exponential moving average factor

# Landmark indices (MediaPipe Hands)
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (one-time download, ~10 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def lerp(value, in_lo, in_hi, out_lo, out_hi):
    t = clamp((value - in_lo) / (in_hi - in_lo + 1e-6), 0.0, 1.0)
    return out_lo + t * (out_hi - out_lo)


def draw_hand_landmarks(frame, landmarks_px):
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, landmarks_px[start_idx], landmarks_px[end_idx], (0, 200, 0), 2)
    for x, y in landmarks_px:
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)


def make_placeholder_mesh():
    """A little scene (box + sphere) so the demo works without any asset file."""
    box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    box.translate(-box.get_center())
    box.paint_uniform_color([0.85, 0.45, 0.2])

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.65)
    sphere.translate(-sphere.get_center())
    sphere.paint_uniform_color([0.2, 0.6, 0.95])

    mesh = box + sphere
    mesh.compute_vertex_normals()
    return mesh


def load_mesh(path):
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh.is_empty():
        raise FileNotFoundError(f"Could not load mesh (or it has no geometry): {path}")
    mesh.translate(-mesh.get_center())  # center it at the origin

    # Normalize size so gesture scaling behaves consistently regardless of
    # the source mesh's original units.
    extent = mesh.get_max_bound() - mesh.get_min_bound()
    largest = max(extent.max(), 1e-6)
    mesh.scale(1.6 / largest, center=(0, 0, 0))

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    if not mesh.has_vertex_colors() and not mesh.has_triangle_uvs():
        mesh.paint_uniform_color([0.7, 0.7, 0.75])
    return mesh


def finger_extended_ratio(tip, wrist_pt, hand_size):
    return np.linalg.norm(tip - wrist_pt) / hand_size


def rotation_matrix_xyz(pitch_deg, yaw_deg, roll_deg):
    """Combined rotation matrix, applied yaw -> pitch -> roll."""
    rx = math.radians(pitch_deg)
    ry = math.radians(yaw_deg)
    rz = math.radians(roll_deg)

    Rx = np.array([[1, 0, 0],
                   [0, math.cos(rx), -math.sin(rx)],
                   [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)],
                   [0, 1, 0],
                   [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0],
                   [math.sin(rz), math.cos(rz), 0],
                   [0, 0, 1]])
    return Rz @ Rx @ Ry


def main():
    parser = argparse.ArgumentParser(description="Gesture-controlled 3D model viewer")
    parser.add_argument("--model", type=str, default=None, help="Path to a mesh file (.obj/.stl/.ply)")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default 0)")
    args = parser.parse_args()

    ensure_model()

    base_mesh = load_mesh(args.model) if args.model else make_placeholder_mesh()
    base_vertices = np.asarray(base_mesh.vertices).copy()

    display_mesh = o3d.geometry.TriangleMesh(base_mesh)  # working copy we mutate every frame

    # --- Open3D viewport setup ---
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="GestureCanvas 3D - Model View", width=800, height=800)
    vis.add_geometry(display_mesh)
    render_opt = vis.get_render_option()
    render_opt.mesh_show_back_face = True
    render_opt.background_color = np.array([0.08, 0.08, 0.1])

    # --- Webcam / MediaPipe setup ---
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try a different --camera index.")

    landmarker_options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=2,
        running_mode=vision.RunningMode.VIDEO,
        min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    landmarker = vision.HandLandmarker.create_from_options(landmarker_options)
    frame_timestamp_ms = 0

    scale = 1.0
    pitch = yaw = roll = 0.0
    smoothed_scale = 1.0
    smoothed_pitch = smoothed_yaw = smoothed_roll = 0.0
    pan_x = pan_y = 0.0
    smoothed_pan_x = smoothed_pan_y = 0.0

    frozen = False
    wireframe = False
    open_palm_since = None

    prev_time = time.time()

    print("GestureCanvas 3D started.")
    print("  Pinch thumb+index         -> zoom/scale")
    print("  Twist wrist                -> roll (Z rotation)")
    print("  Tilt hand up/down          -> pitch (X rotation)")
    print("  Turn hand left/right       -> yaw (Y rotation)")
    print("  Second hand                -> pan")
    print("  Fist                       -> toggle wireframe")
    print("  Open palm (hold ~1s)       -> reset")
    print("  q: quit | r: reset | w: wireframe | e: export mesh | SPACE: freeze rotation")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_timestamp_ms += 33
        detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        gesture_active = False
        fist_detected = False
        palm_open_now = False

        if detection_result.hand_landmarks:
            hands = detection_result.hand_landmarks
            primary = hands[0]

            landmarks_px = [(int(p.x * w), int(p.y * h)) for p in primary]
            draw_hand_landmarks(frame, landmarks_px)

            # Full 3D (normalized) coordinates for the primary hand
            pts = np.array([[p.x, p.y, p.z] for p in primary])
            wrist_pt = pts[WRIST]
            middle_mcp_pt = pts[MIDDLE_MCP]
            thumb_tip_pt = pts[THUMB_TIP]
            index_tip_pt = pts[INDEX_TIP]

            hand_size = np.linalg.norm(middle_mcp_pt - wrist_pt) + 1e-6

            # --- Zoom (pinch, same formula as the 2D version, in pixel space) ---
            thumb_px = np.array([thumb_tip_pt[0] * w, thumb_tip_pt[1] * h])
            index_px = np.array([index_tip_pt[0] * w, index_tip_pt[1] * h])
            wrist_px = np.array([wrist_pt[0] * w, wrist_pt[1] * h])
            mcp_px = np.array([middle_mcp_pt[0] * w, middle_mcp_pt[1] * h])
            hand_size_px = np.linalg.norm(mcp_px - wrist_px) + 1e-6
            pinch_ratio = np.linalg.norm(index_px - thumb_px) / hand_size_px
            scale = lerp(pinch_ratio, PINCH_RATIO_MIN, PINCH_RATIO_MAX, SCALE_MIN, SCALE_MAX)

            if not frozen:
                # Roll: same 2D angle trick as the original script
                dx, dy = index_px - thumb_px
                roll = math.degrees(math.atan2(dy, dx))

                # Pitch/Yaw: MediaPipe gives each landmark a z (depth relative
                # to the wrist), so we can read hand tilt directly from it.
                #   - Yaw (turning the hand left/right) shows up as a depth
                #     difference between the thumb side and pinky side.
                #   - Pitch (tilting the hand up/down) shows up as a depth
                #     difference between the wrist and the middle knuckle.
                dvec = middle_mcp_pt - wrist_pt  # (dx, dy, dz) normalized
                yaw = clamp((thumb_tip_pt[2] - pts[PINKY_TIP][2]) * DEPTH_ANGLE_GAIN, -80, 80)
                pitch = clamp(dvec[2] * DEPTH_ANGLE_GAIN, -80, 80)

            gesture_active = True

            # --- Fist / open-palm detection ---
            tips = [pts[INDEX_TIP], pts[MIDDLE_TIP], pts[RING_TIP], pts[PINKY_TIP]]
            extended_ratios = [finger_extended_ratio(t[:2], wrist_pt[:2], hand_size) for t in tips]
            fist_detected = all(r < FIST_CURL_RATIO for r in extended_ratios)
            palm_open_now = all(r > PALM_OPEN_RATIO for r in extended_ratios)

            cv2.line(frame, tuple(thumb_px.astype(int)), tuple(index_px.astype(int)), (0, 255, 0), 3)

            # --- Second hand controls pan ---
            if len(hands) > 1:
                second = hands[1]
                second_px = [(int(p.x * w), int(p.y * h)) for p in second]
                draw_hand_landmarks(frame, second_px)
                second_wrist = second[WRIST]
                # map normalized screen position (centered) to pan offset
                pan_x = (second_wrist.x - 0.5) * 2.0
                pan_y = (second_wrist.y - 0.5) * -2.0
                cv2.putText(frame, "PAN (2nd hand)", (second_px[WRIST][0] - 40, second_px[WRIST][1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Open-palm hold -> reset
        if palm_open_now:
            if open_palm_since is None:
                open_palm_since = time.time()
            elif time.time() - open_palm_since > RESET_HOLD_SEC:
                scale, pitch, yaw, roll, pan_x, pan_y = 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
                open_palm_since = time.time()  # avoid repeated resets while held
        else:
            open_palm_since = None

        if fist_detected:
            wireframe = True
        # (fist only forces wireframe ON; 'w' key toggles it back off manually)

        # --- Smoothing ---
        smoothed_scale = SMOOTHING * smoothed_scale + (1 - SMOOTHING) * scale
        smoothed_pitch = SMOOTHING * smoothed_pitch + (1 - SMOOTHING) * pitch
        smoothed_yaw = SMOOTHING * smoothed_yaw + (1 - SMOOTHING) * yaw
        smoothed_roll = SMOOTHING * smoothed_roll + (1 - SMOOTHING) * roll
        smoothed_pan_x = SMOOTHING * smoothed_pan_x + (1 - SMOOTHING) * pan_x
        smoothed_pan_y = SMOOTHING * smoothed_pan_y + (1 - SMOOTHING) * pan_y

        # --- Apply full transform to a fresh copy of the base mesh each frame ---
        R = rotation_matrix_xyz(smoothed_pitch, smoothed_yaw, smoothed_roll)
        new_vertices = (base_vertices @ R.T) * smoothed_scale
        new_vertices[:, 0] += smoothed_pan_x
        new_vertices[:, 1] += smoothed_pan_y

        display_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
        display_mesh.compute_vertex_normals()
        vis.update_geometry(display_mesh)
        render_opt.mesh_show_wireframe = wireframe
        vis.poll_events()
        vis.update_renderer()

        # --- HUD on the webcam window ---
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        status = "TRACKING" if gesture_active else "NO HAND DETECTED"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if gesture_active else (0, 0, 255), 2)
        cv2.putText(frame, f"Scale:{smoothed_scale:.2f} Pitch:{smoothed_pitch:.0f} "
                            f"Yaw:{smoothed_yaw:.0f} Roll:{smoothed_roll:.0f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}  Wireframe: {'ON' if wireframe else 'OFF'}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "q: quit  r: reset  w: wireframe  e: export  SPACE: freeze",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("GestureCanvas 3D - Webcam", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            scale, pitch, yaw, roll, pan_x, pan_y = 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
            smoothed_scale, smoothed_pitch, smoothed_yaw, smoothed_roll = 1.0, 0.0, 0.0, 0.0
            smoothed_pan_x = smoothed_pan_y = 0.0
        elif key == ord('w'):
            wireframe = not wireframe
        elif key == ord('e'):
            out_path = f"gesture_model_output_{int(time.time())}.obj"
            o3d.io.write_triangle_mesh(out_path, display_mesh)
            print(f"Saved: {out_path}")
        elif key == ord(' '):
            frozen = not frozen
            print("Rotation frozen" if frozen else "Rotation unfrozen")

        if not vis.poll_events():
            break

    cap.release()
    cv2.destroyAllWindows()
    vis.destroy_window()


if __name__ == "__main__":
    main()
