

Readme · MD
GestureCanvas 3D
Real-time gesture-controlled 3D model viewer using MediaPipe hand tracking and Open3D — pinch to scale, tilt to rotate, no mouse required.

GestureCanvas 3D lets you manipulate a 3D model using only your hands in front of a webcam. It tracks 21 hand landmarks per hand in real time, turns pinch/twist/tilt/turn motions into scale and rotation, and renders the result live in a 3D viewport — all touchless.

This is the 3D evolution of an earlier 2D project (pinch-zoom + twist-rotate on a flat image); this version replaces the flat-image renderer with a real mesh and adds depth-aware gestures using MediaPipe's z landmark coordinate.

Demo
Gesture	Action
Pinch (thumb + index apart/together)	Scale the model
Twist your wrist	Roll (rotate about Z)
Tilt hand up/down	Pitch (rotate about X)
Turn hand left/right	Yaw (rotate about Y)
Second hand	Pan the model
Fist	Toggle wireframe view
Open palm, held ~1s	Reset view
Features
Real-time hand tracking with MediaPipe's HandLandmarker (Tasks API)
Full 3D rotation (pitch, yaw, roll) driven by gesture data, not just in-plane motion
Two-hand support — second hand pans the scene independently
Fist/open-palm gesture detection for wireframe toggle and view reset (rule-based, no extra ML model)
Exponential moving average smoothing to reduce landmark jitter
Load your own .obj / .stl / .ply mesh, or run with a built-in placeholder scene
Export the transformed mesh back to .obj
Live HUD overlay (FPS, tracking status, current scale/rotation values)
Tech Stack
MediaPipe — hand landmark detection (21 points per hand, including depth)
OpenCV — webcam capture, image compositing, HUD rendering
Open3D — 3D mesh loading, transformation, and rendering
NumPy — landmark math (angles, distances, rotation matrices)
How it Works
OpenCV captures webcam frames; each frame is passed to MediaPipe's HandLandmarker.
MediaPipe returns 21 3D landmarks per detected hand — (x, y, z), where z is depth relative to the wrist.
Landmark geometry is converted into gesture signals:
Pinch ratio (thumb-tip to index-tip distance, normalized by hand size) → scale
Thumb–index angle in the image plane → roll
Depth difference between thumb-side and pinky-side of the hand → yaw
Depth difference between wrist and middle knuckle → pitch
Fingertip curl/extension ratios → fist / open-palm detection
Gesture values are smoothed (EMA) and applied as a single transform to a fresh copy of the base mesh each frame (rotation matrix + scale + pan offset), avoiding drift.
Open3D re-renders the transformed mesh every frame in a live viewport, alongside the annotated webcam feed.
Getting Started
Prerequisites
Python 3.9–3.11
A webcam
~200MB free disk space (Open3D + MediaPipe models)
Installation
bash
git clone https://github.com/<your-username>/gesture-canvas-3d.git
cd gesture-canvas-3d
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
Run
With the built-in placeholder scene (box + sphere):

bash
python gesture_control_3d.py
With your own mesh:

bash
python gesture_control_3d.py --model path/to/model.obj
If the wrong camera opens by default:

bash
python gesture_control_3d.py --camera 1
The first run downloads the hand-tracking model (~10MB) automatically. Two windows open: the webcam feed with landmark overlay + HUD, and a separate Open3D viewport with the live mesh.

Controls
Key	Action
q	Quit
r	Reset scale / rotation / pan
w	Toggle wireframe manually
e	Export the current transformed mesh to .obj
SPACE	Freeze / unfreeze rotation tracking
Configuration
Key tunable constants at the top of gesture_control_3d.py:

Constant	Purpose
PINCH_RATIO_MIN / MAX, SCALE_MIN / MAX	Zoom sensitivity range
DEPTH_ANGLE_GAIN	How strongly hand tilt/turn maps to rotation degrees
FIST_CURL_RATIO / PALM_OPEN_RATIO	Thresholds for fist / open-palm detection
SMOOTHING	EMA factor — higher = smoother but laggier
Project Structure
gesture-canvas-3d/
├── gesture_control_3d.py   # main application
├── requirements.txt        # Python dependencies
└── README.md
Roadmap / Ideas
Vertex sculpting — displace mesh vertices near the fingertip for simplified digital clay sculpting
Multi-object scenes — point gesture to select which mesh is active before transforming it
Web version — stream landmark data to a Three.js scene over WebSocket for browser-based demos
Gesture logging — log FPS / detection-rate / gesture events to CSV for an evaluation report
Troubleshooting
Wrong camera opens — run with --camera 1 (try other indices).
No module named cv2/mediapipe/open3d — your virtual environment isn't active, or isn't selected as the interpreter in your editor.
Open3D window is blank/black — usually a GPU/OpenGL driver issue, common over remote desktop, SSH, or WSL without GPU passthrough. Run on a local machine with a real display.
Laggy or twitchy pitch/yaw — lower DEPTH_ANGLE_GAIN if too twitchy, raise it if sluggish; MediaPipe's z is noisier than x/y.
License
MIT — see LICENSE.
