# GestureCanvas 3D

Control a real 3D mesh with hand gestures from your webcam — no mouse, no
keyboard, no VR headset. This extends the original 2D GestureCanvas
(pinch-zoom + twist-rotate on a flat image) into full 3D model manipulation
using Open3D.

- **Pinch** (thumb + index apart/together) → **Scale** the model
- **Twist** your wrist → **Roll** (rotate about Z)
- **Tilt** your hand up/down → **Pitch** (rotate about X)
- **Turn** your hand left/right → **Yaw** (rotate about Y)
- **Second hand** → **Pan** the model around the scene
- **Fist** → toggle **wireframe** view
- **Open palm**, held ~1 second → **Reset** view

## What's new vs. the 2D version

| | 2D (GestureCanvas) | 3D (this version) |
|---|---|---|
| Asset | flat PNG/JPG | `.obj` / `.stl` / `.ply` mesh |
| Renderer | `cv2.warpAffine` overlay | Open3D `Visualizer` window |
| Rotation | 1 axis (in-plane) | 3 axes (pitch/yaw/roll), using MediaPipe's `z` landmark coordinate that the 2D version never used |
| Hands tracked | 1 | 2 (second hand = pan) |
| Extra gestures | — | fist = wireframe, open palm = reset |
| Export | save transformed PNG | export transformed **mesh** (`.obj`) |

The hand-tracking core (MediaPipe `HandLandmarker`, EMA smoothing, HUD
overlay) is unchanged from the original script — only the "what do we do
with the gesture values" part changed.

## How pitch/yaw work (the actual 3D part)

MediaPipe's hand landmarks aren't just `(x, y)` — each point also has a `z`
(depth relative to the wrist, in the same normalized scale as x/y). This
script uses that third coordinate, which the 2D version ignored entirely:

- **Yaw**: compares the depth of the thumb-side vs. pinky-side of the hand.
  Turning your hand like a doorknob makes one side move toward the camera
  and the other away — that depth difference drives left/right rotation.
- **Pitch**: compares the depth of the wrist vs. the middle-finger knuckle.
  Tilting your hand forward/back changes that gap — drives up/down rotation.
- **Roll**: same 2D angle trick as the original (angle of the
  thumb-tip → index-tip line).

## Setup

```bash
pip install -r requirements.txt
```

Requires a working webcam. Tested with Python 3.9+.

## Run it

With the built-in placeholder scene (box + sphere):
```bash
python gesture_control_3d.py
```

With your own mesh:
```bash
python gesture_control_3d.py --model path/to/model.obj
```

If the wrong camera opens by default:
```bash
python gesture_control_3d.py --camera 1
```

Two windows open: your webcam feed with landmark overlay + HUD, and a
separate Open3D viewport showing the live mesh.

## Controls

| Key     | Action                                       |
|---------|-----------------------------------------------|
| `q`     | Quit                                          |
| `r`     | Reset scale / rotation / pan                  |
| `w`     | Toggle wireframe manually                     |
| `e`     | Export the current transformed mesh to `.obj` |
| `SPACE` | Freeze/unfreeze rotation tracking             |

## Tuning

Key constants at the top of `gesture_control_3d.py`:

- `PINCH_RATIO_MIN/MAX`, `SCALE_MIN/MAX` — zoom sensitivity (same idea as 2D version).
- `DEPTH_ANGLE_GAIN` — how strongly hand tilt/turn maps to rotation degrees. Raise it if pitch/yaw feel sluggish, lower it if they're twitchy (MediaPipe's z is noisier than x/y).
- `FIST_CURL_RATIO` / `PALM_OPEN_RATIO` — thresholds for fist/open-palm detection.
- `SMOOTHING` — EMA factor, higher = smoother but laggier.

## Further ideas to extend this

- **Vertex sculpting**: use the fingertip's 3D position (mapped into model
  space) to displace nearby mesh vertices — simplified digital clay sculpting.
- **Multi-object scenes**: load several meshes, use a "point" gesture to
  pick which one is active before transforming it.
- **Swap Open3D for Three.js**: stream landmark data over a local WebSocket
  to a browser-based Three.js scene for nicer materials/lighting and easy
  sharing as a web demo.
- **Gesture confidence logging**: log FPS/detection-rate/gesture events to
  CSV for an evaluation section in a report.
