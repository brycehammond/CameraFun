# Depth Anything v2 — Office Art Display

Real-time colorized depth visualization for an M1 Mac Mini + TV. Webcam feed runs through
Depth Anything v2 and renders as a shifting thermal/rainbow field on a fullscreen display.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python depth_display.py
```

The depth model (~50MB for Small) auto-downloads on first run.

## Models & downloads

Only the depth model is fetched automatically. The extra modes each need a one-time
manual download (the app prints the exact `curl` command if a model is missing, and
runs without that mode until you fetch it). Downloaded models are gitignored.

| What                    | Size   | How                                   | Enables                     |
|-------------------------|--------|---------------------------------------|-----------------------------|
| Depth Anything V2 Small | ~50 MB | auto (Hugging Face, first run)        | depth effects (`e`)         |
| Fast-neural-style `.pth`| ~26 MB | `curl` — see [style transfer](#neural-style-transfer-t-to-cycle-styles) | style modes (`t`) |
| MediaPipe bundles       | ~18 MB | `curl` — see [vision modes](#mediapipe-vision-modes-v-to-cycle) | segmentation / pose / face mesh / hands (`v`) |
| YuNet face detector     | ~230 KB| `curl` — see [face capture](#face-capture-optional) | optional face capture |

## Controls

| Key      | Action                  |
|----------|-------------------------|
| `q`/`ESC`| Quit                    |
| `f`      | Toggle fullscreen       |
| `c`      | Cycle colormap          |
| `e`      | Cycle depth effect      |
| `t`      | Cycle neural style (style transfer) |
| `v`      | Cycle vision mode (segmentation / pose / face mesh) |
| `s`      | Toggle FPS overlay (off by default; `--fps` to start on) |
| `m`      | Toggle mirror           |

### Effects (`e` to cycle, or `--effect NAME`)

| Effect    | Look                                                         |
|-----------|-------------------------------------------------------------|
| `plain`   | Straight colorized depth (default)                          |
| `palette` | Palette scrolls over time — colors flow even when still     |
| `bands`   | Posterized depth bands with contour lines (topographic map) |
| `scanner` | A glowing depth slab sweeps front-to-back; people light up  |
| `neon`    | Glowing depth-edge outlines on black (Predator vision)      |
| `trails`  | Movement leaves decaying comet trails                       |
| `dof`     | Fake depth-of-field; a focal plane sweeps depth, rest blurs |
| `cutout`  | Nearest subject pops; background dims to a muted wash       |
| `dots`    | Halftone / LED-board grid of dots sized & colored by depth  |

Colormap cycling (`c`) still applies within the color-based effects. Pass
`--cycle 30` to auto-advance through effects every 30s for hands-off ambient art.

### Neural style transfer (`t` to cycle styles)

Fast neural style transfer (Johnson et al.) repaints the live webcam frame in the
style of a painting, in real time on the Neural Engine's GPU/MPS. Unlike the depth
effects, these run their own network on the RGB frame, so the depth model is skipped
while a style is active. Four pretrained styles ship from the PyTorch examples repo:
`candy`, `mosaic`, `udnie`, `rain` (rain_princess).

`e` and `t` are independent cycles: `e` steps through the depth effects, `t` steps
through the styles, and pressing `e` from a style snaps straight back to the depth
effects. (`--cycle N` still auto-advances through everything for hands-off ambient art.)

### MediaPipe vision modes (`v` to cycle)

Neural-engine computer-vision modes that work on the RGB frame (depth is skipped
while one is active). Each needs a MediaPipe model bundle downloaded once:

| Mode         | Model bundle                  | Look                                          |
|--------------|-------------------------------|-----------------------------------------------|
| `silhouette` | `selfie_segmenter.tflite`     | People kept in real color over a scrolling color field |
| `pose`       | `pose_landmarker_lite.task`   | Glowing skeletons on a dimmed frame (up to 4 people) |
| `facemesh`   | `face_landmarker.task`        | Glowing face-mesh points (up to 5 faces)      |
| `hands`      | `hand_landmarker.task`        | Glowing hand skeletons + fingertip light-painting trails (up to 4 hands) |

```bash
# One-time: download the model bundles (~10 MB total) into mediapipe_models/
mkdir -p mediapipe_models
curl -L -o mediapipe_models/selfie_segmenter.tflite \
  https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite
curl -L -o mediapipe_models/pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
curl -L -o mediapipe_models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
curl -L -o mediapipe_models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

python depth_display.py        # `v` now cycles the vision modes
```

Each mode appears in the `v` cycle only if its bundle is present; the app prints the
exact `curl` command for any that are missing and runs without them. `e`/`t`/`v` are
three independent cycles, each remembering its place. Flags: `--vision-dir` (default
`mediapipe_models`), `--pose-count` (4), `--face-count` (5), `--hand-count` (4),
`--no-vision`.

MediaPipe's native C++ logging is very chatty (`gl_context`, feedback-manager, and
recurring clearcut-telemetry lines). It's **silenced by default** when a vision mode
is active — which matters for the always-on `err.log` — by redirecting OS-level
stderr; Python tracebacks are preserved. Pass `--verbose-mediapipe` to keep it for
debugging.

```bash
# One-time: download the pretrained style weights (~26 MB) into saved_models/
curl -L -o saved_models.zip \
  "https://www.dropbox.com/s/lrvwfehqdcxoza8/saved_models.zip?dl=1"
unzip saved_models.zip        # creates saved_models/{candy,mosaic,udnie,rain_princess}.pth

python depth_display.py        # `e` now cycles depth effects -> style modes
```

The styles appear in the `e` cycle automatically once the weights are present; if
`saved_models/` is empty the app prints the download hint and runs without them.

| Flag           | Default      | Meaning                                             |
|----------------|--------------|-----------------------------------------------------|
| `--styles-dir` | saved_models | Directory holding the style `.pth` checkpoints      |
| `--style-size` | 640          | Longest side for style inference; M1 MPS: 480 ~83fps, 640 ~52fps, 720 ~40fps |
| `--no-style`   | off          | Disable style modes even if weights are present     |

## Options

```bash
python depth_display.py --camera 0 --infer-size 392 --colormap turbo --smooth 0.5
```

- `--infer-size` — lower (e.g. 256) = faster, higher (518) = sharper
- `--model` — swap `Small` for `depth-anything/Depth-Anything-V2-Base-hf` or `-Large-hf`
- `--smooth` — EMA blend (0.4–0.6) to reduce flicker; off by default

## Troubleshooting

- **Black screen / camera error**: grant camera permission to your terminal app in
  System Settings → Privacy & Security → Camera, then restart the terminal.
- **Slow framerate**: lower `--infer-size`, or stay on the Small model.
- **Wrong webcam**: bump `--camera` to 1, 2, etc.

## Face capture (optional)

Detect faces in the webcam feed and save cropped images, running on a background
thread so the display never stalls.

```bash
# One-time: download the YuNet detector (~230 KB)
curl -L -o face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

python depth_display.py --coreml depth_anything_v2_518.mlpackage --capture-faces
```

Crops land in `captures/` as timestamped JPEGs. A per-face cooldown keyed on
screen position avoids saving a lingering person every frame, and the folder is
rotated to a max file count for always-on use.

| Flag                  | Default | Meaning                                        |
|-----------------------|---------|------------------------------------------------|
| `--capture-faces`     | off     | Enable detection + capture                     |
| `--capture-dir`       | captures| Output directory                               |
| `--face-model`        | (cwd)   | Path to the YuNet `.onnx`                       |
| `--detect-every`      | 5       | Run detection every Nth frame                  |
| `--capture-cooldown`  | 3.0     | Min seconds between saves of a face at one spot |
| `--capture-min-conf`  | 0.7     | Minimum detector confidence                    |
| `--capture-max`       | 500     | Keep at most N crops (oldest deleted; 0=∞)     |

It degrades gracefully: if the model is missing or unsupported it prints a note
and runs the display without capture. **Privacy:** this stores images of people —
post a notice, set a retention limit, and secure the folder as appropriate.

## Core ML / Neural Engine (recommended for always-on)

Running on the Apple Neural Engine (ANE) instead of MPS draws far less power and
is much faster — measured on this M1: **~128 fps @ 518px** and **~184 fps @ 392px**
on the ANE, versus **~29 fps** on MPS. The exported model bakes in all
preprocessing, so the runtime only needs `coremltools` (not torch/transformers).

```bash
# 1. Export the model (one-time, ~1 min). Produces depth_anything_v2_518.mlpackage
python convert_coreml.py                    # 518px, native grid, best quality
python convert_coreml.py --infer-size 392   # smaller/faster square input

# 2. Run the display against it — note --coreml replaces --model/--infer-size
python depth_display.py --coreml depth_anything_v2_518.mlpackage --smooth 0.5
```

Notes:
- The Core ML model uses a **fixed square input** (518 = the ViT's native 37×37
  patch grid, so no positional-embedding interpolation — most reliable + sharpest).
  Smaller sizes convert via a bilinear pos-embedding fallback and run faster.
- The conversion script verifies the Core ML output against the torch output and
  prints the max/mean difference on a normalized depth map.
- `--compute-units` (default `ALL`) lets the runtime pick the ANE; force it with
  `CPU_AND_NE` for testing.

## Always-on with launchd

`install_launchd.sh` installs a per-user **LaunchAgent** that starts the display
at login and relaunches it if it crashes. (It's an agent, not a daemon, because
the fullscreen window + camera need the logged-in GUI session.)

```bash
./install_launchd.sh             # install + start (uses the Core ML model)
./install_launchd.sh uninstall   # stop + remove
```

The agent runs `depth_display.py --coreml depth_anything_v2_518.mlpackage`, so
export the model first. Logs land in `logs/depthdisplay.{out,err}.log`. To tweak
flags (model, colormap, smoothing), edit `launchd/com.bryce.depthdisplay.plist`
and rerun the installer. Check status with:

```bash
launchctl print gui/$(id -u)/com.bryce.depthdisplay | grep -E 'state|pid'
```

On first launch macOS will prompt for Camera permission for the Python binary;
if it doesn't, add it under System Settings → Privacy & Security → Camera.
