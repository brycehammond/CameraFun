#!/usr/bin/env python3
"""
Depth Anything v2 — real-time colorized depth display for a TV.

Webcam -> Depth Anything v2 (MPS) -> colorized depth map -> fullscreen.

This is a working baseline. See CLAUDE.md for the full feature list to build out.
"""

import argparse
import os
import threading
import time

import cv2
import numpy as np

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")

COLORMAPS = {
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "turbo": cv2.COLORMAP_TURBO,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "plasma": cv2.COLORMAP_PLASMA,
    "ocean": cv2.COLORMAP_OCEAN,
}
COLORMAP_ORDER = list(COLORMAPS.keys())


# --- Visual effects ----------------------------------------------------------
# Each effect maps a normalized depth map -> a BGR image. Signature:
#   fx(norm, cmap, t, st) -> uint8 BGR
#     norm : float32 HxW in [0, 1]   (near/far depends on the model)
#     cmap : the active cv2 colormap id (so `c` still cycles palettes)
#     t    : seconds since start (for animation)
#     st   : per-effect dict for state that must persist across frames
# Cycle through them live with the `e` key; the active one shows in the overlay.

def fx_plain(norm, cmap, t, st):
    """Straight colorized depth — the original look."""
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cmap)


def fx_palette_scroll(norm, cmap, t, st):
    """Scroll the palette over time so colors flow even when nobody moves."""
    offset = int(t * 40) % 256  # ~40 LUT steps/sec
    u8 = ((norm * 255).astype(np.int16) + offset) % 256
    return cv2.applyColorMap(u8.astype(np.uint8), cmap)


def fx_depth_bands(norm, cmap, t, st, n=12):
    """Posterize depth into bands with dark contour lines (topographic map)."""
    band = np.clip((norm * n).astype(np.int32), 0, n - 1)
    u8 = (band / (n - 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(u8, cmap)
    edges = np.zeros(norm.shape, dtype=bool)
    edges[:, 1:] |= band[:, 1:] != band[:, :-1]
    edges[1:, :] |= band[1:, :] != band[:-1, :]
    colored[edges] = 0
    return colored


def fx_scanner(norm, cmap, t, st, width=0.07, speed=0.22):
    """A glowing depth slab sweeps front-to-back; people light up as it passes."""
    center = (t * speed) % 1.0
    mask = np.exp(-((norm - center) ** 2) / (2 * width ** 2))
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cmap)
    return (colored * mask[..., None]).astype(np.uint8)


def fx_neon(norm, cmap, t, st):
    """Glowing depth-edge outlines on black (Predator vision)."""
    u8 = (norm * 255).astype(np.uint8)
    gx = cv2.Sobel(u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(u8, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = np.clip(mag / (mag.max() + 1e-6) * 3.0, 0, 1)
    colored = cv2.applyColorMap(u8, cmap).astype(np.float32)
    edges = colored * mag[..., None]
    glow = cv2.GaussianBlur(edges, (0, 0), 4)
    return np.clip(edges + glow, 0, 255).astype(np.uint8)


def fx_trails(norm, cmap, t, st, decay=0.85):
    """Decaying history buffer so movement leaves comet trails."""
    base = cv2.applyColorMap((norm * 255).astype(np.uint8), cmap).astype(np.float32)
    buf = st.get("buf")
    if buf is None or buf.shape != base.shape:
        buf = base.copy()
    else:
        buf = np.maximum(base, buf * decay)
    st["buf"] = buf
    return buf.astype(np.uint8)


def fx_dof(norm, cmap, t, st, sigma=8.0):
    """Fake depth-of-field: a focal plane sweeps depth; off-plane pixels blur.

    The blurred plate is inherently soft, so we compute it at half resolution
    and upscale — a big speed win — then lerp against the full-res sharp frame
    by each pixel's distance from the focal plane, keeping in-focus areas crisp.
    """
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cmap).astype(np.float32)
    focus = 0.5 + 0.5 * np.sin(t * 0.5)  # focal depth drifts in [0, 1]
    blur_amt = np.clip(np.abs(norm - focus) / 0.5, 0, 1)[..., None]  # 0 sharp..1 blurry

    h, w = norm.shape
    small = cv2.resize(colored, (w // 2, h // 2))
    blurred = cv2.resize(cv2.GaussianBlur(small, (0, 0), sigma / 2), (w, h))
    out = colored * (1 - blur_amt) + blurred * blur_amt
    return out.astype(np.uint8)


def fx_cutout(norm, cmap, t, st, thresh=0.55):
    """Isolate the nearest subject; dim, desaturated background behind it.

    Foreground (near) is colorized brightly; the background gets a darker,
    contrast-reduced wash so the closest person pops off the field.
    """
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cmap).astype(np.float32)
    # Soft mask around the threshold so the cutout edge isn't jagged.
    mask = np.clip((norm - thresh) / 0.08 + 0.5, 0, 1)
    mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 2)[..., None]
    bg = colored * 0.25 + 30.0  # dim + lifted floor = muted wash
    out = colored * mask + bg * (1 - mask)
    return np.clip(out, 0, 255).astype(np.uint8)


def fx_dots(norm, cmap, t, st, cell=14):
    """Halftone / LED-board look: a grid of dots sized & colored by depth."""
    h, w = norm.shape
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cmap)
    out = np.zeros_like(colored)
    half = cell // 2
    for cy in range(half, h, cell):
        for cx in range(half, w, cell):
            d = float(norm[cy, cx])
            radius = max(1, int(d * (half - 1)))
            color = tuple(int(c) for c in colored[cy, cx])
            cv2.circle(out, (cx, cy), radius, color, -1, cv2.LINE_AA)
    return out


EFFECTS = {
    "plain": fx_plain,
    "palette": fx_palette_scroll,
    "bands": fx_depth_bands,
    "scanner": fx_scanner,
    "neon": fx_neon,
    "trails": fx_trails,
    "dof": fx_dof,
    "cutout": fx_cutout,
    "dots": fx_dots,
}
EFFECT_ORDER = list(EFFECTS.keys())


def pick_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class TorchBackend:
    """HF Depth Anything v2 on MPS/CUDA/CPU. Returns a frame-sized depth map."""

    def __init__(self, model_name, infer_size):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self._torch = torch
        self.device = pick_device()
        print(f"Loading {model_name} on {self.device} ...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = (
            AutoModelForDepthEstimation.from_pretrained(model_name)
            .to(self.device)
            .eval()
        )
        self.infer_size = infer_size

    def infer(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        scale = self.infer_size / max(h, w)
        small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            depth = self.model(**inputs).predicted_depth  # (1, H, W)
        depth = depth.squeeze().cpu().numpy()
        return cv2.resize(depth, (w, h))


class CoreMLBackend:
    """Exported .mlpackage running on the Apple Neural Engine (see convert_coreml.py).

    The model bakes in preprocessing and takes a fixed square RGB image, so this
    path needs neither torch nor transformers at runtime — just coremltools.
    """

    def __init__(self, package_path, compute_units="ALL"):
        import coremltools as ct
        from PIL import Image
        self._Image = Image
        print(f"Loading Core ML model {package_path} (compute_units={compute_units}) ...")
        self.model = ct.models.MLModel(
            package_path, compute_units=getattr(ct.ComputeUnit, compute_units)
        )
        spec = self.model.get_spec()
        img_in = spec.description.input[0].type.imageType
        self.in_name = spec.description.input[0].name
        self.out_name = spec.description.output[0].name
        self.side = (int(img_in.height), int(img_in.width))
        print(f"  input {self.in_name} {self.side}, output {self.out_name}")

    def infer(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        sq = cv2.resize(frame_bgr, (self.side[1], self.side[0]))
        rgb = cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)
        out = self.model.predict({self.in_name: self._Image.fromarray(rgb)})
        depth = np.squeeze(out[self.out_name])
        return cv2.resize(depth, (w, h))


class FaceCapture:
    """Background face detection + capture (YuNet on a worker thread).

    Detection runs off the render thread so the display never stalls. Each
    detected face is cropped (with margin) and saved, with a per-face cooldown
    keyed on screen position so a lingering person isn't saved every frame. The
    capture directory is rotated to a max file count for always-on use.
    """

    def __init__(self, model_path, capture_dir, min_conf, cooldown,
                 max_files, margin=0.4):
        self.detector = cv2.FaceDetectorYN_create(
            model_path, "", (320, 320), score_threshold=min_conf)
        self.capture_dir = capture_dir
        self.cooldown = cooldown
        self.max_files = max_files
        self.margin = margin
        os.makedirs(capture_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._latest = None
        self._recent = []  # [(cx, cy, last_saved_time), ...]
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def offer(self, frame):
        """Hand the worker the latest frame (cheap copy; called every Nth frame)."""
        with self._lock:
            self._latest = frame.copy()

    def _loop(self):
        while self._running:
            with self._lock:
                frame = self._latest
                self._latest = None
            if frame is None:
                time.sleep(0.005)
                continue
            try:
                self._process(frame)
            except Exception as e:  # never let capture kill the display
                print(f"[face-capture] error: {e}")

    def _process(self, frame):
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        if faces is None:
            return
        now = time.time()
        for f in faces:
            x, y, fw, fh = (int(v) for v in f[:4])
            score = float(f[-1])
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(w, x + fw), min(h, y + fh)
            if x1 <= x0 or y1 <= y0:
                continue
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            if not self._should_save(cx, cy, max(fw, fh), now):
                continue
            self._save(frame, x0, y0, x1, y1, now, score)

    def _should_save(self, cx, cy, size, now):
        """True if no recently-saved face sits near this one; updates state."""
        radius2 = max(size, 1) ** 2
        for i, (rx, ry, ts) in enumerate(self._recent):
            if (rx - cx) ** 2 + (ry - cy) ** 2 <= radius2:
                if now - ts < self.cooldown:
                    return False
                self._recent[i] = (cx, cy, now)
                return True
        self._recent.append((cx, cy, now))
        # Drop entries we haven't seen in a while so the list stays small.
        self._recent = [r for r in self._recent if now - r[2] < self.cooldown * 4]
        return True

    def _save(self, frame, x0, y0, x1, y1, now, score):
        h, w = frame.shape[:2]
        mw, mh = int((x1 - x0) * self.margin), int((y1 - y0) * self.margin)
        crop = frame[max(0, y0 - mh):min(h, y1 + mh),
                     max(0, x0 - mw):min(w, x1 + mw)]
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
        name = f"face_{ts}_{int((now % 1) * 1000):03d}_{(x0 + x1) // 2}.jpg"
        cv2.imwrite(os.path.join(self.capture_dir, name), crop)
        print(f"[face-capture] saved {name} (conf={score:.2f})")
        self._rotate()

    def _rotate(self):
        if self.max_files <= 0:
            return
        files = sorted(f for f in os.listdir(self.capture_dir)
                       if f.endswith(".jpg"))
        for f in files[:max(0, len(files) - self.max_files)]:
            try:
                os.remove(os.path.join(self.capture_dir, f))
            except OSError:
                pass


def build_face_capture(args):
    """Construct a started FaceCapture, or None (with a reason) on any problem."""
    if not args.capture_faces:
        return None
    if not hasattr(cv2, "FaceDetectorYN_create"):
        print("[face-capture] this OpenCV build lacks FaceDetectorYN; skipping.")
        return None
    if not os.path.exists(args.face_model):
        print(f"[face-capture] model not found: {args.face_model}\n"
              f"  Download it (~230 KB), then rerun:\n"
              f"    curl -L -o {args.face_model} {YUNET_URL}\n"
              f"  Continuing without face capture.")
        return None
    try:
        fc = FaceCapture(args.face_model, args.capture_dir,
                         args.capture_min_conf, args.capture_cooldown,
                         args.capture_max)
        fc.start()
        print(f"[face-capture] on -> {args.capture_dir}/ "
              f"(every {args.detect_every} frames, conf>={args.capture_min_conf}, "
              f"cooldown {args.capture_cooldown}s)")
        return fc
    except Exception as e:
        print(f"[face-capture] failed to start ({e}); continuing without it.")
        return None


def resolve_display_size(arg):
    """Return (w, h) to scale output to, or None to leave frames untouched.

    macOS OpenCV fullscreen draws the frame at native size pinned to the
    top-left instead of filling the screen. Scaling each frame to the screen
    resolution first makes it fill. An explicit --display-size wins; otherwise
    we ask Tkinter (stdlib) for the screen size.
    """
    if arg:
        try:
            w, h = (int(v) for v in arg.lower().split("x"))
            return w, h
        except ValueError:
            raise SystemExit(f"--display-size must look like 1920x1080, got {arg!r}")
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        size = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
        print(f"Auto-detected display size: {size[0]}x{size[1]}")
        return size
    except Exception as e:
        print(f"Could not auto-detect display size ({e}); "
              "pass --display-size WxH. Displaying frames at native size.")
        return None


def parse_args():
    p = argparse.ArgumentParser(description="Real-time Depth Anything v2 art display")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    p.add_argument("--infer-size", type=int, default=392,
                   help="Resolution (longest side) for inference. Lower = faster.")
    p.add_argument("--colormap", default="inferno", choices=COLORMAP_ORDER)
    p.add_argument("--effect", default="plain", choices=EFFECT_ORDER,
                   help="Starting visual effect. Cycle live with the `e` key.")
    p.add_argument("--cycle", type=float, default=0.0, metavar="SECONDS",
                   help="Auto-advance to the next effect every N seconds (0=off).")
    p.add_argument("--display-size", default=None, metavar="WxH",
                   help="Output resolution, e.g. 1920x1080. Frames are scaled to "
                        "fill this so fullscreen isn't pinned top-left on macOS. "
                        "Omit to auto-detect the screen size via Tkinter.")
    p.add_argument("--coreml", default=None, metavar="PACKAGE",
                   help="Path to an exported .mlpackage (see convert_coreml.py). "
                        "Runs on the Neural Engine; ignores --model/--infer-size.")
    p.add_argument("--compute-units", default="ALL",
                   choices=["ALL", "CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY"],
                   help="Core ML compute units (only used with --coreml).")
    p.add_argument("--capture-faces", action="store_true",
                   help="Detect faces (YuNet, background thread) and save crops.")
    p.add_argument("--capture-dir", default="captures",
                   help="Where to write captured face crops.")
    p.add_argument("--face-model", default="face_detection_yunet_2023mar.onnx",
                   help="Path to the YuNet ONNX model (see README to download).")
    p.add_argument("--detect-every", type=int, default=5,
                   help="Run detection every Nth frame (lower = more often).")
    p.add_argument("--capture-cooldown", type=float, default=3.0,
                   help="Min seconds between saves of a face at the same spot.")
    p.add_argument("--capture-min-conf", type=float, default=0.7,
                   help="Minimum detector confidence to keep a face.")
    p.add_argument("--capture-max", type=int, default=500,
                   help="Keep at most this many crops (oldest deleted; 0=unlimited).")
    p.add_argument("--no-fps", action="store_true")
    p.add_argument("--mirror", action="store_true")
    p.add_argument("--smooth", type=float, default=0.0,
                   help="EMA weight on previous depth map (0=off, e.g. 0.5).")
    return p.parse_args()


def main():
    args = parse_args()

    display_size = resolve_display_size(args.display_size)
    capture = build_face_capture(args)

    if args.coreml:
        backend = CoreMLBackend(args.coreml, args.compute_units)
    else:
        backend = TorchBackend(args.model, args.infer_size)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {args.camera}. "
            "On macOS, grant camera permission to your terminal in "
            "System Settings > Privacy & Security > Camera."
        )

    win = "Depth Anything v2"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cmap_idx = COLORMAP_ORDER.index(args.colormap)
    show_fps = not args.no_fps
    mirror = args.mirror
    fullscreen = True
    prev_depth = None
    last = time.time()
    fps = 0.0

    effect_idx = EFFECT_ORDER.index(args.effect)
    fx_state = {name: {} for name in EFFECT_ORDER}
    start = time.time()
    last_cycle = start
    frame_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed; retrying...")
                continue
            if mirror:
                frame = cv2.flip(frame, 1)

            frame_count += 1
            if capture is not None and frame_count % args.detect_every == 0:
                capture.offer(frame)

            depth = backend.infer(frame)  # frame-sized float depth map

            # Normalize 0-255
            d_min, d_max = depth.min(), depth.max()
            norm = (depth - d_min) / (d_max - d_min + 1e-6)

            # Optional temporal smoothing
            if args.smooth > 0 and prev_depth is not None:
                norm = args.smooth * prev_depth + (1 - args.smooth) * norm
            prev_depth = norm

            now = time.time()

            # Auto-advance the effect on a timer, if enabled.
            if args.cycle > 0 and now - last_cycle >= args.cycle:
                effect_idx = (effect_idx + 1) % len(EFFECT_ORDER)
                last_cycle = now

            effect = EFFECT_ORDER[effect_idx]
            cmap = COLORMAPS[COLORMAP_ORDER[cmap_idx]]
            colored = EFFECTS[effect](norm.astype(np.float32), cmap,
                                      now - start, fx_state[effect])

            # FPS
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            if show_fps:
                cv2.putText(colored,
                            f"{fps:4.1f} fps  [{effect} / {COLORMAP_ORDER[cmap_idx]}]",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (255, 255, 255), 2, cv2.LINE_AA)

            if display_size is not None:
                colored = cv2.resize(colored, display_size)
            cv2.imshow(win, colored)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break
            elif key == ord("c"):
                cmap_idx = (cmap_idx + 1) % len(COLORMAP_ORDER)
            elif key == ord("e"):
                effect_idx = (effect_idx + 1) % len(EFFECT_ORDER)
                last_cycle = now
            elif key == ord("s"):
                show_fps = not show_fps
            elif key == ord("m"):
                mirror = not mirror
            elif key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    win, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                )
    finally:
        if capture is not None:
            capture.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
