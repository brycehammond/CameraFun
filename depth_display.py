#!/usr/bin/env python3
"""
Depth Anything v2 — real-time colorized depth display for a TV.

Webcam -> Depth Anything v2 (MPS) -> colorized depth map -> fullscreen.

This is a working baseline. See CLAUDE.md for the full feature list to build out.
"""

import argparse
import os
import sys
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


# --- Neural style transfer ---------------------------------------------------
# Fast neural style (Johnson et al. 2016): a small feed-forward TransformerNet,
# one per style, that repaints the *RGB frame* in real time on MPS. Unlike the
# depth effects above, these ignore the depth map entirely — so when a style
# mode is active the depth model is skipped (running two nets/frame is too slow).
#
# Weights are the official pretrained checkpoints from the PyTorch examples repo
# (candy, mosaic, udnie, rain_princess). See the README for the one-time download.

STYLES = {
    "candy": "candy.pth",
    "mosaic": "mosaic.pth",
    "udnie": "udnie.pth",
    "rain": "rain_princess.pth",
}
STYLE_ORDER = list(STYLES.keys())


# --- MediaPipe vision modes --------------------------------------------------
# Person segmentation, pose skeletons, and face mesh via MediaPipe Tasks. Each
# needs a model bundle downloaded once (see README / the printed hint). Like the
# styles, these run on the RGB frame and skip depth inference while active.

MP_MODEL_FILES = {
    "segment": "selfie_segmenter.tflite",
    "pose": "pose_landmarker_lite.task",
    "face": "face_landmarker.task",
    "hand": "hand_landmarker.task",
}
MP_MODEL_URLS = {
    "segment": ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
                "selfie_segmenter/float16/latest/selfie_segmenter.tflite"),
    "pose": ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"),
    "face": ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/latest/face_landmarker.task"),
    "hand": ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task"),
}
# Each vision mode -> the model key it needs. A mode is only offered if its file
# is present, so partial downloads still enable whatever's available.
VISION_MODES = {
    "silhouette": "segment",  # person over an animated color field
    "pose": "pose",           # glowing skeleton on a dimmed frame
    "facemesh": "face",       # glowing face-mesh points
    "hands": "hand",          # glowing hand skeletons + fingertip light-painting
}
VISION_ORDER = list(VISION_MODES.keys())

# BlazePose 33-landmark topology (limbs + torso) for drawing the skeleton.
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]

# MediaPipe 21-landmark hand topology. Fingertips are 4/8/12/16/20; index tip
# (8) drives the light-painting trail.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]


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


def _transformer_net(torch):
    """Build a fresh TransformerNet (Johnson et al.) using the given torch module.

    Defined lazily so importing this file doesn't require torch (the Core ML
    path avoids it). Matches the pytorch/examples architecture exactly so the
    official pretrained checkpoints load without modification.
    """
    nn = torch.nn
    F = torch.nn.functional

    # NB: submodule names (reflection_pad, conv2d) must match the pretrained
    # checkpoints' state_dict keys, or load_state_dict fails.
    class ConvLayer(nn.Module):
        def __init__(self, in_c, out_c, k, stride):
            super().__init__()
            self.reflection_pad = nn.ReflectionPad2d(k // 2)
            self.conv2d = nn.Conv2d(in_c, out_c, k, stride)

        def forward(self, x):
            return self.conv2d(self.reflection_pad(x))

    class UpsampleConvLayer(nn.Module):
        def __init__(self, in_c, out_c, k, stride, upsample=None):
            super().__init__()
            self.upsample = upsample
            self.reflection_pad = nn.ReflectionPad2d(k // 2)
            self.conv2d = nn.Conv2d(in_c, out_c, k, stride)

        def forward(self, x):
            if self.upsample:
                x = F.interpolate(x, mode="nearest", scale_factor=self.upsample)
            return self.conv2d(self.reflection_pad(x))

    class ResidualBlock(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.conv1 = ConvLayer(c, c, 3, 1)
            self.in1 = nn.InstanceNorm2d(c, affine=True)
            self.conv2 = ConvLayer(c, c, 3, 1)
            self.in2 = nn.InstanceNorm2d(c, affine=True)
            self.relu = nn.ReLU()

        def forward(self, x):
            y = self.relu(self.in1(self.conv1(x)))
            y = self.in2(self.conv2(y))
            return x + y

    class TransformerNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = ConvLayer(3, 32, 9, 1)
            self.in1 = nn.InstanceNorm2d(32, affine=True)
            self.conv2 = ConvLayer(32, 64, 3, 2)
            self.in2 = nn.InstanceNorm2d(64, affine=True)
            self.conv3 = ConvLayer(64, 128, 3, 2)
            self.in3 = nn.InstanceNorm2d(128, affine=True)
            self.res1 = ResidualBlock(128)
            self.res2 = ResidualBlock(128)
            self.res3 = ResidualBlock(128)
            self.res4 = ResidualBlock(128)
            self.res5 = ResidualBlock(128)
            self.deconv1 = UpsampleConvLayer(128, 64, 3, 1, upsample=2)
            self.in4 = nn.InstanceNorm2d(64, affine=True)
            self.deconv2 = UpsampleConvLayer(64, 32, 3, 1, upsample=2)
            self.in5 = nn.InstanceNorm2d(32, affine=True)
            self.deconv3 = ConvLayer(32, 3, 9, 1)
            self.relu = nn.ReLU()

        def forward(self, x):
            y = self.relu(self.in1(self.conv1(x)))
            y = self.relu(self.in2(self.conv2(y)))
            y = self.relu(self.in3(self.conv3(y)))
            y = self.res1(y)
            y = self.res2(y)
            y = self.res3(y)
            y = self.res4(y)
            y = self.res5(y)
            y = self.relu(self.in4(self.deconv1(y)))
            y = self.relu(self.in5(self.deconv2(y)))
            return self.deconv3(y)

    return TransformerNet()


class StyleBackend:
    """Fast neural style transfer over the live RGB frame (MPS/CUDA/CPU).

    One pretrained TransformerNet per style; models are loaded lazily on first
    use so an unused style never costs memory. Inference runs at a reduced
    longest-side resolution and the result is upscaled back to frame size.
    """

    def __init__(self, styles_dir, infer_size, device=None):
        import re
        import torch
        self._torch = torch
        self._re = re
        self.device = device or pick_device()
        self.styles_dir = styles_dir
        self.infer_size = infer_size
        self._models = {}  # name -> loaded net

    def available(self):
        """Style names whose checkpoint files are present, in cycle order."""
        return [n for n in STYLE_ORDER
                if os.path.exists(os.path.join(self.styles_dir, STYLES[n]))]

    def _get(self, name):
        net = self._models.get(name)
        if net is not None:
            return net
        torch = self._torch
        path = os.path.join(self.styles_dir, STYLES[name])
        try:
            state = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:  # older torch without weights_only
            state = torch.load(path, map_location=self.device)
        # The 2017-era checkpoints carry deprecated InstanceNorm running stats
        # that modern InstanceNorm2d doesn't register — drop them so load is strict.
        for k in list(state.keys()):
            if self._re.search(r"in\d+\.running_(mean|var)$", k):
                del state[k]
        net = _transformer_net(torch).to(self.device).eval()
        net.load_state_dict(state)
        self._models[name] = net
        print(f"[style] loaded {name} on {self.device}")
        return net

    def stylize(self, frame_bgr, name):
        torch = self._torch
        h, w = frame_bgr.shape[:2]
        scale = self.infer_size / max(h, w)
        small = cv2.resize(frame_bgr, (max(1, int(w * scale)), max(1, int(h * scale))))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32)
        # Models expect RGB in 0-255 (ToTensor()*255), CHW, batched.
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            out = self._get(name)(t).clamp(0, 255)
        out = out.squeeze(0).permute(1, 2, 0).to("cpu").numpy().astype(np.uint8)
        bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return cv2.resize(bgr, (w, h))


def build_style_backend(args):
    """Construct a StyleBackend and its available styles, or (None, []) with a hint."""
    if args.no_style:
        return None, []
    try:
        sb = StyleBackend(args.styles_dir, args.style_size)
    except Exception as e:
        print(f"[style] disabled ({e}); continuing without style transfer.")
        return None, []
    avail = sb.available()
    if not avail:
        print(f"[style] no style checkpoints in {args.styles_dir}/ — style modes off.\n"
              f"  One-time download (~26 MB) to enable them:\n"
              f"    curl -L -o saved_models.zip "
              f'"https://www.dropbox.com/s/lrvwfehqdcxoza8/saved_models.zip?dl=1"\n'
              f"    unzip saved_models.zip   # creates saved_models/*.pth")
        return None, []
    print(f"[style] on -> {', '.join(avail)} @ {args.style_size}px "
          f"(cycle with `t`)")
    return sb, avail


class VisionBackend:
    """MediaPipe Tasks: person segmentation, pose skeletons, and face mesh.

    Each landmarker/segmenter is created lazily on first use (they're only built
    if that mode is actually shown). All run in IMAGE mode on the RGB frame and
    return a BGR image ready for display.
    """

    def __init__(self, models_dir, num_poses=4, num_faces=5, num_hands=4):
        import mediapipe as mp
        from mediapipe.tasks.python import vision as mpv
        self._mp = mp
        self._mpv = mpv
        self.models_dir = models_dir
        self.num_poses = num_poses
        self.num_faces = num_faces
        self.num_hands = num_hands
        self._pose = self._face = self._seg = self._hand = None
        self._trail = None  # decaying fingertip light-painting buffer

    def path(self, key):
        return os.path.join(self.models_dir, MP_MODEL_FILES[key])

    def available(self):
        """Vision mode names whose model file is present, in cycle order."""
        return [m for m in VISION_ORDER if os.path.exists(self.path(VISION_MODES[m]))]

    def _image(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                              data=np.ascontiguousarray(rgb))

    def _pose_lm(self):
        if self._pose is None:
            mpv = self._mpv
            self._pose = mpv.PoseLandmarker.create_from_options(
                mpv.PoseLandmarkerOptions(
                    base_options=self._mp.tasks.BaseOptions(
                        model_asset_path=self.path("pose")),
                    running_mode=mpv.RunningMode.IMAGE,
                    num_poses=self.num_poses))
        return self._pose

    def _face_lm(self):
        if self._face is None:
            mpv = self._mpv
            self._face = mpv.FaceLandmarker.create_from_options(
                mpv.FaceLandmarkerOptions(
                    base_options=self._mp.tasks.BaseOptions(
                        model_asset_path=self.path("face")),
                    running_mode=mpv.RunningMode.IMAGE,
                    num_faces=self.num_faces))
        return self._face

    def _seg_lm(self):
        if self._seg is None:
            mpv = self._mpv
            self._seg = mpv.ImageSegmenter.create_from_options(
                mpv.ImageSegmenterOptions(
                    base_options=self._mp.tasks.BaseOptions(
                        model_asset_path=self.path("segment")),
                    running_mode=mpv.RunningMode.IMAGE,
                    output_confidence_masks=True))
        return self._seg

    def _hand_lm(self):
        if self._hand is None:
            mpv = self._mpv
            self._hand = mpv.HandLandmarker.create_from_options(
                mpv.HandLandmarkerOptions(
                    base_options=self._mp.tasks.BaseOptions(
                        model_asset_path=self.path("hand")),
                    running_mode=mpv.RunningMode.IMAGE,
                    num_hands=self.num_hands))
        return self._hand

    def render(self, frame_bgr, mode, t):
        if mode == "pose":
            return self._render_pose(frame_bgr)
        if mode == "facemesh":
            return self._render_face(frame_bgr)
        if mode == "silhouette":
            return self._render_seg(frame_bgr, t)
        if mode == "hands":
            return self._render_hands(frame_bgr)
        return frame_bgr

    @staticmethod
    def _neon(base, marks, sigma):
        """Composite glowing marks (drawn on black) over a dimmed base frame."""
        glow = cv2.GaussianBlur(marks, (0, 0), sigma)
        out = base.astype(np.int16) + marks.astype(np.int16) + glow.astype(np.int16)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _render_pose(self, frame):
        h, w = frame.shape[:2]
        res = self._pose_lm().detect(self._image(frame))
        marks = np.zeros_like(frame)
        for lms in res.pose_landmarks:
            pts = [(int(l.x * w), int(l.y * h)) for l in lms]
            for a, b in POSE_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(marks, pts[a], pts[b], (0, 255, 255), 3, cv2.LINE_AA)
            for p in pts:
                cv2.circle(marks, p, 5, (255, 255, 255), -1, cv2.LINE_AA)
        return self._neon((frame * 0.25).astype(np.uint8), marks, 6)

    def _render_face(self, frame):
        h, w = frame.shape[:2]
        res = self._face_lm().detect(self._image(frame))
        marks = np.zeros_like(frame)
        for lms in res.face_landmarks:
            for l in lms:
                cv2.circle(marks, (int(l.x * w), int(l.y * h)), 1,
                           (0, 255, 180), -1, cv2.LINE_AA)
        return self._neon((frame * 0.35).astype(np.uint8), marks, 3)

    def _render_seg(self, frame, t):
        h, w = frame.shape[:2]
        res = self._seg_lm().segment(self._image(frame))
        mask = res.confidence_masks[0].numpy_view()  # HxW float, ~1 = person
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h))
        mask = mask[..., None]
        # Animated horizontal color field scrolling behind the person.
        ramp = ((np.arange(w, dtype=np.float32) / w * 255 + t * 40) % 256).astype(np.uint8)
        bg = cv2.applyColorMap(np.tile(ramp, (h, 1)), cv2.COLORMAP_TURBO)
        out = frame.astype(np.float32) * mask + bg.astype(np.float32) * (1 - mask)
        return out.astype(np.uint8)

    def _render_hands(self, frame):
        h, w = frame.shape[:2]
        res = self._hand_lm().detect(self._image(frame))
        # Persistent trail buffer: index fingertips paint into it and it decays,
        # so a waving hand leaves a glowing streak ("paint in the air").
        trail = self._trail
        if trail is None or trail.shape[:2] != (h, w):
            trail = np.zeros((h, w, 3), np.float32)
        trail *= 0.90
        marks = np.zeros_like(frame)
        for lms in res.hand_landmarks:
            pts = [(int(l.x * w), int(l.y * h)) for l in lms]
            for a, b in HAND_CONNECTIONS:
                cv2.line(marks, pts[a], pts[b], (0, 255, 255), 3, cv2.LINE_AA)
            for p in pts:
                cv2.circle(marks, p, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(trail, pts[8], 9, (60, 220, 255), -1, cv2.LINE_AA)  # index tip
        self._trail = trail
        base = (frame * 0.20).astype(np.int16)
        glow = cv2.GaussianBlur(marks, (0, 0), 6).astype(np.int16)
        out = base + trail.astype(np.int16) + marks.astype(np.int16) + glow
        return np.clip(out, 0, 255).astype(np.uint8)


def build_vision_backend(args):
    """Construct a VisionBackend + its available modes, or (None, []) with hints."""
    if args.no_vision:
        return None, []
    try:
        vb = VisionBackend(args.vision_dir, args.pose_count, args.face_count,
                           args.hand_count)
    except Exception as e:
        print(f"[vision] disabled ({e}); continuing without vision modes.")
        return None, []
    avail = vb.available()
    missing = [k for k in MP_MODEL_FILES if not os.path.exists(vb.path(k))]
    if missing:
        print(f"[vision] missing model bundles in {args.vision_dir}/ — "
              f"those modes are off. Download (once):")
        print(f"    mkdir -p {args.vision_dir}")
        for k in missing:
            print(f"    curl -L -o {vb.path(k)} {MP_MODEL_URLS[k]}")
    if not avail:
        return None, []
    print(f"[vision] on -> {', '.join(avail)} (cycle with `v`)")
    return vb, avail


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

    The Tk probe runs in a short-lived subprocess: creating and destroying a
    Tk root in this process leaves the Tcl/Tk runtime half-torn-down, and once
    a Core ML model is loaded, OpenCV's macOS window backend re-enters that
    dead interpreter and aborts (SIGABRT: "Tcl_FindHashEntry on deleted
    table"). Probing out-of-process keeps our GUI runtime pristine.
    """
    if arg:
        try:
            w, h = (int(v) for v in arg.lower().split("x"))
            return w, h
        except ValueError:
            raise SystemExit(f"--display-size must look like 1920x1080, got {arg!r}")
    import subprocess
    import sys
    probe = (
        "import tkinter\n"
        "r = tkinter.Tk(); r.withdraw()\n"
        "print(r.winfo_screenwidth(), r.winfo_screenheight())\n"
        "r.destroy()\n"
    )
    try:
        out = subprocess.run([sys.executable, "-c", probe],
                             capture_output=True, text=True, timeout=15)
        if out.returncode != 0:
            raise RuntimeError((out.stderr or "").strip() or "Tk probe failed")
        w, h = (int(v) for v in out.stdout.split())
        print(f"Auto-detected display size: {w}x{h}")
        return (w, h)
    except Exception as e:
        print(f"Could not auto-detect display size ({e}); "
              "pass --display-size WxH. Displaying frames at native size.")
        return None


def keep_display_awake():
    """Hold a macOS power assertion so the screen won't sleep or lock while we run.

    Spawns `caffeinate` bound to our own PID (`-w`), so it self-terminates if we
    ever exit without cleaning up. -d keeps the display awake — this is what stops
    the lock, and it also holds the whole system up (via powerd) while the display
    is on, which is exactly what an always-on installation wants.
    Returns the Popen handle (terminate it on teardown), or None if unavailable.
    """
    if sys.platform != "darwin":
        return None
    import subprocess
    try:
        proc = subprocess.Popen(
            ["caffeinate", "-d", "-w", str(os.getpid())],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print("[power] caffeinate holding display awake (no sleep/lock while running)")
        return proc
    except FileNotFoundError:
        print("[power] caffeinate not found; screen may sleep/lock while running.")
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
    p.add_argument("--styles-dir", default="saved_models",
                   help="Directory holding the fast-neural-style .pth checkpoints.")
    p.add_argument("--style-size", type=int, default=640,
                   help="Longest side for style-transfer inference. Lower = faster "
                        "(480 ~83fps, 640 ~52fps, 720 ~40fps on M1 MPS).")
    p.add_argument("--no-style", action="store_true",
                   help="Disable neural style-transfer modes even if weights exist.")
    p.add_argument("--vision-dir", default="mediapipe_models",
                   help="Directory holding MediaPipe .task/.tflite model bundles.")
    p.add_argument("--pose-count", type=int, default=4,
                   help="Max simultaneous people for the pose skeleton mode.")
    p.add_argument("--face-count", type=int, default=5,
                   help="Max simultaneous faces for the face-mesh mode.")
    p.add_argument("--hand-count", type=int, default=4,
                   help="Max simultaneous hands for the hand-tracking mode.")
    p.add_argument("--no-vision", action="store_true",
                   help="Disable MediaPipe vision modes even if models exist.")
    p.add_argument("--verbose-mediapipe", action="store_true",
                   help="Keep MediaPipe's noisy native C++ logging (silenced by "
                        "default when vision modes are active).")
    p.add_argument("--fps", action="store_true",
                   help="Show the FPS overlay (off by default; toggle live with `s`).")
    p.add_argument("--mirror", action="store_true")
    p.add_argument("--smooth", type=float, default=0.0,
                   help="EMA weight on previous depth map (0=off, e.g. 0.5).")
    p.add_argument("--allow-sleep", action="store_true",
                   help="Let the display sleep/lock normally. By default (macOS) "
                        "we hold a caffeinate assertion so the screen stays on.")
    return p.parse_args()


def quiet_native_stderr():
    """Silence MediaPipe's native C++ log spam without hiding Python errors.

    MediaPipe 0.10.x logs via absl at the C++ level (the `I0000/W0000/E0000`
    lines: gl_context, feedback-manager, and the recurring clearcut-telemetry
    failures), which ignores `GLOG_minloglevel` and absl's Python `set_verbosity`.
    The only thing that works is redirecting OS-level stderr (fd 2). To keep
    Python tracebacks visible, we first repoint `sys.stderr` at a dup of the real
    stderr, then send fd 2 to /dev/null — so native writes vanish while Python's
    own errors (and our prints) still reach the terminal / launchd log.
    """
    real = os.dup(2)
    sys.stderr = os.fdopen(real, "w", buffering=1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)


def main():
    args = parse_args()

    caffeinate = None if args.allow_sleep else keep_display_awake()
    display_size = resolve_display_size(args.display_size)
    capture = build_face_capture(args)

    if args.coreml:
        backend = CoreMLBackend(args.coreml, args.compute_units)
    else:
        backend = TorchBackend(args.model, args.infer_size)

    style_backend, avail_styles = build_style_backend(args)
    vision_backend, avail_vision = build_vision_backend(args)

    # MediaPipe's native logging is very chatty; hush it for always-on use unless
    # asked to keep it. Only matters when a vision backend actually loaded.
    if vision_backend is not None and not args.verbose_mediapipe:
        print("[vision] silencing MediaPipe native logs (--verbose-mediapipe to keep)")
        quiet_native_stderr()

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
    show_fps = args.fps
    mirror = args.mirror
    fullscreen = True
    prev_depth = None
    last = time.time()
    fps = 0.0

    # Independent mode groups, each bound to a key and remembering its own place:
    #   e -> depth effects, t -> neural styles, v -> MediaPipe vision modes.
    # `active_group` is what's on screen; `pos[group]` is each group's position.
    # Pressing a group's key resumes it (or advances if already active), so you
    # can bounce between them without losing your spot. Empty groups (missing
    # weights/models) are simply unreachable.
    groups = [
        ("depth", EFFECT_ORDER, ord("e")),
        ("style", avail_styles, ord("t")),
        ("vision", avail_vision, ord("v")),
    ]
    members = {g: lst for g, lst, _ in groups}
    keymap = {key: g for g, _, key in groups}
    group_order = [g for g, _, _ in groups]
    pos = {g: 0 for g in group_order}
    pos["depth"] = EFFECT_ORDER.index(args.effect)
    active_group = "depth"

    fx_state = {name: {} for name in EFFECT_ORDER}
    start = time.time()
    last_cycle = start
    frame_count = 0

    def advance_group(g):
        """Next non-empty group after g in group_order (wraps; g if none else)."""
        for step in range(1, len(group_order) + 1):
            ng = group_order[(group_order.index(g) + step) % len(group_order)]
            if members[ng]:
                return ng
        return g

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

            now = time.time()

            # Auto-advance on a timer, if enabled: sweep every mode in the active
            # group, then move to the next non-empty group — hands-off rotation.
            if args.cycle > 0 and now - last_cycle >= args.cycle:
                pos[active_group] += 1
                if pos[active_group] >= len(members[active_group]):
                    pos[active_group] = 0
                    active_group = advance_group(active_group)
                last_cycle = now

            kind = active_group
            name = members[kind][pos[kind]]
            if kind == "style":
                # Style modes ignore depth — repaint the RGB frame directly.
                colored = style_backend.stylize(frame, name)
            elif kind == "vision":
                # Vision modes also work on the RGB frame, no depth needed.
                colored = vision_backend.render(frame, name, now - start)
            else:
                depth = backend.infer(frame)  # frame-sized float depth map

                # Normalize 0-255
                d_min, d_max = depth.min(), depth.max()
                norm = (depth - d_min) / (d_max - d_min + 1e-6)

                # Optional temporal smoothing
                if args.smooth > 0 and prev_depth is not None:
                    norm = args.smooth * prev_depth + (1 - args.smooth) * norm
                prev_depth = norm

                cmap = COLORMAPS[COLORMAP_ORDER[cmap_idx]]
                colored = EFFECTS[name](norm.astype(np.float32), cmap,
                                        now - start, fx_state[name])

            # FPS
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            if show_fps:
                label = (f"{name} / {COLORMAP_ORDER[cmap_idx]}"
                         if kind == "depth" else f"{kind}:{name}")
                cv2.putText(colored,
                            f"{fps:4.1f} fps  [{label}]",
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
            elif key in keymap and members[keymap[key]]:
                # A group key (e/t/v): switch to that group, resuming where you
                # left off, or advance within it if it's already active.
                g = keymap[key]
                if active_group == g:
                    pos[g] = (pos[g] + 1) % len(members[g])
                active_group = g
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
        if caffeinate is not None:
            caffeinate.terminate()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
