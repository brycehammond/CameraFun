#!/usr/bin/env python3
"""
Convert Depth Anything v2 to a Core ML .mlpackage for the Apple Neural Engine.

Why: on an always-on M1 Mac Mini, running inference on the Neural Engine (ANE)
draws far less power than the GPU/MPS path and frees the GPU for compositing the
fullscreen display. The exported model is self-contained — it takes a raw RGB
image and bakes in the same rescale + ImageNet normalization the HF processor
applies, so the runtime doesn't need transformers at all.

Usage:
    python convert_coreml.py                      # Small @ 392, ALL compute units
    python convert_coreml.py --infer-size 518     # sharper, slower
    python convert_coreml.py --model depth-anything/Depth-Anything-V2-Base-hf
    python convert_coreml.py --no-verify          # skip the parity check

Output: depth_anything_v2_<size>.mlpackage  (override with --output)

This is the "stretch goal" path from CLAUDE.md — documented, not required. The
MPS path in depth_display.py works fine without it; pass --coreml <package> to
depth_display.py to use the exported model instead.
"""

import argparse
import contextlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation

# ImageNet stats used by the Depth Anything image processor.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DepthWrapper(nn.Module):
    """Wrap the HF model so it accepts a raw RGB image and returns depth.

    Core ML feeds an ImageType input as float pixels in 0-255 (NCHW, RGB).
    We replicate the processor's rescale (1/255) and ImageNet normalization
    inside the graph, then return the raw ``predicted_depth`` tensor so the
    trace has a single tensor output.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, image):  # image: (1, 3, H, W), RGB, 0-255
        x = image / 255.0
        x = (x - self.mean) / self.std
        depth = self.model(pixel_values=x).predicted_depth  # (1, H', W')
        return depth


@contextlib.contextmanager
def bicubic_to_bilinear():
    """Force F.interpolate's bicubic mode to bilinear during export.

    Dinov2 interpolates its positional embeddings with bicubic whenever the
    input grid differs from the pretrained 37x37 (518px). coremltools has no
    bicubic upsample op, so we swap it for bilinear just while exporting. The
    effect is a tiny smoothing of the position grid — invisible in the depth
    output for an art display. At the native 518 size no interpolation happens
    and this patch is a no-op.
    """
    orig = F.interpolate

    def patched(*a, **kw):
        if kw.get("mode") == "bicubic":
            kw["mode"] = "bilinear"
        return orig(*a, **kw)

    F.interpolate = patched
    try:
        yield
    finally:
        F.interpolate = orig


def coreml_input_size(infer_size):
    """Fixed, square input size (H == W) for the Core ML model.

    Core ML needs a static input shape, and Depth Anything's ViT backbone wants
    both sides to be a multiple of 14 (the patch size). We round ``infer_size``
    to the nearest multiple of 14 and use a square, so the exported model has a
    single fixed shape the runtime can fully optimize for the ANE. The display
    resizes each webcam frame to this square before inference; the colorized
    output is upscaled back to the display's aspect ratio afterwards.

    (The HF processor ignores small sizes and forces ~518 with aspect padding,
    which is why we bypass it and set the shape ourselves.)
    """
    side = max(14, round(infer_size / 14) * 14)
    return side, side


def parse_args():
    p = argparse.ArgumentParser(description="Convert Depth Anything v2 to Core ML")
    p.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    p.add_argument("--infer-size", type=int, default=518,
                   help="Square Core ML input size (rounded to a multiple of 14). "
                        "518 = model's native grid (best quality, no pos-embed "
                        "interpolation). Smaller = faster on the ANE.")
    p.add_argument("--output", default=None,
                   help="Output .mlpackage path (default: depth_anything_v2_<size>.mlpackage)")
    p.add_argument("--compute-units", default="ALL",
                   choices=["ALL", "CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY"],
                   help="Core ML compute units. ALL lets the runtime pick the ANE.")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip the torch-vs-CoreML parity check after conversion.")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        import coremltools as ct
    except ImportError:
        raise SystemExit(
            "coremltools is not installed. Run: pip install coremltools"
        )

    output = args.output or f"depth_anything_v2_{args.infer_size}.mlpackage"

    print(f"Loading {args.model} (CPU/float32 for tracing) ...")
    model = AutoModelForDepthEstimation.from_pretrained(args.model).eval()

    h, w = coreml_input_size(args.infer_size)
    print(f"Core ML input shape: (1, 3, {h}, {w})")

    wrapper = DepthWrapper(model).eval()
    example = torch.randint(0, 256, (1, 3, h, w), dtype=torch.float32)

    # torch.export handles the Dinov2 backbone's dynamic shape->int casts that
    # jit.trace chokes on inside coremltools. Shapes stay fixed (no dynamic
    # dims) since the display always runs at a single infer size.
    print("Exporting (torch.export) ...")
    with torch.inference_mode(), bicubic_to_bilinear():
        exported = torch.export.export(wrapper, (example,))
        # coremltools wants the ATEN/EDGE dialect; decompose out the TRAINING ops.
        exported = exported.run_decompositions({})

    print("Converting to Core ML (this can take a minute) ...")
    mlmodel = ct.convert(
        exported,
        inputs=[ct.ImageType(name="image", shape=(1, 3, h, w),
                             color_layout=ct.colorlayout.RGB,
                             scale=1.0, bias=[0.0, 0.0, 0.0])],
        outputs=[ct.TensorType(name="depth")],
        compute_units=getattr(ct.ComputeUnit, args.compute_units),
        minimum_deployment_target=ct.target.macOS13,
    )

    mlmodel.short_description = (
        f"Depth Anything v2 depth estimation ({args.model}). "
        f"Input: RGB image {w}x{h}. Output: per-pixel relative depth."
    )
    mlmodel.input_description["image"] = f"RGB image, {w}x{h}"
    mlmodel.output_description["depth"] = "Per-pixel relative depth (H x W)"

    mlmodel.save(output)
    print(f"Saved {output}")

    if args.no_verify:
        return

    print("Verifying Core ML output against torch ...")
    from PIL import Image
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)

    with torch.inference_mode():
        torch_depth = wrapper(
            torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        ).squeeze().numpy()

    cm = ct.models.MLModel(output)
    cm_out = cm.predict({"image": Image.fromarray(img)})["depth"]
    cm_depth = np.squeeze(cm_out)

    # Compare on normalized maps — that's what the display actually uses, and it
    # sidesteps any global scale/offset differences in raw depth units.
    def norm(d):
        return (d - d.min()) / (d.max() - d.min() + 1e-6)

    diff = np.abs(norm(torch_depth) - norm(cm_depth))
    print(f"  normalized depth  max |diff|={diff.max():.4f}  mean={diff.mean():.4f}")
    if diff.mean() < 0.02:
        print("  parity OK")
    else:
        print("  WARNING: larger-than-expected divergence; inspect before relying on it.")


if __name__ == "__main__":
    main()
