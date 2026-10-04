"""Finish a cut-out for add_bird.py: finish.py cut.png out.png

add_bird's own downscale premultiplies into 8 bits, which corrupts the colour of
low-alpha pixels; the frame draws anything over alpha 24, so they print as a dotted
ring at the halo's outer edge. A plate handed over at or under the cap skips that
step, so: crop to the alpha, downscale here in floating point, and set every pixel
that is not fully opaque to the exact halo tone - the outer edge is halo by
construction, so nothing of the bird is touched.

A cut true to a faded scan prints washed out beside the style's plates, so the bird
is deepened by how pale its darkest ink is: untouched at luma 30 and under, a black
point of 40, gamma 1.1 and -5% saturation from 70 up. The halo tone maps to itself.
"""

import sys

import numpy as np
from PIL import Image

CAP = 1200
PAPER = (0xF0, 0xEC, 0xE5)

img = Image.open(sys.argv[1]).convert("RGBA")
arr = np.asarray(img).astype(np.float32)
alpha = arr[..., 3]
rows, cols = np.flatnonzero(alpha.any(1)), np.flatnonzero(alpha.any(0))
arr = arr[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
h, w = arr.shape[:2]
scale = min(1.0, CAP / max(w, h))
size = (max(1, round(w * scale)), max(1, round(h * scale)))

a = arr[..., 3] / 255.0
planes = [arr[..., c] * a for c in range(3)] + [arr[..., 3]]
small = [np.asarray(Image.fromarray(p, "F").resize(size, Image.Resampling.LANCZOS)) for p in planes]
out_a = np.clip(small[3], 0, 255)
safe = np.maximum(out_a / 255.0, 1e-6)
rgb = np.stack([np.clip(small[c] / safe, 0, 255) for c in range(3)], axis=-1)
out_a = out_a.round()

LUMA = np.array([0.299, 0.587, 0.114])
paper = np.array(PAPER, dtype=np.float64)
drawn = (out_a == 255) & (np.abs(rgb - paper).max(axis=2) > 12)
ink = float(np.percentile(rgb[drawn] @ LUMA, 1)) if drawn.any() else 0.0
fade = float(np.clip((ink - 30) / 40, 0, 1))
black, gamma, saturation = 40 * fade, 1 + 0.1 * fade, 1 - 0.05 * fade
rgb = np.where(rgb <= paper, paper * np.clip((rgb - black) / (paper - black), 0, 1) ** gamma, rgb)
grey = (rgb @ LUMA)[..., None]
boost = 1 + (saturation - 1) * np.clip((paper @ LUMA - grey) / 40, 0, 1)  # none near paper
rgb = np.clip(grey + boost * (rgb - grey), 0, 255)

rgb[out_a < 255] = PAPER
out = np.concatenate([rgb.round(), out_a[..., None]], axis=-1).astype(np.uint8)
Image.fromarray(out, "RGBA").save(sys.argv[2])
print(sys.argv[2], size, f"ink {ink:.0f}, graded {fade:.0%}")
