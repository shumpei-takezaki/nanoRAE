"""README banner: the nanoRAE wordmark next to a mosaic of generated samples (tiles cut from docs/figures/samples.png).

Usage: uv run python scripts/make_banner.py [--out docs/figures/banner.png]
"""

import argparse
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 2560, 640                      # drawn at 2x; shown at 1280 px wide
BG0, BG1 = (15, 23, 42), (30, 41, 59)  # slate-900 -> slate-800
FONT = "/usr/share/fonts/truetype/noto/NotoSans-{}.ttf"


def tiles(path, cols=10):
    """Cut the tiles out of the make_grid image (32 px tiles, 2 px padding, `cols` per row, upscaled by an integer factor)."""
    g = Image.open(path).convert("RGB")
    k = g.width // (cols * 34 + 2)
    cell, off, size, rows = 34 * k, 2 * k, 32 * k, (g.height // k - 2) // 34
    return [g.crop((off + c * cell, off + r * cell, off + c * cell + size, off + r * cell + size)) for r in range(rows) for c in range(cols)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default="docs/figures/samples.png")
    p.add_argument("--out", default="docs/figures/banner.png")
    a = p.parse_args()

    # background: horizontal gradient
    x = np.linspace(0, 1, W)[None, :, None]
    bg = (np.array(BG0) * (1 - x) + np.array(BG1) * x).repeat(H, 0).astype(np.uint8)
    img = Image.fromarray(bg)

    # sample mosaic on the right
    ts = tiles(a.samples)
    random.Random(0).shuffle(ts)
    t, gap, rows, cols, x0 = 116, 14, 4, 10, 1300
    y0 = (H - rows * t - (rows - 1) * gap) // 2
    mosaic = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        mosaic.paste(ts[i].resize((t, t), Image.NEAREST), (x0 + c * (t + gap), y0 + r * (t + gap)))
    # fade the mosaic in from the left
    alpha = np.clip((np.arange(W) - x0) / 480, 0, 1) ** 1.5
    a_ch = np.array(mosaic.split()[3], dtype=np.float32) * alpha[None, :]
    mosaic.putalpha(Image.fromarray(a_ch.astype(np.uint8)))
    img.paste(mosaic, (0, 0), mosaic)

    # wordmark and tagline
    d = ImageDraw.Draw(img)
    big_l, big_b = ImageFont.truetype(FONT.format("Regular"), 210), ImageFont.truetype(FONT.format("Bold"), 210)
    small = ImageFont.truetype(FONT.format("Regular"), 52)
    xt, yt = 120, 140
    d.text((xt, yt), "nano", font=big_l, fill=(148, 163, 184))
    d.text((xt + d.textlength("nano", font=big_l), yt), "RAE", font=big_b, fill=(251, 191, 36))
    d.text((xt + 6, yt + 270), "Representation Autoencoders at small scale", font=small, fill=(226, 232, 240))

    # rounded corners
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, W - 1, H - 1), radius=48, fill=255)
    img.putalpha(mask)
    img.save(a.out, optimize=True)
    print("saved", a.out, img.size)


if __name__ == "__main__":
    main()
