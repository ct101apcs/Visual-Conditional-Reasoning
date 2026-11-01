"""Render an anchor's model input: candidates outlined in yellow with a black number box at their
interior mark, and (M-contour) the subject outlined in magenta with no number. M-text leaves the
subject unmarked, and M-mark numbers the subject like every other object. Same style as
figures/fig1_draft_running_example.png.

style="alt" is the alternative marker style of Table S9 (Set-of-Mark-like): each candidate gets a
semi-transparent colour fill and a number tag in its colour, with no outline. The subject is pointed
at exactly as in the default style, so only the candidate markers change.

  python tools/eval/render.py val 000000534270 15 contour /tmp/x.jpg [alt]
"""
import functools
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools", "pancap_viewer"))
from pancap import rgb2id  # noqa: E402

DATA = os.path.join(ROOT, "data")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
CAND_RGB, SUBJ_RGB = (255, 230, 0), (255, 0, 255)
CAND_PX, SUBJ_PX = 1, 2          # outline half-width (dilation iterations)
ALT_ALPHA = 0.4
ALT_COLORS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
              (210, 245, 60), (0, 128, 128), (170, 110, 40), (128, 0, 0), (0, 0, 128), (128, 128, 0)]  # no magenta


@functools.lru_cache(maxsize=None)
def _gt_dir(split):
    """The full generate.py build if present, else the Test-image subset shipped in the release bundle."""
    full = os.path.join(DATA, "anchor_gt", split)
    return full if os.path.exists(os.path.join(full, "index.json")) else os.path.join(DATA, "anchor_gt", "test_records", split)


@functools.lru_cache(maxsize=None)
def _index(split):
    return {r["key"]: (r["off"], r["len"]) for r in json.load(open(os.path.join(_gt_dir(split), "index.json")))}


@functools.lru_cache(maxsize=4096)
def record(split, key):
    off, n = _index(split)[key]
    with open(os.path.join(_gt_dir(split), "details.jsonl"), "rb") as f:
        f.seek(off)
        return json.loads(f.read(n))


def anchor(split, key, aid):
    return next(a for a in record(split, key)["anchors"] if a["id"] == aid)


def _outline(mask_ids, ids, px):
    """Boundary pixels of the given segment ids, grown by px."""
    m = mask_ids
    edge = np.zeros(m.shape, bool)
    edge[:, :-1] |= m[:, :-1] != m[:, 1:]
    edge[:, 1:] |= m[:, 1:] != m[:, :-1]
    edge[:-1, :] |= m[:-1, :] != m[1:, :]
    edge[1:, :] |= m[1:, :] != m[:-1, :]
    edge &= np.isin(m, list(ids))
    return ndimage.binary_dilation(edge, iterations=px) if px else edge


def render(split, key, aid, modality, out_path, style="outline"):
    rec, a = record(split, key), anchor(split, key, aid)
    img = np.array(Image.open(os.path.join(DATA, "images", f"{split}2017", key + ".jpg")).convert("RGB"))
    m = rgb2id(np.array(Image.open(os.path.join(DATA, "masks", "png", f"{split}2017", key + ".png")).convert("RGB")))
    numbered = a["cands"] + ([aid] if modality == "mark" else [])
    color = {}
    if style == "alt":
        f = img.astype(np.float32)
        for i, c in enumerate(numbered):
            color[c] = ALT_COLORS[i % len(ALT_COLORS)]
            sel = m == c
            f[sel] = f[sel] * (1 - ALT_ALPHA) + np.array(color[c]) * ALT_ALPHA
        img = f.astype(np.uint8)
    else:
        img[_outline(m, numbered, CAND_PX)] = CAND_RGB
    if modality == "contour":
        img[_outline(m, [aid], SUBJ_PX)] = SUBJ_RGB
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype(FONT, max(13, round(min(im.size) / 30)))
    for c in numbered:
        x, y = rec["marks"][str(c)]
        l, t, r, b = d.textbbox((x, y), str(c), font=font, anchor="mm")
        d.rectangle((l - 3, t - 2, r + 3, b + 2), fill=color.get(c, (0, 0, 0)), outline=(255, 255, 255) if c in color else None)
        d.text((x, y), str(c), font=font, fill=(255, 255, 255), anchor="mm")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    im.save(out_path, quality=95)
    return out_path


if __name__ == "__main__":
    s, k, aid, mod, out = sys.argv[1:6]
    print(render(s, k, int(aid), mod, out, *(sys.argv[6:7] or [])))
