"""Shared logic: dataset index access, caption tag parsing and per-sample verification."""
import json
import os
import re
import urllib.request

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.environ.get("PANCAP_DATA", os.path.join(REPO, "data"))

# Order matters: <3>text</3> must win over the bare <3> alternative.
TAG_RE = re.compile(
    r"<(?P<xid>\d+)>(?P<xtext>.*?)</(?P=xid)>"
    r"|<\s*(?P<ids>\d+(?:\s*,\s*\d+)*)\s*:\s*(?P<text>[^<>]*?)\s*>"
    r"|<\s*(?P<bare>\d+(?:\s*,\s*\d+)*)\s*>"
    r"|<(?P<other>[^<>]*)>",
    re.S,
)


def parse_caption(text):
    """Split a caption into text / tag tokens. Tag fmt is one of std, xml, bare, malformed."""
    tokens, pos = [], 0
    for m in TAG_RE.finditer(text):
        if m.start() > pos:
            tokens.append(dict(type="text", text=text[pos:m.start()]))
        if m.group("xid") is not None:
            tok = dict(ids=[int(m.group("xid"))], text=m.group("xtext"), fmt="xml")
        elif m.group("ids") is not None:
            tok = dict(ids=[int(x) for x in m.group("ids").split(",")], text=m.group("text"), fmt="std")
        elif m.group("bare") is not None:
            tok = dict(ids=[int(x) for x in m.group("bare").split(",")], text="", fmt="bare")
        else:
            tok = dict(ids=[], text=m.group("other"), fmt="malformed")
        tok.update(type="tag", raw=m.group(0))
        tokens.append(tok)
        pos = m.end()
    if pos < len(text):
        tokens.append(dict(type="text", text=text[pos:]))
    return tokens


def rgb2id(arr):
    arr = arr.astype(np.int64)
    return arr[..., 0] + 256 * arr[..., 1] + 256 * 256 * arr[..., 2]


class Split:
    def __init__(self, name, root=DATA_ROOT):
        self.name, self.root = name, root
        meta = json.load(open(os.path.join(root, "index", name + ".json")))
        self.samples = meta["samples"]
        self.keys = sorted(self.samples)
        self.has_caption = set(meta["has_caption"])
        self.caption_dir = os.path.join(root, meta["captions"])
        self.mask_dir = os.path.join(root, meta["mask_dir"])
        self.image_dir = os.path.join(root, meta["image_dir"])

    def caption_path(self, key):
        return os.path.join(self.caption_dir, key + ".txt")

    def mask_path(self, key):
        return os.path.join(self.mask_dir, key + ".png")

    def image_path(self, key, fetch=False):
        """Local image path, or None. With fetch=True, download from the sample's URL into the cache."""
        p = os.path.join(self.image_dir, key + ".jpg")
        if os.path.exists(p):
            return p
        meta = self.samples.get(key)
        if not fetch or not meta or not meta.get("url"):
            return None
        os.makedirs(self.image_dir, exist_ok=True)
        tmp = p + ".part"
        with urllib.request.urlopen(meta["url"], timeout=60) as r, open(tmp, "wb") as f:
            f.write(r.read())
        os.replace(tmp, p)
        return p

    def image_fetchable(self, key):
        meta = self.samples.get(key)
        return bool(meta and meta.get("url"))


def verify(split, key, categories=None, fetch_image=False):
    """Run all consistency checks for one sample. Returns dict(issues=[...], tokens, segments, ...)."""
    issues = []

    def add(level, code, msg):
        issues.append(dict(level=level, code=code, msg=msg))

    meta = split.samples.get(key)
    caption = None
    if key in split.has_caption:
        caption = open(split.caption_path(key), encoding="utf-8", errors="replace").read()
    else:
        add("error", "missing_caption", "No caption file for this sample")
    if meta is None:
        add("error", "missing_mask", "No panoptic mask / segments_info for this sample")

    tokens = parse_caption(caption) if caption else []
    tags = [t for t in tokens if t["type"] == "tag"]
    referenced = {i for t in tags for i in t["ids"]}

    for fmt, level, msg in [("xml", "warn", "XML-style tag(s) like <3>text</3>"),
                            ("bare", "warn", "Tag(s) with ids but no description, e.g. <11>"),
                            ("malformed", "warn", "Tag(s) without a valid id list")]:
        bad = [t["raw"] for t in tags if t["fmt"] == fmt]
        if bad:
            add(level, "tag_" + fmt, f"{msg}: {len(bad)} — " + ", ".join(b[:40] for b in bad[:4]))
    if caption and caption.count("<") != caption.count(">"):
        add("warn", "unbalanced_brackets", f"{caption.count('<')} '<' vs {caption.count('>')} '>'")

    segments, mask_hw, image_hw = [], None, None
    if meta is not None:
        seg_ids = {s["id"] for s in meta["segments"]}
        pixel_area = {}
        mpath = split.mask_path(key)
        if os.path.exists(mpath):
            ids = rgb2id(np.array(Image.open(mpath).convert("RGB")))
            mask_hw = ids.shape
            u, c = np.unique(ids, return_counts=True)
            pixel_area = dict(zip(u.tolist(), c.tolist()))
            if mask_hw != (meta["height"], meta["width"]):
                add("error", "mask_size_mismatch",
                    f"Mask is {mask_hw[1]}x{mask_hw[0]}, metadata says {meta['width']}x{meta['height']}")
            unlisted = sorted(i for i in pixel_area if i != 0 and i not in seg_ids)
            if unlisted:
                add("error", "mask_unlisted_id", f"Mask contains ids not in segments_info: {unlisted[:10]}")
        else:
            add("error", "missing_mask_png", f"Mask PNG not found: {os.path.relpath(mpath, split.root)}")

        unknown = sorted(referenced - seg_ids - {0})
        if 0 in referenced and 0 not in seg_ids:
            add("error", "tag_void_id", "Caption references id 0, the unlabeled (void) region")
        if unknown:
            add("error", "tag_unknown_id", f"Caption references segment ids not in the mask: {unknown}")
        if pixel_area:
            empty = sorted(i for i in referenced & seg_ids if pixel_area.get(i, 0) == 0)
            if empty:
                add("warn", "tag_empty_segment", f"Referenced segments with 0 pixels in mask: {empty}")
            area_off = [s["id"] for s in meta["segments"]
                        if "area" in s and s["area"] != pixel_area.get(s["id"], 0)]
            if area_off:
                add("warn", "segment_area_mismatch",
                    f"segments_info area differs from mask pixel count for ids {area_off[:10]}")

        for s in meta["segments"]:
            cat = categories.get(s["category_id"], {}) if categories else {}
            segments.append(dict(id=s["id"], category_id=s["category_id"], category=cat.get("name", "?"),
                                 isthing=s.get("isthing"), pixels=pixel_area.get(s["id"]),
                                 referenced=s["id"] in referenced))
        if caption is not None and segments:
            visible = [s for s in segments if s["pixels"]] or segments
            uncovered = [s["id"] for s in visible if not s["referenced"]]
            if uncovered:
                add("info", "uncovered_segments",
                    f"{len(uncovered)}/{len(visible)} visible segments not mentioned in caption")

    ipath = split.image_path(key, fetch=fetch_image)
    if ipath:
        with Image.open(ipath) as im:
            image_hw = (im.height, im.width)
        if mask_hw and image_hw != mask_hw:
            add("error", "image_size_mismatch",
                f"Image is {image_hw[1]}x{image_hw[0]}, mask is {mask_hw[1]}x{mask_hw[0]}")
    elif split.image_fetchable(key):
        add("info", "image_not_cached", "Image not downloaded yet (fetched on demand)")
    else:
        add("info", "missing_image", "Image not available locally")

    order = {"error": 0, "warn": 1, "info": 2}
    issues.sort(key=lambda i: order[i["level"]])
    status = issues[0]["level"] if issues and issues[0]["level"] != "info" else "ok"
    return dict(key=key, split=split.name, status=status, issues=issues, tokens=tokens, segments=segments,
                caption=caption, mask_hw=mask_hw, image_hw=image_hw,
                meta={k: v for k, v in (meta or {}).items() if k != "segments"})


def load_categories(root=DATA_ROOT):
    p = os.path.join(root, "index", "categories.json")
    return {c["id"]: c for c in json.load(open(p))} if os.path.exists(p) else {}
