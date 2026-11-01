"""ANCHOR ground-truth viewer: funnel, rule statistics and a per-image browser with GT overlays.

  python tools/anchor_gt/server.py                 # http://127.0.0.1:8778
  python tools/anchor_gt/server.py --port 9000
  python tools/anchor_gt/server.py --host 0.0.0.0  # only if you really want to expose it

Then open http://127.0.0.1:8778 (simple view) or /detailed (every statistic, Q-matrix, design checks) (VS Code forwards the port automatically over Remote-SSH).
Three tabs: Funnel & dataset, Rules, Browse. Links are stable: #tab=browse&split=val&key=000000534270&a=15&r=T11
Keyboard in Browse: Left/Right = previous/next image in the filtered list.

Reads data/anchor_gt/{summary.json, test_draft.jsonl, <split>/index.json, <split>/details.jsonl} (written by
generate.py) plus the image/mask/caption files through tools/pancap_viewer/pancap.py. A split whose anchor_gt files
are missing is simply not offered; restart the server after generating it (train appears at the same paths).

API (all GET, JSON unless noted):
  /api/summary                       summary.json + the splits found on disk
  /api/filters?split=                option values and counts for the Browse filters
  /api/list?split=&stage=&io=&io_stuff=&setting=&food=&people=&type=&cat=&tid=&text_ok=1&draft=1&q=&offset=&limit=
  /api/image?split=&key=             details record + caption tokens + test_draft items of that image
  /api/render?tid=&para=&modality=&phrase=&sid=   exact prompts: direct (render_direct), probe_all (the shared
                                     all-primitive probe, with `touches` when tid is a CLOSURE template) and probe
                                     (render_probe: rule-specific, Lite equivalence check only)
  /image/<split>/<key>               JPEG;  /mask/<split>/<key>  raw little-endian uint16 ids (X-Width/X-Height)
"""
import argparse
import collections
import functools
import gzip
import json
import os
import sys
import threading
import traceback
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools", "pancap_viewer"))
sys.path.insert(0, HERE)
import pancap  # noqa: E402
import rules as R  # noqa: E402

OUT = os.path.join(pancap.DATA_ROOT, "anchor_gt")
SPLIT_NAMES = ["val", "train"]
STAGE_ORDER = ["thing", "size", "phrase_unique", "not_group", "not_mismatch", "clutter", "rule_kept"]

_lock = threading.RLock()
_cache = {}


def cached(name, fn):
    with _lock:
        if name not in _cache:
            _cache[name] = fn()
        return _cache[name]


def available_splits():
    return [s for s in SPLIT_NAMES
            if os.path.exists(os.path.join(OUT, s, "index.json")) and os.path.exists(os.path.join(OUT, s, "details.jsonl"))]


def load_summary():
    p = os.path.join(OUT, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


class GT:
    """One split: the compact index (kept in memory) and random access into details.jsonl."""

    def __init__(self, name):
        self.name = name
        self.rows = json.load(open(os.path.join(OUT, name, "index.json")))
        self.by_key = {r["key"]: r for r in self.rows}
        self.details_path = os.path.join(OUT, name, "details.jsonl")
        try:
            self.pc = pancap.Split(name)
        except Exception:
            self.pc = None
        self._filters = None

    def record(self, key):
        r = self.by_key[key]
        with open(self.details_path, "rb") as f:
            f.seek(r["off"])
            return json.loads(f.read(r["len"]))

    def filters(self):
        if self._filters is None:
            c = collections.defaultdict(collections.Counter)
            for r in self.rows:
                c["stage"][r["stage"]] += 1
                c["io"][str(r["io"])] += 1
                c["io_stuff"][str(r["io_stuff"])] += 1
                for s in r["settings"] or ["(none)"]:
                    c["setting"][s] += 1
                for t in r["types"]:
                    c["type"][t] += 1
                for t in r["cats"]:
                    c["cat"][t] += 1
                for t in r["rules"]:
                    c["tid"][t] += 1
            self._filters = {k: sorted(v.items(), key=lambda kv: (-kv[1], kv[0])) for k, v in c.items()}
            self._filters["tid"].sort()
            self._filters["stage"].sort(key=lambda kv: STAGE_ORDER.index(kv[0]) if kv[0] in STAGE_ORDER else -1)
            self._filters["n"] = len(self.rows)
        return self._filters


def get_gt(name):
    if name not in available_splits():
        return None
    return cached("gt:" + name, lambda: GT(name))


def draft_items():
    def load():
        by = collections.defaultdict(list)
        p = os.path.join(OUT, "test_draft.jsonl")
        if os.path.exists(p):
            for line in open(p):
                if line.strip():
                    it = json.loads(line)
                    by[(it["split"], it["key"])].append(it)
        return by
    return cached("draft", load)


def filter_rows(gt, q):
    rows = gt.rows
    draft = draft_items()
    stage, tid = q.get("stage") or "all", q.get("tid")
    want = []
    if stage == "survived":
        want.append(lambda r: r["stage"] == "rule_kept")
    elif stage == "rejected":
        want.append(lambda r: r["stage"] != "rule_kept")
    elif stage != "all":
        want.append(lambda r: r["stage"] == stage)
    if q.get("io"):
        want.append(lambda r: str(r["io"]) == q["io"])
    if q.get("io_stuff"):
        want.append(lambda r: str(r["io_stuff"]) == q["io_stuff"])
    if q.get("setting"):
        s = q["setting"]
        want.append((lambda r: not r["settings"]) if s == "(none)" else (lambda r: s in r["settings"]))
    if q.get("food") in ("0", "1"):
        want.append(lambda r: bool(r["food"]) == (q["food"] == "1"))
    if q.get("people") in ("0", "1"):
        want.append(lambda r: bool(r["people"]) == (q["people"] == "1"))
    if q.get("type"):
        want.append(lambda r: q["type"] in r["types"])
    if q.get("cat"):
        want.append(lambda r: q["cat"] in r["cats"])
    if tid:
        want.append(lambda r: tid in r["rules"])
    if q.get("text_ok") == "1":
        want.append(lambda r: r["n_text"] > 0)
    if q.get("draft") == "1":
        if tid:
            want.append(lambda r: any(it["tid"] == tid for it in draft.get((gt.name, r["key"]), ())))
        else:
            want.append(lambda r: (gt.name, r["key"]) in draft)
    if q.get("q"):
        want.append(lambda r: q["q"] in r["key"])
    return [r for r in rows if all(f(r) for f in want)]


@functools.lru_cache(maxsize=32)
def mask_bytes(split, key):
    gt = get_gt(split)
    p = gt.pc.mask_path(key)
    if not os.path.exists(p):
        return None
    ids = pancap.rgb2id(np.array(Image.open(p).convert("RGB")))
    if ids.max() > 0xFFFF:
        raise ValueError("segment id exceeds uint16")
    return ids.shape[1], ids.shape[0], ids.astype("<u2").tobytes()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send_body(self, body, ctype, status=200, headers=None):
        if "gzip" in self.headers.get("Accept-Encoding", "") and len(body) > 1024 and not ctype.startswith("image/"):
            body = gzip.compress(body, 5)
            headers = dict(headers or {}, **{"Content-Encoding": "gzip"})
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, obj, status=200):
        self.send_body(json.dumps(obj).encode(), "application/json", status)

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        parts = [unquote(p) for p in url.path.strip("/").split("/")]
        try:
            if url.path in ("/", "/index.html", "/viewer.html"):
                self.send_body(open(os.path.join(HERE, "viewer.html"), "rb").read(), "text/html; charset=utf-8")
            elif url.path in ("/detailed", "/detailed/"):
                self.send_body(open(os.path.join(HERE, "viewer_detailed.html"), "rb").read(), "text/html; charset=utf-8")
            elif url.path == "/api/summary":
                self.send_json(dict(summary=load_summary(), splits=available_splits(),
                                    split_sizes={s: len(get_gt(s).rows) for s in available_splits()},
                                    draft_n=sum(len(v) for v in draft_items().values()),
                                    modalities=R.MODALITIES, predicates=R.PREDICATES))
            elif url.path == "/api/filters":
                gt = get_gt(q.get("split", ""))
                self.send_json(gt.filters() if gt else {"error": "unknown split"}, 200 if gt else 404)
            elif url.path == "/api/list":
                self.api_list(q)
            elif url.path == "/api/image":
                self.api_image(q)
            elif url.path == "/api/render":
                self.api_render(q)
            elif parts[0] == "image" and len(parts) == 3:
                self.serve_image(*parts[1:])
            elif parts[0] == "mask" and len(parts) == 3:
                self.serve_mask(*parts[1:])
            else:
                self.send_json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            traceback.print_exc()
            self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)

    def api_list(self, q):
        gt = get_gt(q.get("split", ""))
        if not gt:
            return self.send_json({"error": "unknown split"}, 404)
        rows = filter_rows(gt, q)
        offset, limit = max(0, int(q.get("offset", 0))), min(1000, int(q.get("limit", 100)))
        draft = draft_items()
        items = [dict(key=r["key"], stage=r["stage"], n_anchors=r["n_anchors"], n_text=r["n_text"],
                      n_rules=len(r["rules"]), in_draft=len(draft.get((gt.name, r["key"]), ())))
                 for r in rows[offset:offset + limit]]
        self.send_json(dict(total=len(rows), offset=offset, items=items))

    def api_image(self, q):
        gt = get_gt(q.get("split", ""))
        if not gt or q.get("key") not in gt.by_key:
            return self.send_json({"error": "unknown split/key"}, 404)
        key = q["key"]
        rec = gt.record(key)
        caption, tokens = None, []
        if gt.pc and os.path.exists(gt.pc.caption_path(key)):
            caption = open(gt.pc.caption_path(key), encoding="utf-8", errors="replace").read()
            tokens = pancap.parse_caption(caption)
        self.send_json(dict(rec=rec, caption=caption, tokens=tokens, draft=draft_items().get((gt.name, key), []),
                            image_local=bool(gt.pc and gt.pc.image_path(key))))

    def api_render(self, q):
        tid, mod = q.get("tid"), q.get("modality", "contour")
        if tid not in R.TEMPLATE_BY_ID or mod not in R.MODALITIES:
            return self.send_json({"error": "bad tid/modality"}, 400)
        para = int(q.get("para", 0))
        if not 0 <= para < len(R.TEMPLATE_BY_ID[tid]["text"]):
            return self.send_json({"error": "bad para"}, 400)
        phrase = q.get("phrase")
        sid = int(q["sid"]) if q.get("sid") else None
        if mod == "text" and not phrase:
            return self.send_json({"error": "text modality needs phrase"}, 400)
        if mod == "mark" and sid is None:
            return self.send_json({"error": "mark modality needs sid"}, 400)
        touches = bool(R.TEMPLATE_BY_ID[tid].get("closure"))
        self.send_json(dict(direct=R.render_direct(tid, para, mod, phrase, sid),
                            probe_all=R.render_probe_all(mod, phrase, sid, touches=touches), touches=touches,
                            probe=R.render_probe(tid, mod, phrase, sid)))

    def serve_image(self, split_name, key):
        gt = get_gt(split_name)
        if not gt or not gt.pc:
            return self.send_json({"error": "unknown split"}, 404)
        try:
            p = gt.pc.image_path(key, fetch=True)
        except Exception as e:
            return self.send_json({"error": f"fetch failed: {e}"}, 502)
        if not p:
            return self.send_json({"error": "image unavailable"}, 404)
        self.send_body(open(p, "rb").read(), "image/jpeg", headers={"Cache-Control": "max-age=86400"})

    def serve_mask(self, split_name, key):
        gt = get_gt(split_name)
        if not gt or not gt.pc:
            return self.send_json({"error": "unknown split"}, 404)
        m = mask_bytes(split_name, key)
        if m is None:
            return self.send_json({"error": "mask unavailable"}, 404)
        w, h, body = m
        self.send_body(body, "application/octet-stream",
                       headers={"X-Width": str(w), "X-Height": str(h), "Access-Control-Expose-Headers": "X-Width, X-Height"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8778)
    args = ap.parse_args()
    print(f"data root: {OUT}", flush=True)
    for s in SPLIT_NAMES:
        print(f"  split {s}: {'found, ' + str(len(get_gt(s).rows)) + ' images' if s in available_splits() else 'not generated yet (restart once it exists)'}", flush=True)
    print(f"serving on http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
