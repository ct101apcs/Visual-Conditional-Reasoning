"""COCONut-PanCap viewer: browse images, panoptic masks and grounded captions, with per-sample checks.

  python tools/pancap_viewer/server.py --port 8765
  then open http://localhost:8765 (VS Code forwards the port automatically over Remote-SSH).
"""
import argparse
import gzip
import json
import os
import threading
import traceback
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote

import numpy as np
from PIL import Image

import pancap

HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT_NAMES = ["val", "train", "o365val"]

_splits, _reports, _lock = {}, {}, threading.Lock()
CATEGORIES = pancap.load_categories()


def get_split(name):
    with _lock:
        if name not in _splits:
            _splits[name] = pancap.Split(name)
            rp = os.path.join(pancap.DATA_ROOT, "index", f"report_{name}.json")
            _reports[name] = json.load(open(rp)) if os.path.exists(rp) else None
        return _splits[name], _reports[name]


def available_splits():
    out = []
    for name in SPLIT_NAMES:
        if not os.path.exists(os.path.join(pancap.DATA_ROOT, "index", name + ".json")):
            continue
        split, report = get_split(name)
        summary = None
        if report:
            summary = {}
            for r in report.values():
                summary[r["status"]] = summary.get(r["status"], 0) + 1
        codes = sorted({c for r in report.values() for c in r["codes"]}) if report else []
        out.append(dict(name=name, count=len(split.keys), verified=report is not None, summary=summary, codes=codes))
    return out


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
            if url.path in ("/", "/index.html"):
                self.send_body(open(os.path.join(HERE, "index.html"), "rb").read(), "text/html; charset=utf-8")
            elif url.path == "/api/splits":
                self.send_json(available_splits())
            elif url.path == "/api/list":
                self.api_list(q)
            elif url.path == "/api/sample":
                split, _ = get_split(q["split"])
                if q["key"] not in split.samples:
                    return self.send_json({"error": "unknown key"}, 404)
                self.send_json(pancap.verify(split, q["key"], CATEGORIES))
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
        split, report = get_split(q["split"])
        keys = split.keys
        if q.get("q"):
            keys = [k for k in keys if q["q"] in k]
        if report and q.get("status"):
            keys = [k for k in keys if report.get(k, {}).get("status") == q["status"]]
        if report and q.get("code"):
            keys = [k for k in keys if q["code"] in report.get(k, {}).get("codes", [])]
        offset, limit = int(q.get("offset", 0)), int(q.get("limit", 200))
        items = [dict(key=k, **(report.get(k, {}) if report else {})) for k in keys[offset:offset + limit]]
        self.send_json(dict(total=len(keys), offset=offset, items=items))

    def serve_image(self, split_name, key):
        split, _ = get_split(split_name)
        try:
            p = split.image_path(key, fetch=True)
        except Exception as e:
            return self.send_json({"error": f"fetch failed: {e}"}, 502)
        if not p:
            return self.send_json({"error": "image unavailable"}, 404)
        self.send_body(open(p, "rb").read(), "image/jpeg", headers={"Cache-Control": "max-age=86400"})

    def serve_mask(self, split_name, key):
        """Segment-id map as raw little-endian uint16 (decoding PNG RGB in the browser is colour-managed)."""
        split, _ = get_split(split_name)
        p = split.mask_path(key)
        if not os.path.exists(p):
            return self.send_json({"error": "mask unavailable"}, 404)
        ids = pancap.rgb2id(np.array(Image.open(p).convert("RGB")))
        h, w = ids.shape
        if ids.max() > 0xFFFF:
            return self.send_json({"error": "segment id exceeds uint16"}, 500)
        self.send_body(ids.astype("<u2").tobytes(), "application/octet-stream",
                       headers={"X-Width": str(w), "X-Height": str(h), "Access-Control-Expose-Headers": "X-Width, X-Height"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    print(f"data root: {pancap.DATA_ROOT}", flush=True)
    print(f"splits: {[s['name'] for s in available_splits()]}")
    print(f"serving on http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
