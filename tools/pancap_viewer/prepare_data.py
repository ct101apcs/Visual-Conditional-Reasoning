"""Unpack COCONut-PanCap captions, panoptic masks and COCO images into a viewer-friendly layout.

Layout produced under --data_root (default: <repo>/data):
  coconut_pancap/{caption_train2017,caption_val2017,pancap_val}/*.txt   captions
  masks/png/{train2017,val2017}/<image_id>.png                          COCONut-S / relabeled COCO val masks
  masks/coconut_val/panoptic_o365val_v3/<objects365 name>.png           COCONut val (Objects365) masks
  images/val2017/<image_id>.jpg                                         COCO val2017 images
  index/{train,val,o365val}.json                                        per-split metadata index

Every step is idempotent; re-run it after more downloads finish.
"""
import argparse
import glob
import io
import json
import os
import tarfile

import pyarrow.parquet as pq

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SPLITS = {
    "train": dict(captions="coconut_pancap/caption_train2017",
                  mask_parquet="masks/coconut_s/data/*.parquet",
                  mask_dir="masks/png/train2017",
                  image_dir="images/train2017"),
    "val": dict(captions="coconut_pancap/caption_val2017",
                mask_parquet="masks/relabeled_coco_val/data/*.parquet",
                mask_dir="masks/png/val2017",
                image_dir="images/val2017"),
    "o365val": dict(captions="coconut_pancap/pancap_val",
                    mask_json="masks/coconut_val/coconut_val.json",
                    mask_dir="masks/coconut_val/panoptic_o365val_v3",
                    image_dir="images/objects365"),
}


def extract_tar(tar_path, dest, expect_dir):
    if os.path.isdir(os.path.join(dest, expect_dir)):
        return
    if not os.path.exists(tar_path):
        print(f"[skip] {tar_path} not downloaded yet")
        return
    print(f"[extract] {tar_path}")
    with tarfile.open(tar_path) as tf:
        tf.extractall(dest)


def index_from_parquet(root, split, cfg):
    files = sorted(glob.glob(os.path.join(root, cfg["mask_parquet"])))
    if not files:
        print(f"[skip] {split}: no mask parquet found ({cfg['mask_parquet']})")
        return None
    mask_dir = os.path.join(root, cfg["mask_dir"])
    os.makedirs(mask_dir, exist_ok=True)
    index = {}
    for f in files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            for row in pf.read_row_group(rg).to_pylist():
                info, seg = row["image_info"], row["segments_info"]
                key = os.path.splitext(info["file_name"])[0]
                png = os.path.join(mask_dir, key + ".png")
                if not os.path.exists(png):
                    with open(png, "wb") as out:
                        out.write(row["mask"]["bytes"])
                index[key] = dict(file_name=info["file_name"], width=info["width"], height=info["height"],
                                  url=info.get("coco_url"), segments=seg["segments_info"])
        print(f"[masks] {split}: {f} -> {len(index)} samples")
    return index


def index_from_json(root, split, cfg):
    path = os.path.join(root, cfg["mask_json"])
    if not os.path.exists(path):
        print(f"[skip] {split}: {path} not found")
        return None
    d = json.load(open(path))
    imgs = {im["id"]: im for im in d["images"]}
    index = {}
    for ann in d["annotations"]:
        key = ann["object365_file_name"]
        im = imgs[ann["image_id"]]
        index[key] = dict(file_name=key + ".jpg", width=im["width"], height=im["height"], url=None,
                          segments=ann["segments_info"])
    return index


def coco_val_images(root):
    out_dir = os.path.join(root, "images/val2017")
    files = sorted(glob.glob(os.path.join(root, "images/_coco_val_parquet/data/val-*.parquet")))
    if not files:
        print("[skip] COCO val2017 image parquet not downloaded yet")
        return
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for f in files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["image_id", "image"])
            for image_id, image in zip(t.column("image_id").to_pylist(), t.column("image").to_pylist()):
                p = os.path.join(out_dir, f"{image_id:012d}.jpg")
                if not os.path.exists(p):
                    with open(p, "wb") as out:
                        out.write(image["bytes"])
                n += 1
    print(f"[images] val2017: {n} images in {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=os.path.join(REPO, "data"))
    args = ap.parse_args()
    root = args.data_root

    cap_root = os.path.join(root, "coconut_pancap")
    for name in ["caption_train2017", "caption_val2017", "pancap_val"]:
        extract_tar(os.path.join(cap_root, name + ".tar"), cap_root, name)
    extract_tar(os.path.join(root, "masks/coconut_val/coconut_val.tar"), os.path.join(root, "masks/coconut_val"),
                "panoptic_o365val_v3")
    coco_val_images(root)

    os.makedirs(os.path.join(root, "index"), exist_ok=True)
    cats = None
    cat_json = os.path.join(root, "masks/coconut_val/coconut_val.json")
    if os.path.exists(cat_json):
        cats = json.load(open(cat_json))["categories"]
        json.dump(cats, open(os.path.join(root, "index/categories.json"), "w"))

    for split, cfg in SPLITS.items():
        out = os.path.join(root, "index", split + ".json")
        index = index_from_parquet(root, split, cfg) if "mask_parquet" in cfg else index_from_json(root, split, cfg)
        if index is None:
            continue
        cap_dir = os.path.join(root, cfg["captions"])
        caps = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(cap_dir, "*.txt"))}
        # keep every sample that has a caption or a mask so missing pieces show up in the viewer
        keys = sorted(caps | set(index))
        meta = dict(split=split, captions=cfg["captions"], mask_dir=cfg["mask_dir"], image_dir=cfg["image_dir"],
                    samples={k: index.get(k) for k in keys}, has_caption=sorted(caps))
        json.dump(meta, open(out, "w"))
        print(f"[index] {split}: {len(keys)} keys ({len(caps)} captions, {len(index)} masks, "
              f"{len(caps & set(index))} paired) -> {out}")


if __name__ == "__main__":
    main()
