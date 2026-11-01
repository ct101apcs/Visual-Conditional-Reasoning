"""Optional: bulk-download all COCO train2017 images (~18 GB) into data/images/train2017.

The viewer already fetches train images on demand, so this is only needed for offline use or for
running verify.py with the image-size check on the train split. Shards come from the
detection-datasets/coco HF mirror (images.cocodataset.org is very slow from this machine); each
~480 MB shard is extracted and deleted before the next, so peak extra disk use stays small.
Images the mirror lacks (those without boxes) are then fetched from images.cocodataset.org.

  python tools/pancap_viewer/download_train_images.py
"""
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

import pancap

REPO_ID = "detection-datasets/coco"


def main():
    out_dir = os.path.join(pancap.DATA_ROOT, "images/train2017")
    tmp_dir = os.path.join(pancap.DATA_ROOT, "images/_coco_train_shard")
    os.makedirs(out_dir, exist_ok=True)
    shards = sorted(f for f in HfApi().list_repo_files(REPO_ID, repo_type="dataset") if f.startswith("data/train-"))
    for i, shard in enumerate(shards):
        done_flag = os.path.join(out_dir, f".done_{os.path.basename(shard)}")
        if os.path.exists(done_flag):
            continue
        free_gb = shutil.disk_usage(out_dir).free / 2**30
        if free_gb < 3:
            raise SystemExit(f"Stopping: only {free_gb:.1f} GB free")
        path = hf_hub_download(REPO_ID, shard, repo_type="dataset", local_dir=tmp_dir)
        t = pq.read_table(path, columns=["image_id", "image"])
        for image_id, image in zip(t.column("image_id").to_pylist(), t.column("image").to_pylist()):
            p = os.path.join(out_dir, f"{image_id:012d}.jpg")
            if not os.path.exists(p):
                with open(p, "wb") as f:
                    f.write(image["bytes"])
        os.remove(path)
        open(done_flag, "w").close()
        print(f"[{i + 1}/{len(shards)}] {shard}: {t.num_rows} images, {free_gb:.1f} GB free", flush=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    split = pancap.Split("train")
    missing = [k for k in split.keys if not split.image_path(k)]
    print(f"fetching {len(missing)} images missing from the mirror", flush=True)
    with ThreadPoolExecutor(8) as ex:
        for k, err in ex.map(lambda k: (k, _fetch(split, k)), missing):
            if err:
                print(f"  {k}: {err}")


def _fetch(split, key):
    try:
        split.image_path(key, fetch=True)
    except Exception as e:
        return repr(e)


if __name__ == "__main__":
    main()
