# COCONut-PanCap viewer

Browse COCONut-PanCap images, panoptic masks and grounded captions side by side, and check that they agree.

```bash
python tools/pancap_viewer/server.py --port 8777       # open http://localhost:8777
```

- Hover a caption tag (`<3, 4: bananas>`) to highlight its segments. Click it to pin the highlight.
- Hover the image to see a segment's id, its category, and the caption text that mentions it.
- Use the sidebar filters (status and issue code) to jump to failing samples. `←` / `→` move to the previous or next sample, and `o` cycles the overlay mode.
- The URL hash (`#split=val&key=000000000139`) links straight to a sample.

## Data (under `data/`)

| split | captions | masks | images |
|---|---|---|---|
| `train` | `coconut_pancap/caption_train2017` (118,113) | COCONut-S, `masks/png/train2017` (118,200) | fetched on demand from images.cocodataset.org and cached in `images/train2017` |
| `val` | `coconut_pancap/caption_val2017` (5,000) | relabeled COCO val, `masks/png/val2017` (5,000) | `images/val2017` (local) |
| `o365val` | `coconut_pancap/pancap_val` (20,587) | COCONut val, `masks/coconut_val/panoptic_o365val_v3` (20,622) | **not available**: Objects365 ships only as ~343 GB of tar.gz patches. Mask-only view. |

To rebuild everything, run the three scripts in order:

```bash
python tools/pancap_viewer/prepare_data.py              # unpack tars and parquet masks, write data/index/*.json
python tools/pancap_viewer/verify.py --split val        # write data/index/report_val.json (drives the viewer filters)
python tools/pancap_viewer/download_train_images.py     # optional: all train images, ~18 GB
```

## Checks

| code | level | meaning |
|---|---|---|
| `tag_unknown_id` | error | The caption cites a segment id that isn't in `segments_info`. |
| `tag_void_id` | error | The caption cites id 0, the unlabeled region. |
| `mask_unlisted_id` | error | The mask has pixel ids that `segments_info` doesn't list. |
| `mask_size_mismatch` / `image_size_mismatch` | error | The mask size differs from the metadata or from the image. |
| `missing_caption` / `missing_mask_png` | error | The caption or mask file is missing. |
| `tag_empty_segment` | warn | A cited segment has 0 pixels in the mask. |
| `segment_area_mismatch` | warn | The `area` in `segments_info` differs from the mask's pixel count. |
| `tag_bare` / `tag_xml` / `tag_malformed` | warn | A tag is non-standard: `<11>`, `<0>text</0>`, or `<no id: text>`. |
| `unbalanced_brackets` | warn | The caption has different numbers of `<` and `>`. |
| `uncovered_segments` | info | Some visible segments are never mentioned in the caption. |
