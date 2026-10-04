# CSRNet external-scene evaluation — 2 October 2026

Evaluated the existing Mall-trained CSRNet checkpoint unchanged on HT21-01,
an indoor train-station sequence from [CroHD / Head Tracking 21](https://motchallenge.net/data/Head_Tracking_21/).
No training, fine-tuning, scale search, density clamping or count calibration was performed.

## Results

All 429 frames (17.16 seconds, 25 FPS) were evaluated. Model counts are the
float64 sum of the raw output density map. Primary ground truth counts valid,
visible human heads, including stationary people; ignore-labelled non-people
and fully occluded tracks are excluded. CroHD classes are 1 pedestrian, 2 static,
3 ignore, and 4 person on vehicle, as documented in the
[dataset paper](https://arxiv.org/html/2103.13516v1).

| Measurement | Result |
|---|---:|
| Visible-head count MAE | 6.583950 people |
| Visible-head count RMSE | 7.241862 people |
| Mean signed error (prediction minus ground truth) | -6.583950 people |
| Average visible ground-truth count | 46.575758 |
| Average predicted count | 39.991807 |
| Visible ground-truth count range | 39–55 |
| Predicted count range | 33.422834–48.501767 |
| Frames with undercount | 429 / 429 |
| Secondary MAE including fully occluded human tracks | 8.987214 people |
| Secondary RMSE including fully occluded human tracks | 9.671838 people |

The model systematically undercounts this scene. One short, strongly correlated
sequence is not sufficient to judge other cameras, dense festival crowds, or
deployment safety. This is a counting adaptation using public CroHD training
annotations, not an official tracking benchmark or hidden-test result.

## Input and performance

- Device: NVIDIA GeForce GTX 1650, CUDA; PyTorch 2.12.1+cu126.
- Checkpoint: `outputs/checkpoints/csrnet_drive_best.pt`, epoch 16.
- Checkpoint SHA256: `622956dd59ba559f877abb6f76e3fc8ca73a6a7ccd6c4ac9f477be48a7a7ba43`.
- Published MP4 is actually a compressed 960×540 preview with baked-in sequence
  title/counters, not the original 1920×1080 frame sequence stated on the page.
- Model input: `(1, 3, 544, 960)`. Explicitly stretch height from 540 to 544 solely
  for stride-8 compatibility; no cropping. Model output: `(1, 1, 68, 120)`.
- Visualization retains 960×540. Annotation boxes are scaled by 0.5 in each
  direction for sampled visual checks; annotation counts are never scaled.
- Average forward-pass inference: 169.93 ms (5.88 model-only FPS).
- End-to-end inference/overlay/encoding: 4.31 FPS, average frame latency 231.97 ms.
- Inference processing time: 99.56 seconds, excluding model loading and evaluation.
- This is slower than the source's 25 FPS, so full-rate live processing is not achieved.

## Verification and artifacts

- Archive CRC check succeeded. Download byte lengths and SHA256 hashes saved in
  `videos/ht21/download_manifest.json`; these hashes record what was downloaded,
  not an independently published checksum from the dataset authors.
- Used only `HT21Labels/train/HT21-01/gt/gt.txt`, not detector predictions in `det/`.
- 21,456 annotation rows: 19,981 visible human heads, 21,012 human rows including
  occluded tracks, 444 ignored rows. Included valid classes 1, 2 and 4.
- Annotation frames 1–429 map to decoded video indexes 0–428. FPS, timestamps,
  complete lengths and uniqueness checked. All 429 output frames decoded again.
- Visually inspected matching ground-truth boxes at indexes 0, 214 and 428.
- Independent recalculation from the per-frame CSV confirmed the metrics.
- Seventeen annotation, inference and counting-rule tests passed.

Outputs under `outputs/video_inference/ht21_01_gpu_20261002/`:

- `annotated.mp4`: model density overlay and count.
- `measurements.csv`, `summary.json`: raw predictions and timings.
- `evaluation/comparison.mp4`: model overlay plus visible ground-truth count and error.
- `evaluation/comparison.csv`, `evaluation/evaluation.json`: per-frame errors and full report.
- `evaluation/frame_000000_comparison.jpg`, `frame_000214_comparison.jpg`,
  `frame_000428_comparison.jpg`: human head boxes alongside the model overlay.

## Commands

From `C:\Users\YCIN\cv_learning` (choose new output directories for another run):

```powershell
.\.venv\Scripts\python.exe -u video_inference.py --video videos/ht21/HT21-01-raw.mp4 --checkpoint outputs/checkpoints/csrnet_drive_best.pt --output-dir outputs/video_inference/ht21_01_gpu_20261002 --device cuda --input-size 544 960
.\.venv\Scripts\python.exe -u evaluate_ht21.py --labels videos/ht21/HT21Labels.zip --sequence HT21-01 --run-dir outputs/video_inference/ht21_01_gpu_20261002 --output-dir outputs/video_inference/ht21_01_gpu_20261002/evaluation
.\.venv\Scripts\python.exe -m unittest test_ht21_evaluation test_video_inference test_video_annotations -v
```

Created `download_ht21_sample.py`, `evaluate_ht21.py` and
`test_ht21_evaluation.py`. Added only `videos/ht21/` to the existing `.gitignore`
so downloaded dataset binaries are not accidentally committed. Model, training,
preprocessing, Mall data and splits were left unchanged.

## Playback-compatible copy

After the original comparison video could not be opened in the user's viewer,
verified that the original MPEG-4 Part 2 video fully decodes (429 frames).
Saved `evaluation/comparison_h264.mp4` as a separate H.264/yuv420p copy with
fast-start MP4 metadata. Verified all 429 frames, 960×540 dimensions and 25 FPS;
the MP4 metadata precedes the video data. The original video and numeric results
were preserved. Encoding is lossy but does not change the saved count measurements.
The video converter is isolated under `outputs/tools/video_playback/`; project
dependencies and the ML environment were not changed.
