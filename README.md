# CCTV Crowd Awareness — Local V1

One prerecorded video + the preserved Mall CSRNet epoch-16 model + a local
monitoring dashboard. **Research demonstration, not a crowd-safety system.**
FastAPI/Uvicorn, PyTorch/OpenCV, native HTML video, plain JavaScript and Canvas.
No training, recalibration, frontend CDN or database is needed to launch.

## Launch on this Windows laptop

From any PowerShell directory:

```powershell
& 'C:\Users\YCIN\cv_learning\start_dashboard.cmd'
```

The launcher uses this project's Python 3.11 `.venv`, checks dependencies,
configuration, checkpoint SHA-256 and FFmpeg, starts **one** loopback backend,
waits for model readiness, then opens **http://127.0.0.1:8000/**. It reuses an
already-ready matching service rather than starting a duplicate. Logs stay in the
terminal. **Ctrl+C stops the backend; closing it stops the site.** No automatic
package/model downloads or training.

Options: `start_dashboard.cmd --port 8001`, `--check` (preflight only), or
`--no-browser`. A conflicting/not-ready service produces a clear port error.

## Use, export and failure recovery

Choose a permitted local video or drop it on the video panel, wait for preparation,
then Play. Native video playback is independent of at most **4 model samples/s**.
Controls: Pause, ±5 seconds, Restart, timeline seek, density toggle and opacity.
Restart stays paused; paused seek takes one snapshot.

Counts use the **float64 sum of raw signed density**. No clipping or correction
multiplier. Heatmap scaling is display-only. RGB float32 [0,1] + ImageNet
normalization is unchanged. Whole-frame stride-eight resizing retains the scene;
the explicit configurable default budget is **921,600 pixels**. Changing image
scale can change predictions/accuracy. OOM stops with guidance, never hidden retry.

Current measurements expire after **5 seconds wall-clock age** or **2 seconds
video misalignment**. Disconnect/error pauses playback and clears count, attention
and overlay. Local Pause and historical export still work. Recovery never
autoplays: press Play after readiness returns. After a backend restart, first
Play reopens the retained File as a new segment; a second Play starts playback.
Export prior history first. Page reload loses browser history/file selection.

Graph-panel CSV/JSON exports retain full-precision counts, inference timestamps,
UTC capture/observation labels, generation/segment, attention/reason, shapes and
timings. JSON adds checkpoint hash/epoch/device/preprocessing, thresholds,
sampling policy, original **and** playback metadata, conversion settings, retained
range and discarded count. Exports work even while Python is disconnected.
Pair CSV with JSON for provenance. No ground truth/confidence/accuracy is invented.

Seek/restart/video replacement resets the segment/history. At most **1,200**
observations remain; earlier points are dropped and that loss is reported. Graph
shows the latest 60 seconds. Prior measurements are clearly labelled history.

Metrics distinguish nominal source FPS, synchronized model forward time, rolling
completed samples/s, service time, browser capture-to-result latency and sample
age. Browser latency uses `performance.now()`, includes JPEG + transfer + service,
and is **not live-camera latency**. Browser/server UTC clocks are labels only.

## Trusted model / fresh checkout

Use Python **3.11**, uv, a compatible NVIDIA driver for CUDA and sufficient
temporary disk. Restore dependencies with `uv sync --frozen` (includes the dev HTTP
test client). CPU is chosen if CUDA is unavailable, but can be very slow. Node is
optional for frontend regression tests, not runtime. Launcher never installs.

Datasets, checkpoints, footage and generated outputs are **not in Git**. Obtain
the owner's preserved epoch-16 artifact from their trusted Colab/Drive or verified
local backup. No public artifact download URL is provided. Place it at
`outputs/checkpoints/csrnet_drive_best.pt` and verify:

```powershell
Get-FileHash outputs/checkpoints/csrnet_drive_best.pt -Algorithm SHA256
```

Expected SHA-256:

```text
622956dd59ba559f877abb6f76e3fc8ca73a6a7ccd6c4ac9f477be48a7a7ba43
```

Launcher and backend verify the hash; backend checks epoch 16 too.
[model_manifest.json](model_manifest.json) records identity/provenance. Do not
substitute weights or run a training notebook to launch. Mall data is unnecessary
for the dashboard; the full historical test suite imports the Mall loader and
requires its 2,000 JPG/HDF5 pairs. Focused dashboard tests need no Mall data.

## Results — separate scopes

| Artifact/evaluation | MAE (people) | RMSE (people) | Provenance |
| --- | ---: | ---: | --- |
| Current epoch-16 Mall **validation** | 1.511238 | 1.940314 | Preserved checkpoint metadata |
| Historical epoch-10 Mall **test** | 1.901865 | 2.363966 | reports/csrnet_10_epoch_results.json |
| Epoch-16 HT21 diagnostic | 6.583950 | 7.241862 | 429 correlated frames, one compressed preview; reports/ht21_01_evaluation.md |

HT21 signed bias: **−6.583950**, every evaluated frame undercounted. **No verified
epoch-16 Mall test result** is present locally. Incomplete AI draft annotations
do not establish accuracy. Mall plus one short clip do not establish festival
reliability. A constant count correction is not a scientific fix.

## Verification / troubleshooting

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
node dashboard/test_dashboard_js.cjs
```

[Release verification](reports/v1_release_20261005.md) distinguishes automated,
real GPU/service and browser checks, including unverified checks.
[DASHBOARD.md](DASHBOARD.md) explains configuration/synchronization and remaining
manual acceptance. [VIDEO_INFERENCE.md](VIDEO_INFERENCE.md) covers offline inference;
[COLAB_README.md](COLAB_README.md) covers historical training, not launch.

- Missing/hash-mismatched model: restore the trusted artifact; don't edit the hash
  just to accept a different file.
- CUDA OOM: explicitly reduce `max_model_pixels` in configuration, then restart.
- Codec error: try H.264 MP4. Other decodable codecs use a lossy CRF18 H.264 copy,
  remove audio and pad odd dimensions by at most one pixel. Originals stay
  unchanged. JPEG quality 0.92 samples are not pixel-identical to offline input.
- Backend gone: restart launcher, wait for readiness, export retained history,
  then Play to reopen/resume. Read the visible log for errors.

## Limits and provenance

“Prototype attention thresholds; not validated safety limits.”
“Heatmap shows predicted image-space density, not people/m².”

Demonstration NORMAL/WATCH/ELEVATED/REVIEW thresholds are 40/70/100, configurable.
Negative raw counts remain in measurements but attention is unavailable.
No localized bottleneck, physical density, crowd danger or personnel advice.

**Local only**, one shared in-memory video/model/session; use one tab. No auth,
session isolation or hardened media sandbox. Upload limit is 2 GiB **after**
multipart parsing. Do not expose to LAN/internet or tunnel publicly. Forced kill
can leave owned temporary files. Only process permitted footage. No media/model
distribution rights, safety certification or repository license is invented.
Owner must review dataset/media provenance and permissions before redistribution.
Learning/data code remains preserved; the results notebook with embedded images
stays local, outside this source-only dashboard release.

Next stage: human-reviewed **target-camera** empty/night/dense scenes; measure
count error and bias before choosing scene-specific thresholds. Network hosting
also needs authentication, session isolation, upload limits before parsing,
duration/conversion quotas and bounded compute. Hosting choice needs audience,
hardware and budget details; it is not part of V1.
