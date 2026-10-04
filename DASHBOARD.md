# Local V1 dashboard behavior

Launch `start_dashboard.cmd` (absolute path also works outside the project).
It checks existing Python 3.11 dependencies, configuration, checkpoint SHA-256
and FFmpeg. Waits up to 120 seconds for model readiness before opening
http://127.0.0.1:8000/. One worker, no reload. Ctrl+C requests cleanup/shutdown;
forced kill can leave temporary files. Browser refresh doesn't start a server.

## Implementation and synchronization

- `crowd_inference.py`: existing strict CSRNet loading, RGB/ImageNet preparation,
  synchronized forward timing and raw signed float64 sum. Alternate deliberately
  unnormalized metadata now supports null mean/std; required missing ImageNet
  statistics are rejected rather than guessed.
- `dashboard_service.py`: one predictor, nonblocking inference mutex (429 when
  busy, no GPU queue), generation guards, segment IDs, bounded history, prototype
  attention and export provenance.
- `dashboard.py`: loopback FastAPI, uploads/playback copies, original/playback
  metadata, responsive health/control. `/api/export` inspects accepted backend
  history; browser exports additionally measure browser latency.
- `dashboard/`: native video, bounded/cancellable sampled requests, Canvas
  heatmap/chart, CSV/JSON downloads that remain available offline.
- `launch_dashboard.py`, `start_dashboard.cmd`: readiness and duplicate-process
  checks; no installs/model downloads/training.

Explicit states: loading, ready, processing, paused, stale, unavailable, error.
Browser video stays native-rate. JPEG quality 0.92 samples are attempted at most
4/s, one outstanding request. Session/generation checks reject obsolete responses,
including asynchronous PNG decoding. Duplicate/out-of-order timestamps stay
rejected even after visible data expires.

Play/pause invalidates pending work; seek/restart also starts a new segment and
clears history. Paused seek schedules one snapshot. Ordinary Pause clears current
values; paused-seek snapshots expire by wall-clock age too. History is explicitly
historical, never a current safety assessment.

Backend loss/error pauses video, invalidates count/attention/overlay, retains
history, and leaves local Pause usable. Aborting a browser request cannot stop a
running CUDA kernel: backend stays serialized, rejects retry backlog with 429,
and discards work invalidated before completion. Health/control don't acquire the
GPU mutex. Recovery requires readiness, never autoplays. After restart loses the
session, first Play reopens the retained File as a new segment; second Play starts.
Export previous history before reopening. Reload loses browser history/File.

## Central configuration

Types, finite values/ranges, integer-only history/resource limits and ordered
attention thresholds are validated. Defaults in `dashboard_config.json`:

| Setting | Default | Purpose |
| --- | ---: | --- |
| max_sample_hz | 4 | client sample-attempt cap |
| max_model_pixels | 921600 | explicit whole-scene model input budget |
| max_video_bytes | 2147483648 | local upload cap AFTER multipart parsing |
| max_frame_bytes | 8388608 | JPEG body cap, header checked before decode |
| history_limit | 1200 | retained observations, allowed 1..1200 |
| overlay_max_age_seconds | 2 | maximum video-time misalignment |
| observation_max_age_seconds | 5 | wall-clock expiry, including paused |
| health_timeout_seconds | 5 | health request bound |
| control_timeout_seconds | 5 | control request bound |
| inference_timeout_seconds | 30 | permits cold inference, not indefinite flight |
| upload_timeout_seconds | 660 | permits bounded 600-second FFmpeg conversion |

CPU may exceed interactive defaults; change settings explicitly if necessary.
OOM gives reduce-budget/restart guidance, not an automatic resize retry.
Full-frame bilinear resize rounds to stride eight within the pixel budget:
960×540 → 960×544, model (1,3,544,960), density (1,1,68,120).
Larger scenes downscale explicitly. No crop/count compensation; scale affects
predictions and is not an accuracy-preservation guarantee.

## Exports, metrics and visualization

Browser CSV has full numeric precision, timestamps/UTC labels, generation/segment,
attention/reason, shapes and timings for valid retained observations. JSON adds
exact checkpoint identity/device/normalization, configured rate/thresholds,
original vs playback metadata, conversion settings, history capacity, discarded
count and retained range. No invented ground truth/confidence. CSV should be
paired with JSON for provenance. Offline export works without Python; browser
refresh loses in-memory data. Seek/restart resets history; graph shows 60 seconds,
retains at most 1,200 observations.

Raw signed density is summed before display operations. Signed per-sample
99th-percentile scaling makes red-positive/blue-negative PNG alpha; expansion
and opacity are display-only. Color isn't absolute density across time nor
people/m². Negative raw count remains recorded but attention is unavailable.
Prototype thresholds are 40/70/100, and the triggering threshold is explained.

Source FPS is nominal metadata, not measured browser presentation FPS.
CUDA-synchronized forward milliseconds/reciprocal FPS differ from rolling
completed samples/s. Service includes decode/resize/predict/PNG, excludes HTTP.
Capture-to-result includes JPEG+upload+service+response on one browser monotonic
clock. Browser/server UTC labels are not subtracted; this is not live-camera latency.

## Formats, ownership and limitations

OpenCV-decodable MP4/MOV/M4V/AVI/MKV/WebM accepted. H.264 MP4 passes through;
other codecs create H.264 CRF18/yuv420p/faststart copies, remove audio and pad
odd dimensions by at most one pixel. Original metadata stays separate. Lossy
copies and JPEG sampling are not pixel-identical to offline evaluation.
Original files/checkpoints/data/results remain unchanged. Only owned upload
copies are removed on replacement or clean shutdown.

One shared session, local origin/host checks, no authentication, upload quota
before parsing, duration quota or untrusted-media sandbox. Do not expose publicly.

## Release verification and manual acceptance still required

See `reports/v1_release_20261005.md`. Fake-DOM tests are **not** browser tests.
Browser automation lost its session and native automation was blocked during
this release. Old screenshots don't establish that changed V1 passes acceptance.

1. Launch; select a permitted real video and Play. Verify moving people, updating
   count/heatmap/history. Exercise Pause/seek/restart/±5s and density controls.
2. Stop backend during playback: playback pauses, current measurements/overlay
   clear, Pause/history exports work. Restart without page reload; recovery
   requires Play (reopen after lost session), then Play again.
3. Paused seek: new snapshot, expires after 5 seconds. Rapid seek/video replacement
   must never restore future/obsolete results. Check timeout/error messages.
4. Download CSV/JSON. Verify one count, exact hash, original vs playback properties,
   shapes, segment/generation and discarded count against the graph/history.
5. Do a 15-minute **browser playback** soak with resource/responsiveness checks.
   A completed real-backend soak does not replace this blocked browser test.
