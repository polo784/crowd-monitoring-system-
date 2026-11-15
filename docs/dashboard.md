# Local V1 dashboard behavior

See the [README](../README.md#run-the-dashboard) for setup and launch commands.
The launcher waits up to 120 seconds for model readiness. It runs one worker
without reload. Ctrl+C requests a clean shutdown. A forced exit may leave
temporary files.

Choose or drop a permitted local video, wait for preparation, then press Play.
Use Pause, ±5 seconds, Restart, the timeline, and the heatmap toggle and opacity
controls. Restart leaves the video paused.

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

## Counting zones

1. Choose a video, play briefly, then pause.
2. Select **Draw zone** and drag a rectangle over the image.
3. Enter a name (up to 40 characters), then select **Save zone**. A paused
   snapshot shows the first regional estimate. Press Play for updates.
4. Use **Clear zones** to remove the layout. Seek and restart keep the layout.
   Selecting a different video clears it. Export JSON to retain definitions.

The dashboard supports up to six rectangles. Saving or clearing the layout starts
a new history segment, so export the previous segment first. In-flight results
from the old layout are discarded. Expired measurements clear regional estimates
and leave the outlines visible.

Each rectangle uses normalized playback-frame edge coordinates `[left, top,
right, bottom]` in `[0,1]`. Its estimate sums each raw signed density cell times
the fraction of the cell it covers. A full-frame rectangle equals the full raw
count. Adjacent disjoint rectangles covering the frame sum to that count within
floating-point tolerance. Overlapping rectangles both include the shared mass.
The coarse grid and density smoothing make regional counts approximate. The
prototype attention indicator uses the full-frame count.

`POST /api/control` accepts `action: "zones"` with a `zones` array of objects
such as `{"id":"gate","name":"Gate","rect":[0,0,0.5,1]}`. The request also
includes the session, generation, timestamp and playing fields. Invalid layouts
leave the current generation and history intact. Inference and history rows
include `zone_counts` by ID. JSON retains definitions and the integration policy,
and browser CSV adds `zone_<id>_count` columns. If the launcher detects an older
backend API, stop that process and relaunch or select a different port.

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

## Troubleshooting

- If the page refuses the connection, run `start_dashboard.cmd` and keep its
  terminal open. If the port is occupied, use `--port 8001` and open that port.
- If checkpoint verification fails, restore the trusted artifact recorded in
  [the model manifest](../model_manifest.json). The launcher performs no downloads.
- For CUDA memory errors, lower `max_model_pixels` in `dashboard_config.json`
  and restart. Image scale can affect the predicted count.
- If codec preparation fails, try an H.264 MP4 and check the terminal log.

## Manual acceptance checklist

See the [verification report](../reports/v1_release_20261005.md) for the real
browser checks, backend soak, and outstanding scenarios.

1. Launch; select a permitted real video and Play. Verify moving people, updating
   count/heatmap/history. Exercise Pause/seek/restart/±5s and density controls.
2. Stop backend during playback: playback pauses, current measurements/overlay
   clear, Pause/history exports work. Restart without page reload; recovery
   requires Play (reopen after lost session), then Play again.
3. Paused seek: new snapshot, expires after 5 seconds. Rapid seek/video replacement
   must never restore future/obsolete results. Check timeout/error messages.
4. Download CSV/JSON. Verify one count, exact hash, original vs playback properties,
   shapes, segment/generation and discarded count against the graph/history.
5. Run a 15-minute browser playback soak and measure resource use and
   responsiveness.
