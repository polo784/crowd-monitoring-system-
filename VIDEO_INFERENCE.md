# Recorded-video crowd inference

`crowd_inference.py` loads a compatible CSRNet checkpoint and predicts one frame.
`video_inference.py` handles video decoding, visualization, measurements, and
encoding. Neither imports the Mall Dataset, DataLoaders, or training script.
To use a later CSRNet model, change `--checkpoint`.

## Run

From the project directory, with the project's dependencies installed:

```powershell
uv --cache-dir .uv-cache run --frozen python video_inference.py --video "C:\path\to\crowd.mp4" --checkpoint "outputs/checkpoints/csrnet_drive_best.pt" --output-dir "outputs/video_inference/run_001"
```

If using the existing activated environment, `python video_inference.py ...`
works too. In Colab, install `opencv-python-headless` if needed and use Colab's
CUDA-enabled PyTorch; do not sync the laptop environment into Colab.

Choose a new output directory for each run. Existing runs are never overwritten.
The default device is CUDA when available, otherwise CPU. Use `--device cuda`
to require a GPU, or `--device cpu` to explicitly select CPU.

### Local NVIDIA GPU setup

On Windows x64, `pyproject.toml` selects the official CUDA 12.6 builds of PyTorch
2.12.1 and torchvision 0.27.1. These sources are saved in `uv.lock`, so a later
sync retains GPU support. Other platforms retain their normal package sources.
This follows the [official version pairing](https://pytorch.org/get-started/previous-versions/)
and [uv's PyTorch source configuration](https://docs.astral.sh/uv/guides/integration/pytorch/).

```powershell
uv --cache-dir .uv-cache sync --frozen
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Select this project's `.venv` interpreter in VS Code. Restart an already-running
Python notebook kernel after changing the installed packages. Colab remains a
separate environment; keep using its supplied CUDA installation.

Optional controls:

- `--max-frames 12`: a short verification run, explicitly marked in the report.
- `--input-size 480 640`: explicitly resize the model input to height 480,
  width 640. This can distort aspect ratio and change accuracy; choose dimensions
  that preserve your video's aspect ratio where possible. The output video still
  has the original dimensions. Counts are always sums at model output resolution.
- `--heatmap-alpha 0.5`: overlay opacity, between 0 and 1.
- `--heatmap-max 0.1`: a fixed visualization scale in raw model-density units.
- `--normalization imagenet` or `none`: explicit preprocessing for a state-only
  checkpoint without metadata. Normally leave this at `auto`.

The default is native-resolution processing with no resizing, cropping or padding.
Native model input dimensions must be multiples of 8 for an exact match to
CSRNet's output grid. Otherwise the runner stops and requests an explicit size.
MP4 source dimensions must be even; the runner refuses silent encoder cropping.
Very large native frames can exceed GPU memory; no automatic resolution reduction
or device fallback occurs after an out-of-memory error.

## Checkpoint contract

The normal checkpoint contains `model_state_dict` and `training_config`.
`training_config.pretrained_vgg16` determines whether ImageNet normalization is
required. Mean/std are read from the saved configuration, with standard ImageNet
values as defaults. Bare state dictionaries require explicit normalization.
Weights are loaded strictly, on CPU first, using `weights_only=True`.
No pretrained weights are downloaded during inference.

Frames enter OpenCV as BGR uint8, are converted to RGB float32 in [0,1], and then
normalized as configured. Inference uses `model.eval()` and inference mode, batch 1.
The count is a float64 sum of the **raw** predicted density map. Negative values
are retained; they are neither clipped nor rounded away in the measurements.

## Outputs

Each run saves:

- `annotated.mp4`: source frames with count, video timestamp, and density overlay.
- `measurements.csv`: one row per processed frame, zero-based frame index,
  video timestamp in seconds, raw estimated count, and stage timings.
- `summary.json`: checkpoint identity/hash/epoch, normalization, device, video
  properties, actual tensor shapes, timing averages, file paths, and warnings.

The heatmap is relative model density, not people per square metre or a danger
score. Red is positive and blue is negative. By default each frame uses its own
99th-percentile absolute density scale; colors cannot be compared quantitatively
between frames. A fixed `--heatmap-max` gives a shared scale at one input size.
Heatmap interpolation/scaling never changes the count.

Source decoder timestamps are recorded where available. Missing/non-monotonic
timestamps use a documented nominal-FPS approximation. The annotated video is
silent `mp4v`, encoded at the input's nominal FPS; variable-frame-rate timing is
approximated. There is no intentional frame skipping or realtime playback delay.

## Timing definitions

- **Average inference time:** forward pass only, with CUDA synchronization.
- **Model-only FPS:** reciprocal of mean forward-pass time.
- **Effective inference FPS:** processed frames divided by video-loop wall time,
  including decoding, preprocessing, prediction, overlay, CSV, and encoder finish.
- **Approximate per-frame latency:** decode through video encode submission.
  This is processing latency for a file, not live camera-to-screen latency.

The first cold inference is included and also reported separately. Model loading,
checkpoint hashing, and final output verification are excluded from these averages.
Batch size is 1; each video frame has its own measurement.

If execution fails, `summary.json` records the error and any partial video is
named `annotated.partial.mp4`. A failed run must not be treated as complete.
Decoded frame count is compared with source metadata, and the output is checked
for readability, dimensions, and reported frame count.

## Verify the infrastructure

```powershell
uv --cache-dir .uv-cache run --frozen python -m unittest test_video_inference -v
```

Tests cover BGR/RGB conversion, normalization, signed raw counts, visualization
separation, explicit resizing, checkpoint metadata, timestamps, encoded video
frames, output overwrite protection, and non-finite prediction failure.

## Verified local run

The video pipeline was checked using the trained checkpoint downloaded from the
existing Drive best-checkpoint link. That file now contains **epoch 16**, validation
MAE **1.5112376492**, which is newer than the saved 10-epoch results report.
Its SHA-256 is `622956dd59ba559f877abb6f76e3fc8ca73a6a7ccd6c4ac9f477be48a7a7ba43`.

No application video was selected, so the verification input was assembled from
Mall frames 1–12 at an arbitrary 6 FPS. It is a 640x480, 2-second MP4, solely a
functional test fixture. Its timing does not represent the original CCTV timing.
No crowd-accuracy or temporal-performance claim is made from this fixture.

Exact command used from `C:\Users\YCIN\cv_learning`:

```powershell
.\.venv\Scripts\python.exe video_inference.py --video outputs/video_inference/mall_verification_input.mp4 --checkpoint outputs/checkpoints/csrnet_drive_best.pt --output-dir outputs/video_inference/mall_verification_final --device cpu
```

For another run, select a new output directory. The completed outputs are:

- `outputs/video_inference/mall_verification_final/annotated.mp4`
- `outputs/video_inference/mall_verification_final/measurements.csv`
- `outputs/video_inference/mall_verification_final/summary.json`
- `outputs/video_inference/mall_verification_final/preview.jpg`

All 12 encoded frames were decoded and checked against 12 finite CSV count rows.
Native input was `(1,3,480,640)`, prediction `(1,1,60,80)`, with no resize.
CPU forward passes averaged **1346.32 ms**; model-only throughput **0.743 FPS**;
full processing throughput **0.712 FPS**; approximate per-frame processing latency
**1403.92 ms**. These are short-run CPU measurements, including the first cold pass.
Six infrastructure tests also passed. See the subsequent GPU verification below.

The broken local virtual environment was rebuilt using the installed Python
3.11.15 and the project lockfile. That CPU run used **2.12.1+cpu**, torchvision
**0.27.1+cpu**, and OpenCV **5.0.0** was added. No training or preprocessing code
was changed.

### Subsequent GPU verification

The project environment now uses **torch 2.12.1+cu126** and **torchvision
0.27.1+cu126**. CUDA 12.6 is available on the laptop's **NVIDIA GeForce GTX 1650,
4 GB**, compute capability 7.5. The existing driver **560.94** was unchanged.
A GPU tensor matrix multiplication and the complete inference pipeline passed.

Exact GPU command used:

```powershell
.\.venv\Scripts\python.exe video_inference.py --video outputs/video_inference/mall_verification_input.mp4 --checkpoint outputs/checkpoints/csrnet_drive_best.pt --output-dir outputs/video_inference/mall_verification_gpu --device cuda
```

The checkpoint, 12-frame input, batch size 1, and native 640x480 processing are
identical to the earlier CPU verification. All 12 output frames decoded correctly.
The largest CPU/GPU count difference was **0.000002704 people**; no material count
change was observed. All six infrastructure tests passed after installation.

| Measurement | Earlier CPU run | GTX 1650 run |
| --- | ---: | ---: |
| Mean forward pass, including first frame | 1346.32 ms | 155.34 ms |
| Model-only throughput | 0.743 FPS | 6.437 FPS |
| Whole processing loop | 0.712 FPS | 4.089 FPS |
| Approximate processing latency per frame | 1403.92 ms | 243.50 ms |

The first GPU forward pass took 854.32 ms. The remaining eleven averaged
91.80 ms. These are short verification measurements, not a sustained-video or
multi-camera benchmark. Startup/model loading is excluded from both runs.

GPU results are in `outputs/video_inference/mall_verification_gpu/`:
`annotated.mp4`, `measurements.csv`, and `summary.json`. Select a new output
directory when running a real video. Larger input dimensions may require more
memory; the successful 640x480 run does not establish that every video size fits.
