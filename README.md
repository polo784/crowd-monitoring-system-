# CCTV Crowd Awareness

A local dashboard for prerecorded video. CSRNet estimates crowd counts and
produces a heatmap. The dashboard supports named counting zones, playback
controls, and timestamped CSV and JSON exports. Offline video inference and
annotated-image benchmarks use the same predictor.

## Run the dashboard

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run
   `uv sync --frozen` from the repository root. The project uses Python 3.11.
2. Obtain the preserved Mall CSRNet epoch-16 checkpoint from a trusted backup.
   Put it at `outputs/checkpoints/csrnet_drive_best.pt`. The launcher verifies
   its SHA-256 against the [model manifest](model_manifest.json). The checkout
   contains source and documentation. The checkpoint, datasets, and footage
   are supplied separately.
3. In PowerShell, run `.\start_dashboard.cmd`. When the model is ready, the
   launcher opens [the dashboard](http://127.0.0.1:8000/). Keep the terminal open
   while using the app. Ctrl+C stops the server.
4. Choose or drop a local video, wait for preparation, then press Play. Pause and
   select **Draw zone** to measure a part of the frame. Export measurements from
   the graph panel before resetting the session.

The app uses CUDA when available and falls back to CPU. Use
`start_dashboard.cmd --check` for a startup check, `--port 8001` if the usual
port is occupied, or `--no-browser` to start only the backend.

## Project guides

- [Dashboard controls, counting zones, exports, and troubleshooting](docs/dashboard.md)
- [Offline video inference](docs/video-inference.md)
- [Annotated-image benchmarking and ShanghaiTech manifests](docs/benchmarking.md)
- [Mall training and result review in Colab](docs/colab.md)
- [Verification report](reports/v1_release_20261005.md)

## Model results

| Evaluation | MAE (people) | RMSE (people) |
| --- | ---: | ---: |
| Epoch-16 Mall validation, from checkpoint metadata | 1.511238 | 1.940314 |
| [Historical epoch-10 Mall test](reports/csrnet_10_epoch_review.md) | 1.901865 | 2.363966 |
| [Epoch-16 HT21 diagnostic](reports/ht21_01_evaluation.md) | 6.583950 | 7.241862 |

The HT21 diagnostic covers 429 correlated frames from one compressed preview.
The recorded signed bias was −6.583950 people. The prototype is for local use.
Attention thresholds are demonstrations. Target-camera validation is required
before interpreting counts as operational evidence.

## Run tests

From the repository root with the environment installed:

```powershell
uv run --frozen python tools/run_source_tests.py
node tests/test_dashboard_js.cjs
```

The source tests use synthetic fixtures and run without the Mall dataset or model
weights. Node is required only for the frontend tests.
