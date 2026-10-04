# CSRNet + Mall: completed training and result review

## Current next step: review the completed run

The 10-epoch continuation completed successfully. Best epoch: 10; validation
MAE: 1.622362; test MAE: 1.901865. These were checked against the Drive result
JSON on 2026-09-08. Both checkpoints and all three prediction PNGs exist there.

`Review_CSRNet_Results.ipynb` has already run locally and contains the metrics and
all three prediction images. Open it to view the saved outputs without running
any cells. To refresh them in Colab, run its two cells on a CPU runtime.
It mounts Drive, reads saved metrics, and displays the three predictions. No
dataset ZIP, model loading, GPU, or training is needed for this review. Locally,
its first cell can read `reports/csrnet_10_epoch_results.json` from a working
Python kernel; images require the PNG files described in its second cell.

See `reports/csrnet_10_epoch_review.md` for the reviewed results and artifact links.
The instructions below describe the original training/continuation workflow;
they are not necessary just to view completed results.

## Start from VS Code

1. Reopen `CSRNet_Mall_Colab.ipynb` from this project so VS Code loads the updated file.
2. Select the notebook kernel: **Colab → your GPU server**.
3. Run the first code cell. It must print **CUDA available: True**, the GPU name,
   the Python interpreter, and the installed torch/torchvision versions.
4. Choose **Run All** and authorize the Drive mount if requested.

The notebook now contains a snapshot of the six current Python scripts. No manual
script upload, new dataset ZIP, or local terminal command is needed. Keep the
existing `My Drive/mall_csrnet_bundle.zip` uploaded.

The kernel name `Python 3` and notebook format 4.5 are normal. They do not select
a GPU by themselves: the kernel selection and the first cell's CUDA allocation
test establish that. The notebook requires Python 3.11 or newer and preserves
Colab's supplied CUDA-enabled torch/torchvision pair. Do not install the laptop's
`uv.lock` or run `pip install -e .` on Colab.

## What this run does

- Continues toward **10 total epochs**, including the completed first epoch.
- Uses pretrained VGG16 CSRNet, batch size **4**, Adam learning rate **1e-5**.
- Keeps frames 1–1400 / 1401–1700 / 1701–2000 as train / validation / test.
- Verifies the original ZIP, all 2,000 image/map pairs, and count-preserving target resizing.
- Saves the latest completed epoch and the best validation-MAE checkpoint.
- Loads the best model for one final test pass and saves three prediction panels.

Nine more epochs took an estimated 39 minutes at the previous run's 258 seconds
per epoch; setup, saving and GPU availability add variability.

## Where results are saved

The original experiment stays in:

```text
My Drive/csrnet_mall_runs/20260903_010838_885145_UTC/
```

The notebook copies the starting checkpoint and results into:

```text
My Drive/csrnet_mall_runs/20260903_010838_885145_UTC_continued/
    checkpoints/csrnet_best.pt
    checkpoints/csrnet_latest.pt
    csrnet_results.json
    initial_epoch_results.json
    csrnet_predictions/test_sample_001.png   (plus 002 and 003)
    csrnet_<timestamp>.log
    source_snapshot/
    colab_setup_<timestamp>.json
```

A missing original checkpoint or results file stops setup. No silent fresh start
occurs. To use a different existing experiment, edit `ORIGINAL_RUN` and select a
separate `RUN_DIR` in section 3.

## If the runtime disconnects

Reopen the notebook, select a Colab GPU kernel, and run cells in order. Setup
selects the most recent completed checkpoint in the continuation folder.
Completed runs display saved results without training or testing again.

Model weights, Adam state, completed epochs, and history are restored. Newly
saved checkpoints also preserve PyTorch random-generator state. The original
one-epoch checkpoint did not save random state, so its first continuation is
valid but cannot be exactly identical to an uninterrupted run.

Checkpoint replacements are written to a temporary file before replacing the
previous save. This reduces partial-write risk; it is not a guarantee against
Drive/runtime failure. An unfinished epoch must be repeated.

If training has finished but final test evaluation was interrupted, setup stops
for recovery rather than launching more training. `SKIP_TEST = True` can be used
for a future validation-only run. Leave it False for this planned final test.

## Keeping the notebook's code current

After editing the Python scripts locally, refresh the embedded snapshot:

```powershell
uv run python prepare_colab_bundle.py --refresh-notebook
```

This only updates the notebook's source cell. It does not rebuild the 1.10 GiB ZIP.
The original ZIP hash remains:

```text
3088c200659757144c6e35c38bcf9c3afc9cf0ede1b9cdcc590fee0c1c8f73bc
```

The verified ZIP is extracted separately from the updated scripts. The updated
script directory links to those same images/maps; no density maps are regenerated.

## CLI versus VS Code

The WSL CLI can list a server created outside its own session records as `[?]`.
That means it has no local name for that assignment; it does **not** establish
that the server is stale or that its kernel lacks a GPU. The VS Code notebook can
use its selected Colab server directly. A local PowerShell `python` command still
uses the laptop's environment.

After saving the results, use the extension's **Colab: Remove Server** command to
release a server you no longer need. GPU availability and runtime limits vary:
see the [Colab FAQ](https://research.google.com/colaboratory/faq.html) and
[official VS Code extension](https://github.com/googlecolab/colab-vscode).
