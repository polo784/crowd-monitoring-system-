# Measure before retraining

The existing Mall model remains the immutable baseline. More crowded training
images may help, but do not establish performance at a new camera automatically.
This milestone adds **evaluation infrastructure**, not a new trained checkpoint.

## Dense crowd dataset candidates

- First candidate: **ShanghaiTech Part A**, using the original author's dataset
  distribution: https://github.com/desenzhou/ShanghaiTechDataset. Part B is useful
  for a different street-scene distribution; assess both rather than assuming
  one dense dataset solves every camera.
- Larger/diverse candidate: **UCF-QNRF**, from UCF's research group:
  https://www.crcv.ucf.edu/research/data-sets/ucf-qnrf/.
- Ultimately collect permitted, human-reviewed **target-camera** samples,
  including empty, low-count, night, occluded and very crowded scenes. Different
  image scales and viewpoints can change predictions substantially.

These are research candidates, not guarantees of improvement. Review original
dataset/image permissions and access conditions before downloading or sharing.
No dataset is downloaded, relabelled, preprocessed or trained by the commands below.
Do not upload datasets/annotations to GitHub.

## Evaluate locally supplied ShanghaiTech annotations

Place an authorized download under ignored `datasets/ShanghaiTech/`, retaining:

```text
datasets/ShanghaiTech/
  part_A_final/
    train_data/images/IMG_1.jpg
    train_data/ground-truth/GT_IMG_1.mat
    test_data/images/IMG_1.jpg
    test_data/ground-truth/GT_IMG_1.mat
  part_B_final/...
```

The reader requires `image_info.location` points, checks `number` when supplied,
handles zero/one/multiple points, and does not regenerate density maps. Its MAT
schema is tested with nested cell/struct fixtures; a real downloaded ShanghaiTech
run has not yet been performed in this repository.

```powershell
.\.venv\Scripts\python.exe prepare_shanghaitech_manifest.py --root datasets/ShanghaiTech --part A --split test --out outputs/benchmarks/shanghai_A_test_manifest.json
.\.venv\Scripts\python.exe count_benchmark.py --manifest outputs/benchmarks/shanghai_A_test_manifest.json --checkpoint outputs/checkpoints/csrnet_drive_best.pt --device cuda --max-model-pixels 921600 --out outputs/benchmarks/mall_on_shanghai_A_baseline
```

Use fresh output paths; previous results are not overwritten. Evaluation stops on
invalid labels, missing/repeated images, non-finite output or a broken raw-count
interface. There are no silent skips, clipped predictions or calibration factors.
Images use the existing RGB/ImageNet inference path, whole-frame bilinear
stride-eight resizing and the **explicit reported** pixel budget, not cropping.
Source images remain unchanged. Image hashes, checkpoint hash/epoch, manifest hash,
ground-truth provenance and source/model shapes are recorded. No target maps are
needed for this count-only benchmark.

Reports: `results.json` and `observations.csv`. They contain aggregate MAE/RMSE,
signed bias, worst errors, condition groups and count bands 0, below 50, 50-199,
and 200+. These bands are analytical, not safety thresholds. An absent band is
null, not a perfect zero error. Zero-count scenes are handled without MAPE or
division by zero. A count benchmark does **not** validate ROI localization.

Freeze the baseline once. For future training, derive a **fixed validation subset
from training data**, keep related scenes together, and reserve the official test
split for the final comparison. Do not choose epochs or thresholds on test scores.
Because test data has already been used in historical project experiments, describe
each evaluation's actual role honestly rather than calling every result untouched.

## Other annotated image sources

A JSON manifest can describe complete human-reviewed target-camera annotations or
another dataset, without adding a Dataset class:

```json
{
  "schema_version": 1,
  "dataset": "Target camera review",
  "split": "validation",
  "ground_truth_source": "Reviewed complete head counts from approved footage",
  "ground_truth_kind": "human-reviewed",
  "annotations_complete": true,
  "samples": [
    {"id": "gate_night_001", "image": "images/gate_001.jpg", "true_count": 42, "condition": "night"}
  ]
}
```

The JSON above is a schema example, **not real annotation data or a measured
result**. Relative image paths resolve from the manifest directory. The application
cannot prove annotation quality from a flag; a human must verify completeness.
AI draft/incomplete annotations are rejected as benchmark ground truth.

## Recommended next training experiment

Keep epoch-16 Mall weights and existing results untouched. First run the frozen
baseline on the permitted new validation set. Then, on a GPU, train/fine-tune a
separate CSRNet experiment with denser/diverse training scenes, including empty
and low-count examples where available. Keep preprocessing, image scale, target
mass preservation and the independent evaluation protocol explicit. Compare MAE,
RMSE and signed bias by condition/count before adopting new weights.

Do not change the dashboard checkpoint/hash merely to accept a new model. Adopt
only an independently validated artifact with its own recorded configuration and
updated model manifest. A multiplicative count correction on HT21 is not a
substitute for this experiment.
