# Region measurement and dense crowd evaluation update

## What changed

1. Named rectangular crowd-estimate regions in the existing local dashboard:
   draw, name, save, clear, display and export up to six regions. These integrate
   raw signed density with fractional boundary-cell coverage, not visualization
   pixels. Full-frame counts and the preserved epoch-16 checkpoint are unchanged.
2. Region layout edits start a new generation/history segment. Late GPU results
   from an old layout are discarded. Seek/restart preserve definitions; selecting
   a new video clears them. Stale/failed current region values clear with the
   full-frame metrics. JSON records definitions/policy, browser CSV adds numeric
   region columns. Overlapping regions are explicitly not additive.
3. A reusable annotated-image benchmark for any compatible CSRNet checkpoint,
   with aggregate/condition/crowd-band MAE, RMSE and signed bias, worst cases,
   per-image measurements, image/manifest/model hashes and explicit size policy.
   Incomplete/draft ground truth and missing/duplicate images are rejected.
4. An existing-annotation ShanghaiTech A/B reader creates count manifests from
   original MAT point labels. It does not download, train or regenerate density
   maps. MAT parsing is verified with nested struct/cell fixtures, not yet with
   a downloaded real ShanghaiTech archive.
5. Training helpers now import without the Mall dataset. Source tests explicitly
   block accidental Mall-loader imports. GitHub Actions defines CPU-only tests
   with locked direct dependency versions selected for the runner's Python/OS,
   plus frontend checks. CI never requires datasets, weights, Drive or CUDA.
6. Stronger Git exclusions protect generated PDF/QA folders, future datasets,
   and the results notebook with embedded imagery. API version 0.2.0 prevents
   the launcher reusing an older process solely because its model hash matches.

## Verification

- **71 Python tests passed**, including six new region tests and four count-
  benchmark/annotation-reader tests. All passed through the source-only runner
  with Mall-loader imports prohibited. Tiny training fixtures test mechanics;
  no real crowd model was retrained.
- **11 frontend regression tests passed**, including region freshness, old-layout
  result rejection and full-precision region export. These are mock-DOM tests.
- **Real browser and CUDA:** selected the existing HT21-01 clip, performed actual
  codec preparation, played it with updating counts/heatmap/history, drew and
  saved an Entrance rectangle, obtained a paused regional snapshot, restarted
  playback with the same layout and downloaded JSON. Browser error/warning logs
  were empty during those checks. Saved JSON had 14 observations, matching
  segment IDs, region definitions, exact epoch-16 fingerprint and browser latency.
  A later paused seek showed full-frame estimate 42.0 and regional estimate 3.5;
  these are displayed model estimates, **not human-verified head counts**.
- Paused snapshots expired under the existing wall-clock freshness rule, clearing
  current metrics while leaving historical observations exportable. A screenshot
  under `outputs/dashboard/zones_verified_20261104.jpg` shows the zone outline in
  this intentionally stale/paused state, not a permanent fresh measurement.
- **Real GPU numerical sanity:** three existing Mall validation frames (1401,
  1550, 1700) ran through the new count benchmark. Their original MAT annotation
  counts were used, not AI labels. On all three, the full-frame region equalled
  the raw total, and adjacent fractional partitions summed back with observed
  difference **0.0**. Report: `outputs/benchmarks/zones_sanity_20261104/results.json`.
  This is a tiny same-camera validation sanity check, not a new independent
  accuracy result or proof of improvement. It does not validate region accuracy.
- Checkpoint fingerprint is preserved:
  `622956dd59ba559f877abb6f76e3fc8ca73a6a7ccd6c4ac9f477be48a7a7ba43`.
- No new 15-minute soak, full browser failure matrix, dense-dataset training or
  external-dataset accuracy evaluation was performed in this update. The previous
  15-minute backend soak remains historical evidence, not a newly repeated run.

## Scope and next experiment

The priority is still **accuracy at a target camera**, not more UI features.
ShanghaiTech Part A is a sensible first denser-scene experiment; consider Part B
and UCF-QNRF for complementary distributions. Original sources:

- https://github.com/desenzhou/ShanghaiTechDataset
- https://www.crcv.ucf.edu/research/data-sets/ucf-qnrf/

Review media/dataset rights, acquire data separately, run the preserved baseline,
create a fixed validation subset from training data, and only then train a
separate model on representative data. Include empty/night/low-count failure
conditions as well as large crowds. Compare unchanged image-size policies,
count errors and bias before replacing weights. See `BENCHMARKING.md`.

Region counts are approximate image-space integrations. Gaussian smoothing,
coarse cells, camera perspective and domain shift limit localization. No physical
area, people/m², flow, bottleneck/danger detection or police recommendation was
added. Region definitions are currently in memory; persistent camera profiles,
polygon editing and temporal analysis are future work, not completed features.

## GitHub and preservation

Target supplied by the owner: https://github.com/polo784/crowd-monitoring-system-.
The destination initially had no branch heads. Source and curated documentation
are intended for push, not datasets, original/private footage, weights, generated
evaluation outputs or the results notebook's embedded images. No licence is
invented and no access/security setting is changed. Existing local historical
reports and unrelated deletion of the old main.py placeholder are left alone.
GitHub Actions is only considered verified after an actual hosted run completes;
local passing tests alone do not establish hosted CI success.
