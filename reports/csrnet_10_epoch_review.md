# Mall CSRNet: completed 10-epoch run

Reviewed on 2026-09-08 using the result JSON read directly from Google Drive.
The saved training notebook independently agrees with the final metrics.

## Results

| Metric | Value |
| --- | ---: |
| Completed epochs | 10 |
| Best epoch by validation MAE | 10 |
| Final training MSE | 0.00001254834 |
| Final validation MSE | 0.00002727153 |
| Validation MAE (people) | 1.622362 |
| Validation RMSE (people) | 2.077742 |
| Test MSE | 0.00002693925 |
| Test MAE (people) | 1.901865 |
| Test RMSE (people) | 2.363966 |
| Maximum recorded target count-preservation error | 5.4833e-7 people |

Configuration: pretrained VGG16 CSRNet, 16,263,489 parameters, Tesla T4,
batch size 4, Adam at 1e-5, MSE loss. Images are (B,3,480,640);
predictions and resized targets are (B,1,60,80). Original density maps remain
unchanged. Train/validation/test ranges are 1-1400/1401-1700/1701-2000.

The completed epochs total 45.37 minutes of recorded training plus validation.
This excludes setup, checkpoint writes, interrupted attempts, and final testing.
Epoch 10 took 272.9 seconds (4.55 minutes).

## Interpretation

Validation MAE fell from 4.37 at epoch 1 to 1.62 at epoch 10, with fluctuations
between epochs. The smallest validation MSE occurred at epoch 8, whereas the
smallest validation count MAE occurred at epoch 10: these measure different
things. Selecting epoch 10 is consistent with the configured selection rule.

Test MAE 1.90 means an average absolute count error of about 1.9 people over
300 test frames. It is not a percentage accuracy or a guarantee for each frame.
MSE measures pixel-level density error; MAE/RMSE here measure total crowd-count
error. Accurate total counts alone do not establish accurate spatial localization.

All splits use the same fixed camera. Adjacent CCTV frames remain related even
with contiguous splits, so these results do not establish performance on another
camera, station, or dataset. The test set was evaluated after the initial smoke
run and again after this continuation; it is not a fresh holdout for future tuning.
Use validation, not repeated test results, to guide any further model changes.

## Saved artifacts and evidence

- [Run folder](https://drive.google.com/drive/folders/1OF_Ay8UCJzM3oQcuezpoFyNz_FTrwyfE)
- [Original result JSON](https://drive.google.com/file/d/1nZgRIiY2xKV6PCj2V1_z_DjQjkn-mu92/view)
- [Best checkpoint](https://drive.google.com/file/d/1wpxMPW0flIH1gE9Fl-ok89UwS74WFZk-/view)
- [Latest checkpoint](https://drive.google.com/file/d/1D07yedA6Eg25rOYrmuOkqZizeFe2FWKe/view)
- [Three saved prediction panels](https://drive.google.com/drive/folders/1ja4DtQl3Pb6ZykpLc0xB05FT5lgNIZBR)

Both checkpoint files exist in Drive (195,211,947 bytes each). Their weights
were not downloaded or loaded during this review. All three PNG files were
downloaded through the Drive connector on 2026-09-08, checked against the listed
file sizes, and visually inspected. No training or evaluation was rerun.

`csrnet_10_epoch_results.json` beside this report preserves the fetched metrics.
`../Review_CSRNet_Results.ipynb` was executed locally: both code cells completed,
and its saved outputs contain the epoch table and all three prediction images.
Downloaded images are under `../outputs/csrnet_review/csrnet_predictions/`.
The notebook also supports mounting Drive in Colab without a GPU or dataset.

## Visual inspection

| Test frame | Ground truth | Prediction | Undercount |
| --- | ---: | ---: | ---: |
| 1701 | 21.00 | 18.61 | 2.39 |
| 1702 | 21.00 | 18.09 | 2.91 |
| 1703 | 21.00 | 16.73 | 4.27 |

Counts above are rounded as printed on the saved panels. Predicted hotspots
broadly align with ground-truth occupied regions: the distant group in the upper
left, the seated group near the tree, and people in the foreground. Large empty
floor areas are mostly dark. Predicted hotspots are smoother and differ in
intensity and shape from the targets. All three samples undercount, with the
largest discrepancy in frame 1703. These images alone do not establish a general
undercount bias or identify exactly which individuals were missed.

## Immediate next step

Open the executed results-only notebook to see the metrics and three images
without rerunning cells. These first three test frames are consecutive examples,
not a representative accuracy assessment. Review the project changes and record
this completed run before considering additional training or new applications.
