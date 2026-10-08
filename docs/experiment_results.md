# Experiment findings

Progress as of 8 October 2026. No candidate was retrained or fine-tuned.

| Candidate | Actual work | Finding and limitation |
|---|---|---|
| Landslide4Sense U-Net | Data and random-initialisation architecture checks | Author checkpoint unavailable; no valid accuracy evaluation |
| Community Prithvi landslide model | Actual weights on 20 public labelled patches | Pixel recall 52.23%, F1 62.66%; not Kaikoura road accuracy; training/test independence not audited |
| Delft/Deltares RF | Actual saved model on author example | Optical change and terrain predictors, not SAR; no reference labels for accuracy; segmentation scale and legacy dependencies are concerns |
| SpaceNet8 KARI flood checkpoint | Actual weights on three public image pairs | High-resolution RGB flood task; not direct evidence for earthquake landslides or Sentinel-scale roads |
| SAR-LRA VV+VH configuration | Input checks, label screening and repeat Kaikoura inference | 125 positive reference windows across three new areas were missed at the fixed threshold; not 125 independent slides; older baseline deviated from original configuration |
| CNN M_ALL_006 | Author example, local exploratory inference and terrain audit | Terrain derivatives were aligned, but RGB conversion sensitivity remained; full preprocessing reproduction unresolved |
| BBUnet | Author weights, initial pilot and seven additional SH1 regions | Of 56 near-road reference objects, 28 were evaluable; 10 had prediction overlap and 18 did not. Another 28 were unevaluable. This is reference overlap, not road accuracy |

These are not comparable scores from a shared independent test set. Non-detection
does not establish passability; inventory omissions also prevent treating all
unmapped predictions as false alarms.

## Current data status

Eight VV regions were downloaded and checked. Eleven feature bands were prepared.
Official GNS V3 source/debris polygons improved reference coverage relative to the
earlier author archive. All 54 proposed review locations received coarse visual
screening. The frozen point proposals contain 350 event-reference positive pixels
and 180 background pixels from 20 groups, totalling 530 correlated sample pixels.

The 5/1/2 geographic region split yields 314 training, 81 validation and 135 test
pixel proposals. It is within-event spatial separation, not a blind or cross-event
validation. No RF fitting, threshold calibration, road scoring or routing has run.
