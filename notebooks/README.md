# Final Sentinel-2 analysis notebook

## Source and attribution

`final_optical_analysis.ipynb` is an unchanged archival copy of the Group 9
notebook supplied through Colab as `needscleaning.ipynb`:

https://colab.research.google.com/drive/1FcFVtmqUcOL1G4qNc4VPn4VAZ8hNidvI

Archived on 25 October 2026 with the repository owner's permission. The group
presentation credits Jasraj with the final code and figures. This upload does
not reassign authorship or grant a new licence for third-party code or data.

SHA-256 of the archived notebook:

`b5613deb0379f5f49c755d1c2a79c4b22c9cf75782625ac13bd116c96ba74541`

## What the notebook contains

- Recovery of an existing Drive-based project and cached SH1 results.
- Current OSM/OSRM Route 70 geometry and processing-grid planning.
- Calls to a pretrained change-detection model at a fixed threshold of 0.5.
- Route 70 detection merging and calls to road-segment assessment.
- Three-corridor matching against GNS Kaikoura landslide source points V2.
- Historical-image maps and presentation examples based on saved predictions.

## Reproduction limits

The notebook has not been rerun as part of this archival upload. It expects
Google Colab, a mounted personal Drive, an authorised Earth Engine project,
and an existing `GEOG761/igarss-landslide-delineation` directory. Do not assume
that opening it in a new account will reproduce the saved results.

Important external dependencies that are not included here are:

- `scripts/kaikoura_full_sh1_corridor.py`, the custom corridor runner loaded
  dynamically by the first cell. It implements model loading, scene selection,
  inference, merging and road classification.
- The compatible `src/model_utils.py` and associated model code/configuration.
- Cached SH1 inputs/results, per-tile rasters and the selected model checkpoint.

The underlying model paper and upstream code are by Monopoli, Montello and
Rossi (2024):

https://github.com/links-ads/igarss-landslide-delineation

The original notebook includes package installation, authentication, downloads
and file-writing cells. Inspect paths and dependencies before executing it.
Its displayed RGB mosaics may differ from the original inference dates and
must not be treated as a record of identical model inputs.

## Interpretation of saved results

The notebook reports 2,564 observable GNS source points across SH1 North,
SH1 South and Route 70. Pooled source-point matching is 59.52% within 50 m
and 69.38% within 100 m. These are retrospective point-to-polygon spatial
matching rates, not pixel accuracy, precision, F1 or road-closure accuracy.

Road classes indicate possible impingement, inspection priority, no nearby
detection, or insufficient imagery. A negative detection is not a safety
clearance. No local Random Forest training is implied by this final notebook.
