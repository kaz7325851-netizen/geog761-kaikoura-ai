# Kaikoura landslide model investigation and VV data preparation

GEOG761 group project, research progress as of 8 October 2026.

This repository records our investigation of pretrained landslide models and
provides a portable Sentinel-1 VV feature-preparation and sample-extraction CLI.
The project ultimately concerns transport inland from candidate landing points
after the 2016 Kaikoura earthquake. The code here does **not** determine whether
a road is passable or whether a beach is safe for landing.

No local Random Forest has been trained. The 530 published sampling proposals
are weak references, not verified road-state labels. Different pretrained models
were tested on different data; the results are not an accuracy leaderboard.

## Repository layout

```
scripts/vv_pipeline.py       Portable preparation, validation and sampling CLI
gee/export_vv.js            Original GEE script for the eight fixed pilot grids
configs/                    Portable grids, band order, split and review proposals
experiments/                Historical experiment scripts with provenance
docs/                       Data guide, results, dependencies and limitations
tests/                      Synthetic, offline checks
provenance/                 Source hashes and packaging changes
```

The repository intentionally excludes satellite rasters, inventory shapefiles,
weights, third-party model architecture files, virtual environments, credentials
and temporary download URLs. Code for generating Word documents is also excluded.

## Start with the portable VV workflow

Use Python 3.12 or newer and a dedicated environment. The exact Python version
used for local validation is recorded in `provenance/validation.json`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-core.txt
python scripts/vv_pipeline.py --help
python -m unittest discover -s tests -v
```

The requirements record locally observed versions; installing a fresh environment
on another machine has not been verified. See [environment notes](docs/environments.md).

1. Run `gee/export_vv.js` in the Earth Engine Code Editor using your own authorised
   account/project. Save each resulting 17-band GeoTIFF as `<region_id>.tif` in
   a local input directory. See [data acquisition](docs/data_acquisition.md).
2. Prepare and validate features, without modifying the raw files:

```powershell
python scripts/vv_pipeline.py prepare --input-dir data/raw_vv --output-dir outputs/vv_features
python scripts/vv_pipeline.py validate --feature-dir outputs/vv_features
```

3. Only if accepting the explicitly provisional label interpretation, extract
   features at the existing review locations:

```powershell
python scripts/vv_pipeline.py samples --feature-dir outputs/vv_features --output-dir outputs/weak_samples --allow-weak-references
```

Run commands from the repository root, or pass explicit `--config`, `--split`
and `--proposals` paths. Input and output directories may be anywhere locally.
Choose a new output directory for each run; existing outputs are not overwritten.
No network request is made by the portable Python CLI.

The main baseline is VV on 3 November and 15 November 2016 **UTC**, Sentinel-1A,
ascending relative orbit 52. Eight 6.4 km squares cover 327.68 km², a pilot subset
of SH1 rather than the full project AOI. Eleven predictors use VV and terrain,
not Sentinel-2 or BBUnet predictions. The numerical processing matches the
historical feature script: 3×3 mean in linear power followed by 5×5 local dB
standard deviation, with invalid neighbourhoods excluded. This is not GLCM,
precise coregistration, or radiometric terrain flattening.

## Research scripts versus a portable application

`experiments/` preserves the scripts used during investigation, not a second
turnkey application. Many scripts expect dated directories, generated manifests,
author architectures and particular checkpoints. Missing external dependencies
are documented; syntax compilation does not prove end-to-end reproduction.
Do not execute all historical scripts in sequence without reviewing their input
requirements and write locations. Their embedded reports are historical snapshots.

The portable CLI reproduces VV features and extraction at the frozen 530 review
locations. It does not repeat the human review or independently regenerate the
polygon-derived label selection. The V3 selection and independent checks remain
documented in the archived RF scripts and [label notes](docs/labels_and_limits.md).

## Results and next work

See [experiment results](docs/experiment_results.md). Initial optical candidates
have input or replication limitations; SAR-LRA's tested Kaikoura configuration
missed its reference-positive windows; BBUnet detections declined in expanded
testing; CNN M_ALL_006 remained sensitive to uncertain RGB conversion.

Next work is a geographically held-out, weak-reference RF feasibility experiment,
subject to accepting the label interpretation. Later project claims require
contemporaneous road-status evidence and network/vehicle analysis. Missing data
and negative model predictions must not be equated with safe roads.

## Attribution and publication status

Sources and checkpoint provenance are listed in [source notes](docs/sources.md).
No blanket open-source licence is applied here: the team must choose a licence
for its own contributions and review upstream terms before publication. This
package does not establish redistribution permission for external data or models.
.gitattributes preserves file bytes so recorded source hashes survive checkouts.
Public repository: https://github.com/kaz7325851-netizen/geog761-kaikoura-ai
This is a code-only research snapshot; satellite arrays, checkpoints and local
QA outputs are not published. See `UPLOAD_CHECKLIST.md` before future updates.
