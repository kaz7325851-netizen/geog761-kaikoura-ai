# Historical experiment archives

These are research-stage scripts, not portable independent applications. The
folder hierarchy mirrors the experiment names, but excluded assets, generated
manifests and fixed dated paths must be restored/configured for reruns. Read
each script before execution: some download public data or create reports.

No archives were automatically executed during packaging. Python syntax was
checked without importing module-level code. A syntax pass does not establish
successful inference or complete dependency reproduction.

The `rf_vv` archive includes all 16 investigation stages; stages 03/04 concern
an earlier pilot, while stage 16 renders a historically fixed report. The latest
portable CLI is `scripts/vv_pipeline.py`, not the archived numbered scripts.

BBUnet, SAR-LRA and CNN archives retain acquisition, inference and validation
scripts but omit author model definitions and all weights. Initial-candidate
archives cover U-Net, Prithvi, Delft RF and SpaceNet8. Supporting imagery-quality
and selection scripts are also retained where selected by the provenance list.

The packaging pass removes original workstation paths and literal temporary
Earth Engine download URLs from textual
snapshots. Hashes record original and packaged versions and identify any such
changes. Historical embedded progress text may be superseded by `docs/`.
URL placeholders require fresh authorised exports before a historical download
script can run; they are deliberately not usable links.
