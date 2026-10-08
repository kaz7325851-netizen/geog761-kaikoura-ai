# Sources and attribution

These links identify the implementations/datasets investigated. Availability and
licensing are not guarantees that the models fit the road-access problem.

- Sentinel-1 GRD: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD
- SRTM: https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003
- GNS V3 inventory publication: https://doi.org/10.17603/ds2-rpyg-0h84
- GNS record: https://doi.org/10.21420/WX2X-H603
- SAR-LRA: https://github.com/lorenzonava96/SAR-and-DL-for-Landslide-Rapid-Assessment
- BBUnet: https://github.com/links-ads/igarss-landslide-delineation
  (historical source commit `ce34d3ac02b80d4db7e542203b7e534649a46352`)
- CNN M_ALL_006: https://github.com/nprksh/landslide-mapping-with-cnn
  (historical source commit `2268ee5aab022454a21344c3601069afd6054690`)
- Delft RF: https://github.com/mhscience/landslides_detection
- SpaceNet8 KARI: https://github.com/SpaceNetChallenge/SpaceNet8/tree/main/01-ohhan777/code
- Community Prithvi: https://huggingface.co/srivassid/prithvi-landslide
- Landslide4Sense: author code and sample records remain in the historical U-Net
  archive; no retrieved author checkpoint is included.
- RichDEM terrain-formula source: https://github.com/r-barnes/richdem

GNS publication metadata declares CC BY 4.0; preserve dataset attribution and
its completeness/fitness disclaimers. This does not license all other assets.
The team has not selected a blanket licence for this repository. Do not label
third-party models, formulas, training or imagery as original project contributions.

Model weights and author architecture code are not redistributed. The original
VV feature formulas were modularised here; historical wrappers and checks retain
their source hashes in `provenance/archive_manifest.json`.
