# Environments and reproducibility boundaries

## Portable CLI

`requirements-core.txt` records the installed versions used for packaging tests.
The CLI only imports NumPy, SciPy and Rasterio beyond the standard library.
Dependencies were not newly installed, and installation on a teammate's machine
has not been tested. Do not assume every wheel is available on every platform.
Rasterio/GDAL binary compatibility can require a platform-specific environment.

`requirements-research.txt` adds the geospatial packages observed during historical
data/reference processing. It is not a lockfile for all deep-learning experiments.

## Historical models

Use separate environments; do not merge the following requirements blindly.

- BBUnet: PyTorch and segmentation-models-pytorch 0.3.3, the fixed author source,
  the exact landslide/cloud checkpoints, NumPy, Rasterio, geometry and terrain
  helpers. Historical code also refers to a task-local runtime directory. The
  adapted `richdem_numpy_audit.py` helper requires upstream attribution review.
- CNN M_ALL_006: legacy Keras/TensorFlow-compatible saved models, NumPy, Rasterio,
  OpenCV and author Utilities; GDAL 3.0.2 was used for the controlled terrain
  comparison. Full RGB preprocessing remains unknown.
- SAR-LRA: TensorFlow with legacy Keras, h5py, author `utils.py`, strict matching
  orbit-specific checkpoint, NumPy/Rasterio and GIS packages.
- Prithvi: PyTorch, TerraTorch and the particular community checkpoint/config.
- Delft RF: historical Python 3.7/scikit-learn 0.20.3 and RSGIS/GDAL environment;
  not interchangeable with the current VV pipeline environment.
- SpaceNet8: PyTorch and the selected KARI architecture/checkpoint plus its
  upstream dependencies. Only the tested flood head is recorded here.

Exact original scripts and checkpoint/source assertions are retained in the
archives. Third-party architectures, utilities, binaries and environments are
not bundled. No historical inference has been rerun during packaging. Core
validation reports distinguish syntax, synthetic tests and real-array checks
from end-to-end model reproduction.
