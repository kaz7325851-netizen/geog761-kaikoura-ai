# Label meaning and limitations

The point configuration is a frozen record of reviewed **proposals**, not a
training-ready ground-truth dataset. It intentionally contains no image pixels
or feature vectors. The portable CLI extracts VV features only when the caller
supplies the corresponding grids and explicitly accepts weak references.

- Positive proposals derive from official GNS V3 source/debris footprints, with
  the historical 4×4 subpixel coverage estimate >=75% and numerical/geometric
  screening. This is an experimental boundary rule, not an official requirement.
- Negative proposals come from 20 coarsely reviewed 3×3 locations, with guards
  around official footprints, the older inventory and official centroids, plus
  optical-quality screening. Inventory absence alone was not treated as class 0.
- Optical review used 2 November and 22 November UTC imagery. The later optical
  image is seven days after the VV post-event image. Optical review is not an RF
  predictor and does not establish slide timing on 15 November.
- V3 includes mainshock and aftershock slides and does not date every object.
  Small slides can be omitted. Geometric masks are approximations, not certified
  radar visibility or road safety.
- Spatial roles are frozen; do not randomly split adjacent pixels, change the
  split after observing scores, or reuse evaluation labels for tuning. Pixels
  in the same review group/object are correlated, not independent examples.
- The initial optical regions and model results were already seen. The test
  regions are not blind new-event tests. The sampled class ratio is artificial.
- A slide near a road does not prove blockage; bridge damage, narrow obstacles,
  road topology, vehicle limits and temporal road records remain separate issues.

The sampling CLI checks provenance, duplicate locations and group/object crossing
between spatial roles. These checks do not approve the labels. All output manifests
retain `training_ready=false` and `training_performed=false`.
