// GEOG761 Group Project
// Step 2: build the selected pre-event and post-event Sentinel-1 mosaics.

var studyArea = ee.Geometry.Rectangle([
  173.35, -42.72,
  174.13, -41.95
]);

// Relative orbit 52 is the only candidate with a full VV+VH ascending
// acquisition both before and immediately after the earthquake.
var orbit52 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(studyArea)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.eq('resolution_meters', 10))
  .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
  .filter(ee.Filter.eq('relativeOrbitNumber_start', 52))
  .filter(ee.Filter.listContains(
    'transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains(
    'transmitterReceiverPolarisation', 'VH'));

// Each acquisition is split into two adjacent products over the broad pilot
// area, so mosaic the two products belonging to the same acquisition date.
var preCollection = orbit52.filterDate('2016-08-23', '2016-08-24');
var postCollection = orbit52.filterDate('2016-11-15', '2016-11-16');

var pre = preCollection.mosaic().select(['VV', 'VH']).clip(studyArea);
var post = postCollection.mosaic().select(['VV', 'VH']).clip(studyArea);

// Because the bands are already in dB, subtraction represents a logarithmic
// change measure. Do not divide one dB image by another.
var change = post.subtract(pre).rename(['dVV', 'dVH']);

print('Pre-event products',
      preCollection.aggregate_array('system:index'));
print('Post-event products',
      postCollection.aggregate_array('system:index'));
print('Pre product count', preCollection.size());
print('Post product count', postCollection.size());
print('Selected orbit', 52);
print('Selected pass', 'ASCENDING');

Map.centerObject(studyArea, 9);
Map.addLayer(pre.select('VV'),
             {min: -25, max: 5}, 'Pre-event VV 2016-08-23');
Map.addLayer(post.select('VV'),
             {min: -25, max: 5}, 'Post-event VV 2016-11-15', false);
Map.addLayer(change.select('dVV'),
             {min: -6, max: 6,
              palette: ['08306b', '4292c6', 'f7f7f7', 'fb6a4a', '99000d']},
             'VV change post minus pre', false);
Map.addLayer(change.select('dVH'),
             {min: -6, max: 6,
              palette: ['08306b', '4292c6', 'f7f7f7', 'fb6a4a', '99000d']},
             'VH change post minus pre', false);
Map.addLayer(studyArea, {color: 'yellow'}, 'Pilot study area', false);

// Export commands will be added after checking image coverage and agreeing on
// the final road-based training area. Exporting this broad pilot rectangle now
// would create unnecessarily large files.
