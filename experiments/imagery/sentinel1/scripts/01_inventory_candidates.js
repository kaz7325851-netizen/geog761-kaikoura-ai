// GEOG761 Group Project
// Step 1: inventory Sentinel-1 GRD scenes around the Kaikoura coastal corridor.
// Run this script in the Google Earth Engine Code Editor.

// Pilot area: Oaro / Kaikoura / Clarence coastal corridor.
// This is deliberately broader than the final training area so that suitable
// same-orbit pre- and post-event scenes can be identified first.
var studyArea = ee.Geometry.Rectangle([
  173.35, -42.72,
  174.13, -41.95
]);

// 2016 Kaikoura earthquake: 14 Nov 2016 00:02 NZDT = 13 Nov 2016 11:02 UTC.
var eventTime = ee.Date('2016-11-13T11:02:00');
var searchStart = '2016-01-01';
var searchEnd = '2017-02-01';

// Keep only comparable Sentinel-1 GRD acquisitions.
var candidates = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(studyArea)
  .filterDate(searchStart, searchEnd)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.eq('resolution_meters', 10))
  .filter(ee.Filter.listContains(
    'transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains(
    'transmitterReceiverPolarisation', 'VH'))
  .sort('system:time_start');

// Convert image metadata into a table that is easier to inspect and export.
var candidateList = candidates.toList(candidates.size());
var candidateTable = ee.FeatureCollection(candidateList.map(function(item) {
  var image = ee.Image(item);
  var acquisition = image.date();
  var period = ee.String(ee.Algorithms.If(
    acquisition.millis().lt(eventTime.millis()), 'pre', 'post'));

  return ee.Feature(null, {
    product_id: image.get('system:index'),
    acquisition_utc: acquisition.format('YYYY-MM-dd HH:mm:ss'),
    time_millis: acquisition.millis(),
    period: period,
    platform: image.get('platform_number'),
    orbit_pass: image.get('orbitProperties_pass'),
    relative_orbit: image.get('relativeOrbitNumber_start'),
    absolute_orbit: image.get('orbitNumber_start'),
    instrument_mode: image.get('instrumentMode'),
    resolution_m: image.get('resolution_meters'),
    polarisations: image.get('transmitterReceiverPolarisation')
  });
}));

var tableWithSummary = candidateTable.map(function(feature) {
  var summary = ee.String(feature.get('acquisition_utc'))
    .cat(' | ')
    .cat(ee.String(feature.get('period')))
    .cat(' | ')
    .cat(ee.String(feature.get('orbit_pass')))
    .cat(' | relative orbit ')
    .cat(ee.Number(feature.get('relative_orbit')).format('%.0f'))
    .cat(' | ')
    .cat(ee.String(feature.get('product_id')));
  return feature.set('scene_summary', summary);
});

print('Study area', studyArea);
print('Earthquake time (UTC)', eventTime);
print('Number of candidate scenes', candidates.size());
print('Candidate scene metadata', candidateTable.sort('time_millis'));
print('Compact scene list',
      tableWithSummary.sort('time_millis').aggregate_array('scene_summary'));
print('Scenes by period', candidateTable.aggregate_histogram('period'));
print('Scenes by orbit pass', candidateTable.aggregate_histogram('orbit_pass'));
print('Scenes by relative orbit',
      candidateTable.aggregate_histogram('relative_orbit'));

Map.centerObject(studyArea, 8);
Map.addLayer(studyArea, {color: 'yellow'}, 'Pilot study area');

// After choosing one orbit pass and one relative orbit, uncomment this block
// to export the inventory as a CSV file to Google Drive.
// Export.table.toDrive({
//   collection: candidateTable.sort('time_millis'),
//   description: 'Kaikoura_S1_candidate_inventory',
//   fileNamePrefix: 'kaikoura_s1_candidate_inventory',
//   fileFormat: 'CSV'
// });
