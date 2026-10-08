// GEOG761 Group Project
// Final Sentinel-1 processing and export for the Kaikoura road corridor.
// Study area: 2 km buffer around SH1 from Oaro to Kekerengu and Route 70
// from Kaikoura to Waiau. Road geometry is derived from OpenStreetMap.

__ROUTE_GEOMETRY__

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

var preCollection = orbit52.filterDate('2016-08-23', '2016-08-24');
var postCollection = orbit52.filterDate('2016-11-15', '2016-11-16');

var preRaw = preCollection.mosaic().select(['VV', 'VH']).clip(studyArea);
var postRaw = postCollection.mosaic().select(['VV', 'VH']).clip(studyArea);

// Apply a 3 x 3 mean filter in natural-power space to reduce SAR speckle.
function speckleFilterDb(dbImage) {
  var natural = ee.Image(10).pow(dbImage.divide(10));
  var filtered = natural.reduceNeighborhood({
    reducer: ee.Reducer.mean(),
    kernel: ee.Kernel.square({radius: 1, units: 'pixels'}),
    skipMasked: true
  });
  return filtered.log10().multiply(10).rename(['VV', 'VH']);
}

var pre = speckleFilterDb(preRaw).clip(studyArea);
var post = speckleFilterDb(postRaw).clip(studyArea);
var change = post.subtract(pre).rename(['dVV', 'dVH']);

// Derive GLCM contrast and entropy from the filtered VV and VH images.
function textureBands(image) {
  var vvByte = image.select('VV')
    .unitScale(-25, 5).clamp(0, 1).multiply(255).toByte()
    .rename('VV_tex');
  var vhByte = image.select('VH')
    .unitScale(-32, -5).clamp(0, 1).multiply(255).toByte()
    .rename('VH_tex');
  var vvTexture = vvByte.glcmTexture({size: 1})
    .select(['VV_tex_contrast', 'VV_tex_ent'],
            ['VV_contrast', 'VV_entropy']);
  var vhTexture = vhByte.glcmTexture({size: 1})
    .select(['VH_tex_contrast', 'VH_tex_ent'],
            ['VH_contrast', 'VH_entropy']);
  return vvTexture.addBands(vhTexture);
}

var preTexture = textureBands(pre);
var postTexture = textureBands(post);
var textureChange = postTexture.subtract(preTexture).rename([
  'dVV_contrast', 'dVV_entropy',
  'dVH_contrast', 'dVH_entropy'
]);

var modelStack = pre.rename(['VV_pre', 'VH_pre'])
  .addBands(post.rename(['VV_post', 'VH_post']))
  .addBands(change)
  .addBands(textureChange)
  .toFloat();

var validBoth = preRaw.select('VV').mask()
  .and(postRaw.select('VV').mask());
var coveredArea = ee.Number(
  ee.Image.pixelArea().updateMask(validBoth).reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: studyArea,
    scale: 100,
    maxPixels: 1e9
  }).get('area')
);
var coveragePercent = coveredArea.divide(studyArea.area()).multiply(100);

print('Pre product count', preCollection.size());
print('Post product count', postCollection.size());
print('Study-area coverage percent', coveragePercent);
print('Export bands', modelStack.bandNames());

Map.centerObject(studyArea, 8);
Map.addLayer(routeLines, {color: '00ffff'}, 'Selected road corridors');
Map.addLayer(studyArea, {color: 'ffff00'}, '2 km study area');
Map.addLayer(pre.select('VV'),
             {min: -25, max: 5}, 'Filtered pre-event VV', false);
Map.addLayer(post.select('VV'),
             {min: -25, max: 5}, 'Filtered post-event VV', false);
Map.addLayer(change.select('dVV'),
             {min: -6, max: 6,
              palette: ['08306b', '4292c6', 'f7f7f7', 'fb6a4a', '99000d']},
             'VV change', true);

Export.image.toDrive({
  image: modelStack.unmask(-9999),
  description: 'Kaikoura_S1_route_corridor_10m',
  folder: 'GEOG761',
  fileNamePrefix: 'Kaikoura_S1_route_corridor_20160823_20161115',
  region: studyArea,
  scale: 10,
  crs: 'EPSG:2193',
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {
    cloudOptimized: true,
    noData: -9999
  }
});

Export.table.toDrive({
  collection: ee.FeatureCollection([
    ee.Feature(studyArea, {buffer_m: 2000})
  ]),
  description: 'Kaikoura_S1_study_area_2km',
  folder: 'GEOG761',
  fileNamePrefix: 'Kaikoura_S1_study_area_2km',
  fileFormat: 'GeoJSON'
});
