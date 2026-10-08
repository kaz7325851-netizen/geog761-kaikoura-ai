// GEOG761 exploration only, NOT the original SAR-LRA pre60/post12 deployment.
// Geometry-selected pilot frozen before inventory inspection or predictions.
// No Drive export or permanent cloud assets. Run in Earth Engine Code Editor.
var grid = [10, 0, 1645530, 0, -10, 5305720];
var dims = [640, 640];
var aoi = ee.Geometry.Rectangle([1645530,5299320,1651930,5305720], 'EPSG:2193', false);
var base = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(aoi)
  .filter(ee.Filter.eq('instrumentMode','IW'))
  .filter(ee.Filter.eq('orbitProperties_pass','ASCENDING'))
  .filter(ee.Filter.eq('relativeOrbitNumber_start',52))
  .filter(ee.Filter.eq('platform_number','A'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV'));
var dual = base.filter(ee.Filter.listContains('transmitterReceiverPolarisation','VH'));
var preScenes = dual.filterDate('2016-08-23','2016-08-24');
var postScenes = dual.filterDate('2016-11-15','2016-11-16');
var recentScenes = base.filterDate('2016-09-14T11:02:56Z','2016-11-13T11:02:56Z');
var metadata = {
  experiment: 'Kaikoura 2016 exploratory older dual-polarization baseline',
  final_team_aoi: false, crs:'EPSG:2193', crs_transform:grid, dimensions:dims,
  relative_orbit:52, pass:'ASCENDING', platform:'A',
  pre_dates:['2016-08-23','2016-08-24'], post_dates:['2016-11-15','2016-11-16'],
  pre_scene_ids:preScenes.aggregate_array('system:index').getInfo(),
  post_scene_ids:postScenes.aggregate_array('system:index').getInfo(),
  recent_vv_scene_ids:recentScenes.aggregate_array('system:index').getInfo(),
  band_order:['postVV','postVH','diffVV','diffVH','valid','preVV','preVH','recentPreVV','baselineVVShift','recentVVValid'],
  processing:'GEE dB-space medians; post minus pre; no additional normalization or speckle filtering',
  nodata:-9999,
  limitations:['82-day-old single-date pre baseline','Coverage masks do not remove SAR shadow/layover','VV baseline comparison cannot prove VH stability']
};
print('EXPERIMENT_METADATA_JSON', JSON.stringify(metadata));
if (!metadata.pre_scene_ids.length || !metadata.post_scene_ids.length || !metadata.recent_vv_scene_ids.length) {
  throw new Error('Required imagery absent from Earth Engine. Stop: do not fill missing bands.');
}
var pre = preScenes.select(['VV','VH']).median();
var post = postScenes.select(['VV','VH']).median();
var diff = post.subtract(pre).rename(['diffVV','diffVH']);
var valid = pre.mask().reduce(ee.Reducer.min()).and(post.mask().reduce(ee.Reducer.min()))
  .unmask(0,false).rename('valid');
var recent = recentScenes.select('VV').median().rename('recentPreVV');
var recentValid = recent.mask().unmask(0,false).rename('recentVVValid');
var shift = recent.subtract(pre.select('VV')).rename('baselineVVShift');
var pack = post.rename(['postVV','postVH']).addBands(diff).addBands(valid)
  .addBands(pre.rename(['preVV','preVH'])).addBands(recent).addBands(shift)
  .addBands(recentValid).select(metadata.band_order).toFloat().clip(aoi).unmask(-9999,false);
// 640x640x10 float32 bands is ~16.4MB before compression, under the 32MB limit.
pack.getDownloadURL({
  name:'Kaikoura_2016_ASC52_oldbaseline_exploration',
  bands:metadata.band_order, crs:'EPSG:2193', crs_transform:grid, dimensions:dims,
  format:'GEO_TIFF', filePerBand:false
}, function(url,error) {
  if (error) {print('DOWNLOAD_ERROR', error); return;}
  print('DOWNLOAD_URL', url);
});
Map.centerObject(aoi,12);
Map.addLayer(aoi,{color:'yellow'},'Temporary 40.96km2 pilot, NOT final team AOI');
Map.addLayer(pack.select('diffVV').updateMask(valid),{min:-6,max:6},'Exploratory dVV',true);
Map.addLayer(pack.select('diffVH').updateMask(valid),{min:-6,max:6},'Exploratory dVH',false);
Map.addLayer(pack.select('baselineVVShift').updateMask(valid.and(recentValid)),{min:-6,max:6},'Recent preVV minus Aug23 preVV (diagnostic only)',false);
