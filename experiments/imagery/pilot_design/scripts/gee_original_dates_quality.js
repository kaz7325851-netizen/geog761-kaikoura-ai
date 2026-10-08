// Pixel-quality check only. No persistent export, training or prediction.
var region = ee.Geometry.Rectangle([1667250,5321630,1673650,5328030], 'EPSG:2193', false);
var grid = [10,0,1667250,0,-10,5328030];
var scoreCollection = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
var probabilityCollection = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY');
var optical = ee.ImageCollection('COPERNICUS/S2_HARMONIZED').filterBounds(region).filter(ee.Filter.eq('MGRS_TILE','59GQP'));
function getQuality(start,end,name) {
  var candidates=optical.filterDate(start,end).sort('system:time_start');
  var image=ee.Image(candidates.first());
  var id=image.get('system:index');
  var scores=ee.Image(scoreCollection.filter(ee.Filter.eq('system:index',id)).first());
  print(name+'_metadata', image.toDictionary(['system:index','system:time_start','MGRS_TILE','CLOUDY_PIXEL_PERCENTAGE','PROCESSING_BASELINE']));
  print(name+'_candidate_count',candidates.size());
  print(name+'_score_count',scoreCollection.filter(ee.Filter.eq('system:index',id)).size());
  print(name+'_probability_count',probabilityCollection.filter(ee.Filter.eq('system:index',id)).size());
  var bands=image.select(['B2','B3','B4','B8','QA60']).toFloat()
    .addBands(scores.select(['cs','cs_cdf']).toFloat());
  var clipped=bands.clip(region).unmask(-9999,false);
  print(name+'_TOA_quality_download', clipped.getDownloadURL({name:name+'_TOA_quality',region:region,crs:'EPSG:2193',crs_transform:grid,format:'GEO_TIFF',filePerBand:false}));
  Map.addLayer(image.select(['B4','B3','B2']),{min:0,max:3000},name+'_RGB');
}
getQuality('2016-11-02','2016-11-03','pre_20161102');
getQuality('2016-11-22','2016-11-23','post_20161122');
Map.addLayer(region,{color:'yellow'},'Fixed SH1 pilot');
Map.centerObject(region,12);
