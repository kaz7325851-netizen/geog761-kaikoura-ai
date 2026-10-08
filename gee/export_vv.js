var sites=[{"id": "SH1-EXP-01", "bounds": [1640820.0, 5294490.0, 1647220.0, 5300890.0], "grid": [10, 0, 1640820.0, 0, -10, 5300890.0], "shape": [640, 640], "sh1_length_km": 8.28392201320763}, {"id": "SH1-EXP-02", "bounds": [1648830.0, 5300650.0, 1655230.0, 5307050.0], "grid": [10, 0, 1648830.0, 0, -10, 5307050.0], "shape": [640, 640], "sh1_length_km": 6.699346116026802}, {"id": "SH1-EXP-03", "bounds": [1653640.0, 5308680.0, 1660040.0, 5315080.0], "grid": [10, 0, 1653640.0, 0, -10, 5315080.0], "shape": [640, 640], "sh1_length_km": 8.16150245504909}, {"id": "SH1-EXP-04", "bounds": [1660240.0, 5315760.0, 1666640.0, 5322160.0], "grid": [10, 0, 1660240.0, 0, -10, 5322160.0], "shape": [640, 640], "sh1_length_km": 7.794061808958628}, {"id": "SH1-EXP-05", "bounds": [1672350.0, 5329240.0, 1678750.0, 5335640.0], "grid": [10, 0, 1672350.0, 0, -10, 5335640.0], "shape": [640, 640], "sh1_length_km": 8.867710735796116}, {"id": "SH1-EXP-06", "bounds": [1674450.0, 5337610.0, 1680850.0, 5344010.0], "grid": [10, 0, 1674450.0, 0, -10, 5344010.0], "shape": [640, 640], "sh1_length_km": 7.032786414934404}, {"id": "SH1-EXP-07", "bounds": [1680150.0, 5345710.0, 1686550.0, 5352110.0], "grid": [10, 0, 1680150.0, 0, -10, 5352110.0], "shape": [640, 640], "sh1_length_km": 8.296396498622704}, {"id": "PILOT-ANCHOR", "bounds": [1667250.0, 5321630.0, 1673650.0, 5328030.0], "grid": [10, 0, 1667250.0, 0, -10, 5328030.0], "shape": [640, 640], "sh1_length_km": 9.029673551000752}];
var names=["VV_pre", "VV_post", "dVV", "VV_pre_median", "dVV_median", "angle_pre", "angle_post", "elevation", "slope_deg", "aspect_deg", "alpha_r_pre", "alpha_r_post", "valid_pre", "valid_post", "valid_median", "geometry_visible_pre", "geometry_visible_post"];

// VV-only RF DATA PREPARATION. No training, Drive export or persistent asset.
// Same exact output grids as previous geometry-selected SH1 squares.
var study=ee.Geometry.Rectangle([1640820,5294490,1686550,5352110], 'EPSG:2193',false);
var base=ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(study)
  .filter(ee.Filter.eq('instrumentMode','IW'))
  .filter(ee.Filter.eq('resolution_meters',10))
  .filter(ee.Filter.eq('platform_number','A'))
  .filter(ee.Filter.eq('orbitProperties_pass','ASCENDING'))
  .filter(ee.Filter.eq('relativeOrbitNumber_start',52))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV'));
var preC=base.filterDate('2016-11-03','2016-11-04');
var postC=base.filterDate('2016-11-15','2016-11-16');
var medC=base.filter(ee.Filter.or(
  ee.Filter.date('2016-09-16','2016-09-17'),
  ee.Filter.date('2016-10-10','2016-10-11'),
  ee.Filter.date('2016-11-03','2016-11-04')));
var ids={pre:preC.aggregate_array('system:index').getInfo(),
  post:postC.aggregate_array('system:index').getInfo(),
  median_pre:medC.aggregate_array('system:index').getInfo()};
if(!ids.pre.length || !ids.post.length || !ids.median_pre.length) {
  throw new Error('Required actual VV scenes missing; stop without substitute data.');
}
var dem=ee.Image('USGS/SRTMGL1_003').select('elevation');
var terrain=ee.Terrain.products(dem);
var slope=terrain.select('slope');var aspect=terrain.select('aspect');
var sceneMetadata=[];
function attachGeometry(collection,phase) {
  var list=collection.toList(collection.size());var output=[];
  for(var i=0;i<collection.size().getInfo();i++) {
    var im=ee.Image(list.get(i));
    // Direction estimate used in previous local angular QA; NOT ray tracing.
    var heading=ee.Terrain.aspect(im.select('angle')).reduceRegion({
      reducer:ee.Reducer.mean(),geometry:im.geometry(),scale:1000,maxPixels:1e8
    }).get('aspect').getInfo();
    if(heading===null){throw new Error('Missing geometry direction');}
    sceneMetadata.push({phase:phase,id:im.get('system:index').getInfo(),
      time_utc:ee.Date(im.get('system:time_start')).format('YYYY-MM-dd HH:mm:ss').getInfo(),
      heading_deg:heading,platform_heading_deg:im.get('platformHeading').getInfo()});
    var proj=im.select('VV').projection();
    var localSlope=slope.setDefaultProjection(proj).clip(im.geometry());
    var localAspect=aspect.setDefaultProjection(proj).clip(im.geometry());
    var alpha=localSlope.multiply(Math.PI/180).tan()
      .multiply(ee.Image.constant(heading).subtract(localAspect).multiply(Math.PI/180).cos())
      .atan().multiply(180/Math.PI).rename('alpha_r');
    var coverage=im.select('VV').mask().and(im.select('angle').mask());
    var known=coverage.and(alpha.mask()).and(dem.mask());
    var good=alpha.lt(im.select('angle')).and(alpha.gt(im.select('angle').subtract(90)));
    output.push(im.select(['VV','angle']).addBands(alpha.updateMask(coverage))
      .addBands(known.rename('geometry_known'))
      .addBands(good.updateMask(known).rename('geometry_good'))
      .addBands(known.and(good.not()).unmask(0,false).rename('geometry_bad')));
  }
  var c=ee.ImageCollection.fromImages(output);
  return {image:c.select(['VV','angle','alpha_r']).median(),
    known:c.select('geometry_known').max().unmask(0,false),
    visible:c.select('geometry_known').max().and(c.select('geometry_bad').max().not()).unmask(0,false)};
}
var preG=attachGeometry(preC,'pre');var postG=attachGeometry(postC,'post');
var pre=preG.image.select('VV');var post=postG.image.select('VV');
var median=medC.select('VV').median();
var pack=pre.rename('VV_pre').addBands(post.rename('VV_post'))
  .addBands(post.subtract(pre).rename('dVV'))
  .addBands(median.rename('VV_pre_median'))
  .addBands(post.subtract(median).rename('dVV_median'))
  .addBands(preG.image.select('angle').rename('angle_pre'))
  .addBands(postG.image.select('angle').rename('angle_post'))
  .addBands(dem.rename('elevation')).addBands(slope.rename('slope_deg')).addBands(aspect.rename('aspect_deg'))
  .addBands(preG.image.select('alpha_r').rename('alpha_r_pre'))
  .addBands(postG.image.select('alpha_r').rename('alpha_r_post'))
  .addBands(pre.mask().unmask(0,false).rename('valid_pre'))
  .addBands(post.mask().unmask(0,false).rename('valid_post'))
  .addBands(median.mask().unmask(0,false).rename('valid_median'))
  .addBands(preG.visible.rename('geometry_visible_pre'))
  .addBands(postG.visible.rename('geometry_visible_post'));
var common={ids:ids,scenes:sceneMetadata,pre_date_utc:'2016-11-03',post_date_utc:'2016-11-15',
  relative_orbit:52,pass:'ASCENDING',platform:'A',crs:'EPSG:2193',band_order:names,nodata:-9999,
  processing:'GEE GRD sigma0 dB; same-date overlapping slices median; no extra speckle filter or terrain flattening; default nearest output sampling; separate three-date pre median; local active angular QA only',
  geometry_limitations:'No passive ray tracing or error buffer; SRTM30m; approximation not proof of full radar visibility'};
print('RF_VV_COMMON_METADATA',JSON.stringify(common));
sites.forEach(function(s){
  var aoi=ee.Geometry.Rectangle(s.bounds,'EPSG:2193',false);
  var metadata={id:s.id,grid:s.grid,shape:s.shape,common:common};
  print('RF_VV_METADATA_'+s.id,JSON.stringify(metadata));
  pack.select(names).toFloat().clip(aoi).unmask(-9999,false).getDownloadURL({
    name:'RF_VV_'+s.id,bands:names,crs:'EPSG:2193',crs_transform:s.grid,
    dimensions:[s.shape[1],s.shape[0]],format:'GEO_TIFF',filePerBand:false
  },function(url,error){if(error){print('RF_VV_ERROR_'+s.id,error);return;}
    print('RF_VV_URL_'+s.id,url);});
  Map.addLayer(aoi,{color:'yellow'},s.id,false);
});
Map.centerObject(study,9);
Map.addLayer(pack.select('dVV'),{min:-6,max:6,palette:['2166ac','f7f7f7','b2182b']},'VV change Nov3-Nov15');
