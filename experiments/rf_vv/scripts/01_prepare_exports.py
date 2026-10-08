"""Freeze existing SH1 sampling geometries and generate a VV-only GEE export.

No model fitting or credentials. Generated files stay in this task directory.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'bbunet/expanded_test/20261007T210924Z/frozen_design.json'
old = json.loads(SOURCE.read_text(encoding='utf-8'))
regions = old['regions'] + [old['anchor']]
sites = [{'id': r['region_id'], 'bounds': r['bounds_nztm'],
          'grid': r['transform'], 'shape': r['shape'],
          'sh1_length_km': r['sh1_length_km']} for r in regions]
names = ['VV_pre', 'VV_post', 'dVV', 'VV_pre_median', 'dVV_median',
         'angle_pre', 'angle_post', 'elevation', 'slope_deg', 'aspect_deg',
         'alpha_r_pre', 'alpha_r_post', 'valid_pre', 'valid_post',
         'valid_median', 'geometry_visible_pre', 'geometry_visible_post']
design = {'created_utc': datetime.now(timezone.utc).isoformat(),
          'status': 'frozen geometry; GEE execution pending', 'sites': sites,
          'pre_date_utc': '2016-11-03', 'post_date_utc': '2016-11-15',
          'median_pre_dates_utc': ['2016-09-16', '2016-10-10', '2016-11-03'],
          'crs': 'EPSG:2193', 'pass': 'ASCENDING', 'relative_orbit': 52,
          'platform': 'A', 'nodata': -9999, 'band_order': names,
          'selection': 'Reuse all seven previously geometry-selected SH1 squares and the optical pilot anchor; no deletion based on RF scores or SAR coverage',
          'source_geometry_file': str(SOURCE),
          'source_geometry_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          'limitations': ['Pilot subset, not all SH1/Route70 or final team AOI',
                         'Prior sites originally limited by optical product footprint; not an unbiased event-wide sample',
                         'Prior optical model outputs and inventory counts are already known; this is not a blind new test',
                         'Local active distortion approximation only, no passive ray tracing',
                         'No radiometric terrain flattening; SRTM terrain is about30m, not native10m',
                         'No verified negative labels or train/test assignment yet'],
          'training': False, 'inference': False}
(ROOT / 'frozen_data_design.json').write_text(json.dumps(design, ensure_ascii=False, indent=2), encoding='utf-8')

source = 'var sites=' + json.dumps(sites) + ';\nvar names=' + json.dumps(names) + ';\n'
source += r"""
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
"""
(ROOT / '02_export_vv.js').write_text(source, encoding='utf-8')
print(json.dumps({'sites': len(sites), 'area_km2': len(sites)*40.96,
                  'script': str(ROOT/'02_export_vv.js'),
                  'per_export_uncompressed_bytes': 640*640*len(names)*4}, ensure_ascii=False))
