var sites=[{"id": "A", "grid": [10, 0, 1608840, 0, -10, 5293230], "dimensions": [640, 640], "crs": "EPSG:2193", "aoi_nztm": [1608840.0, 5286830.0, 1615240.0, 5293230.0]}, {"id": "B", "grid": [10, 0, 1618220, 0, -10, 5301980], "dimensions": [640, 640], "crs": "EPSG:2193", "aoi_nztm": [1618220.0, 5295580.0, 1624620.0, 5301980.0]}, {"id": "C", "grid": [10, 0, 1652460, 0, -10, 5313700], "dimensions": [640, 640], "crs": "EPSG:2193", "aoi_nztm": [1652460.0, 5307300.0, 1658860.0, 5313700.0]}, {"id": "OLD_GEO", "crs": "EPSG:4326", "grid": [8.983152841195213e-05, 0, 173.5532351825287, 0, -8.983152841195213e-05, -42.39908013326351], "dimensions": [873, 647], "aoi_nztm": [1645530, 5299320, 1651930, 5305720], "limitations": "Same AOI and scenes, nominal geographic10m grid.64pixel ground width differs; not exact paired physical windows or proof of author original training grid."}];

// Frozen diagnostic sites: all results retained. No model or threshold changes.
var ids=['S1A_IW_GRDH_1SDV_20160823T072219_20160823T072248_012724_01403F_DA98','S1A_IW_GRDH_1SDV_20161115T072214_20161115T072243_013949_016775_19DF','S1A_IW_GRDH_1SDV_20161115T072243_20161115T072308_013949_016775_0C46'];
var ims=ids.map(function(id){return ee.Image('COPERNICUS/S1_GRD/'+id);});
var pre=ims[0].select(['VV','VH']);
var post=ee.ImageCollection.fromImages([ims[1],ims[2]]).select(['VV','VH']).median();
var diff=post.subtract(pre).rename(['diffVV','diffVH']);
var valid=pre.mask().reduce(ee.Reducer.min()).and(post.mask().reduce(ee.Reducer.min())).unmask(0,false).rename('valid');
var modelPack=post.rename(['postVV','postVH']).addBands(diff).addBands(valid).addBands(pre.rename(['preVV','preVH']));
var modelNames=['postVV','postVH','diffVV','diffVH','valid','preVV','preVH'];
var dem=ee.Image('USGS/SRTMGL1_003').select('elevation');
var slope=ee.Terrain.slope(dem);var aspect=ee.Terrain.aspect(dem);
var geometryPack=dem.rename('elevation').addBands(slope.rename('slope_deg')).addBands(aspect.rename('aspect_deg'));
var geometryNames=['elevation','slope_deg','aspect_deg'];var headings=[];
ims.forEach(function(im,i){
 var heading=ee.Terrain.aspect(im.select('angle')).reduceRegion({reducer:ee.Reducer.mean(),geometry:im.geometry(),scale:1000,maxPixels:1e8}).get('aspect');
 headings.push(heading.getInfo());var proj=im.select('VH').projection();
 var alphaS=slope.multiply(Math.PI/180).setDefaultProjection(proj).clip(im.geometry());
 var phiS=aspect.multiply(Math.PI/180).setDefaultProjection(proj).clip(im.geometry());
 var alphaR=alphaS.tan().multiply(ee.Image.constant(heading).multiply(Math.PI/180).subtract(phiS).cos()).atan().multiply(180/Math.PI);
 var coverage=im.select(['VV','VH']).mask().reduce(ee.Reducer.min()).and(im.select('angle').mask());
 var names=['scene'+i+'_angle','scene'+i+'_alpha_r','scene'+i+'_coverage'];
 var extra=im.select('angle').addBands(alphaR).updateMask(coverage).addBands(coverage.unmask(0,false)).rename(names);
 geometryPack=geometryPack.addBands(extra);geometryNames=geometryNames.concat(names);
});
sites.forEach(function(s){
 var aoi=ee.Geometry.Rectangle(s.aoi_nztm,'EPSG:2193',false);
 var pack=modelPack;var names=modelNames;
 if(s.id!=='OLD_GEO'){pack=pack.addBands(geometryPack);names=names.concat(geometryNames);}
 var metadata={id:s.id,scene_ids:ids,headings_deg:headings,crs:s.crs,grid:s.grid,dimensions:s.dimensions,band_order:names,nodata:-9999,
  processing:'Original4 bands,dB,post-pre,nearest default export; SRTM local active visibility; no input radiometric slope correction',relative_orbit:52,pass:'ASCENDING'};
 print('SITE_METADATA_'+s.id,JSON.stringify(metadata));
 pack=pack.select(names).toFloat().clip(aoi).unmask(-9999,false);
 pack.getDownloadURL({name:'Kaikoura_confirmation_'+s.id,bands:names,crs:s.crs,crs_transform:s.grid,dimensions:s.dimensions,format:'GEO_TIFF',filePerBand:false},function(url,error){
  if(error){print('SITE_ERROR_'+s.id,error);return;}print('SITE_URL_'+s.id,url);
 });
 Map.addLayer(aoi,{color:s.id==='A'?'orange':s.id==='B'?'cyan':s.id==='C'?'green':'yellow'},'Frozen '+s.id,false);
});
Map.centerObject(ee.Geometry.Rectangle([1608840,5286830,1658860,5313700],'EPSG:2193',false),9);
