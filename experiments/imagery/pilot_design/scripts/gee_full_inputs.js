// Staging only: these are L1C TOA data, NOT L2A or final model tensors.
var region=ee.Geometry.Rectangle([1667250,5321630,1673650,5328030],'EPSG:2193',false);
var grid=[10,0,1667250,0,-10,5328030];
var bands=['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12'];
var collection=ee.ImageCollection('COPERNICUS/S2_HARMONIZED').filterBounds(region).filter(ee.Filter.eq('MGRS_TILE','59GQP'));
function optical(start,end,name){
  var sourceId=name==='pre_20161102'?'20161102T221602_20161102T221601_T59GQP':'20161122T221602_20161122T232031_T59GQP';
  var image=ee.Image(collection.filterDate(start,end).filter(ee.Filter.eq('system:index',sourceId)).first());
  print(name+'_source',image.get('system:index'));
  var selected=image.select(['B2','B3','B4','B8']).toFloat()
    .addBands(image.select(['B1','B5','B6','B7','B8A','B9','B11','B12']).toFloat().resample('bilinear'))
    .select(bands).clip(region).unmask(-9999,false);
  print(name+'_full_TOA_download',selected.getDownloadURL({name:name+'_full_TOA',region:region,crs:'EPSG:2193',crs_transform:grid,format:'GEO_TIFF',filePerBand:false}));
  Map.addLayer(image.select(['B4','B3','B2']),{min:0,max:3000},name+'_RGB');
}
optical('2016-11-02','2016-11-03','pre_20161102');
optical('2016-11-22','2016-11-23','post_20161122');
var demCollection=ee.ImageCollection('JAXA/ALOS/AW3D30/V4_1').filterBounds(region.buffer(3000));
print('terrain_source_tiles',demCollection.aggregate_array('system:index'));
var nativeProjection=ee.Image(demCollection.first()).select('DSM').projection();
var mosaic=demCollection.mosaic().setDefaultProjection(nativeProjection);
// Derivatives computed on a metric 30 m grid, not on artificial 10 m detail.
var dsm30=mosaic.select('DSM').toFloat().resample('bilinear').reproject({crs:'EPSG:2193',scale:30});
var products=ee.Terrain.products(dsm30);
var aspect=products.select('aspect');
var terrain=dsm30.rename('DSM_m').addBands(products.select('slope').rename('slope_deg'))
  .addBands(aspect.rename('aspect_deg'))
  .addBands(ee.Terrain.hillshade(dsm30,315,45).rename('hillshade_315_45'))
  .addBands(aspect.multiply(Math.PI/180).sin().rename('aspect_sin'))
  .addBands(aspect.multiply(Math.PI/180).cos().rename('aspect_cos'))
  .addBands(mosaic.select(['MSK','STK']));
print('terrain_10m_download',terrain.toFloat().clip(region).unmask(-9999,false).getDownloadURL({name:'ALOS_terrain_staged_10m',region:region,crs:'EPSG:2193',crs_transform:grid,format:'GEO_TIFF',filePerBand:false}));
var terrainRegion=ee.Geometry.Rectangle([1667160,5321520,1673760,5328120],'EPSG:2193',false);
print('terrain_30m_audit_download',terrain.toFloat().clip(terrainRegion).unmask(-9999,false).getDownloadURL({name:'ALOS_terrain_30m_audit',region:terrainRegion,crs:'EPSG:2193',crs_transform:[30,0,1667160,0,-30,5328120],format:'GEO_TIFF',filePerBand:false}));
Map.addLayer(region,{color:'yellow'},'Fixed SH1 pilot');
Map.centerObject(region,12);
