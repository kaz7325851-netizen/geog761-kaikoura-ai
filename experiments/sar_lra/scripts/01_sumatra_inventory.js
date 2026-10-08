// Fixed pilot BEFORE model predictions, inside author's Sumatra reference rectangle.
// Read-only public-image inventory; no cloud assets or Drive export.
var aoi = ee.Geometry.Rectangle([99.985, 0.060, 100.045, 0.120], null, false);
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(aoi)
  .filter(ee.Filter.eq('instrumentMode','IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VH'));
var preStart='2021-12-26', preEnd='2022-02-24';
var postStart='2022-02-26', postEnd='2022-03-10';
function metadata(c) {
  return c.toList(c.size()).map(function(x) {
    var im=ee.Image(x);
    return ee.Dictionary({id:im.get('system:index'),time:im.date().format('YYYY-MM-dd HH:mm'),
      pass:im.get('orbitProperties_pass'),rel:im.get('relativeOrbitNumber_start'),
      bands:im.bandNames(), coverage:im.geometry().intersection(aoi,10).area(10).divide(aoi.area(10))});
  });
}
print('SAR_LRA_PRE',metadata(s1.filterDate(preStart,preEnd)));
print('SAR_LRA_POST',metadata(s1.filterDate(postStart,postEnd)));
print('PILOT_BOUNDS',[99.985,0.060,100.045,0.120]);
Map.centerObject(aoi,12);
Map.addLayer(aoi,{color:'yellow'},'SAR-LRA fixed Sumatra pilot');
