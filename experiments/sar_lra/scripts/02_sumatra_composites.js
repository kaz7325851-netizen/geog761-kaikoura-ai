// Fixed pilot; matches published 60-day pre / 12-day post dB median method.
// Separate passes AND relative orbits, no mixing acquisition geometries.
var aoi=ee.Geometry.Rectangle([99.985,0.060,100.045,0.120],null,false);
var s1=ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(aoi)
 .filter(ee.Filter.eq('instrumentMode','IW'))
 .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV'))
 .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VH'));
function make(pass,rel){
 var c=s1.filter(ee.Filter.eq('orbitProperties_pass',pass))
         .filter(ee.Filter.eq('relativeOrbitNumber_start',rel));
 var pre=c.filterDate('2021-12-26','2022-02-24').select(['VV','VH']).median();
 var post=c.filterDate('2022-02-26','2022-03-10').select(['VV','VH']).median();
 var diff=post.subtract(pre).rename(['diffVV','diffVH']);
 var valid=pre.mask().reduce(ee.Reducer.min()).and(post.mask().reduce(ee.Reducer.min())).rename('valid');
 var input=post.rename(['postVV','postVH']).addBands(diff).addBands(valid).toFloat().clip(aoi);
 print('SCENES_'+pass,JSON.stringify(c.filterDate('2021-12-26','2022-03-10').aggregate_array('system:index').getInfo()));
 input.unmask(-9999).getDownloadURL({name:'Sumatra_2022_'+pass+'_REL'+rel,
   bands:['postVV','postVH','diffVV','diffVH','valid'],region:aoi,scale:10,
   crs:'EPSG:4326',format:'GEO_TIFF',filePerBand:false},function(url,error){
     print('DOWNLOAD_'+pass,error || url);
   });
 Map.addLayer(input.select('diffVV'),{min:-6,max:6},'diffVV '+pass,pass==='DESCENDING');
}
make('ASCENDING',142);make('DESCENDING',164);
Map.centerObject(aoi,12);
