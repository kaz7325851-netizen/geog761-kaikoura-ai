"""Independent saved-result checks; no TensorFlow or model imports."""
from pathlib import Path
import json,csv,hashlib
import numpy as np,rasterio
from rasterio.features import rasterize
from shapely.geometry import shape,box
from shapely.ops import transform,unary_union
import pyproj
ROOT=Path(__file__).resolve().parent
out=Path(json.loads((ROOT/'latest_result.json').read_text(encoding='utf-8'))['result_directory'])
summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
manifest=json.loads((ROOT/'latest_download.json').read_text(encoding='utf-8'))
design=json.loads((ROOT/'frozen_design.json').read_text(encoding='utf-8'))
assert hashlib.sha256((ROOT/'frozen_design.json').read_bytes()).hexdigest()==summary['frozen_design_sha256']==manifest['frozen_design_sha256']
geoms=[box(*s['bounds_nztm']) for s in design['sites']]
assert all(a.intersection(b).area==0 for i,a in enumerate(geoms) for b in geoms[i+1:])
checks=[]
for record,stats in zip(manifest['records'],summary['sites']):
    sid=record['id'];site=out/sid
    path=Path(record['file']);assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']==stats['input_sha256']
    with rasterio.open(path) as src:data=src.read();grid=(src.shape,src.crs,src.transform)
    masks={}
    for name in ['raw_reference','clean_reference','visible_mask','land_proxy','near_inventory200','road500']:
        with rasterio.open(site/(name+'.tif')) as src:
            assert (src.shape,src.crs,src.transform)==grid
            masks[name]=src.read(1).astype(bool)
    assert np.array_equal(masks['clean_reference'],masks['raw_reference']&masks['visible_mask'])
    if sid!='OLD_GEO':
        known=[];bad=[]
        for i in range(3):
            angle,alpha,coverage=data[10+3*i:13+3*i]
            k=(coverage==1)&(alpha!=-9999)&(angle!=-9999)
            known.append(k);bad.append(k&((alpha>=angle)|(alpha<=angle-90)))
        visible=(data[4]==1)&known[0]&(known[1]|known[2])&~(bad[0]|bad[1]|bad[2])
        assert np.array_equal(visible,masks['visible_mask'])
        assert np.array_equal((data[7]>1)&(data[7]!=-9999),masks['land_proxy'])
        # Independent vector rasterization from archived preparation GeoJSON.
        inverse=pyproj.Transformer.from_crs(4326,2193,always_xy=True).transform
        vectors=[transform(inverse,shape(f['geometry'])) for f in json.loads((ROOT/'reference'/sid/'inventory_with_margin.geojson').read_text())['features']]
        if vectors:
            raw=rasterize([(v,1) for v in vectors],out_shape=grid[0],transform=grid[2],fill=0,dtype='uint8').astype(bool)
            near=rasterize([(unary_union(vectors).buffer(200),1)],out_shape=grid[0],transform=grid[2],fill=0,dtype='uint8').astype(bool)
        else:raw=np.zeros(grid[0],bool);near=raw.copy()
        assert np.array_equal(raw,masks['raw_reference']) and np.array_equal(near,masks['near_inventory200'])
    saved=np.load(site/'predictions.npz')
    expected=np.array([(x,y) for y in range(0,grid[0][0]-64,32) for x in range(0,grid[0][1]-64,32) if (data[4,y:y+64,x:x+64]==1).all()],dtype='int32')
    assert np.array_equal(expected,saved['coords'])
    rawf=[];cleanf=[];bg=[];road=[]
    for x,y in expected:
        sl=np.s_[y:y+64,x:x+64]
        r=masks['raw_reference'][sl].mean();c=masks['clean_reference'][sl].mean()
        rawf.append(r);cleanf.append(c)
        bg.append(r==0 and not masks['near_inventory200'][sl].any() and masks['visible_mask'][sl].mean()>=.95 and masks['land_proxy'][sl].mean()>=.95)
        road.append(masks['road500'][sl].any())
    rawf=np.array(rawf);cleanf=np.array(cleanf);bg=np.array(bg);road=np.array(road)
    assert np.array_equal(rawf,saved['raw_fraction']) and np.array_equal(cleanf,saved['clean_fraction'])
    assert np.array_equal(bg,saved['background']) and np.array_equal(road,saved['road_near'])
    assert np.array_equal(cleanf>.05,saved['positive']) and np.array_equal(saved['scores']>.6,saved['predicted'])
    nonoverlap=(expected[:,0]%64==0)&(expected[:,1]%64==0)
    assert np.array_equal(nonoverlap,saved['nonoverlap'])
    groups={'all':np.ones(len(expected),bool),'nonoverlap':nonoverlap,'marginal5to15':(cleanf>.05)&(cleanf<=.15),'dense_gt15':cleanf>.15,'road_near':road,'road_near_marginal':road&(cleanf>.05)&(cleanf<=.15)}
    for name,selection in groups.items():
        pos=(cleanf>.05)&selection;b=bg&selection;prd=saved['predicted'];m=stats['metrics'][name]
        for key,value in {'positive_windows':pos.sum(),'reference_background_windows':b.sum(),'TP':(pos&prd).sum(),'FN':(pos&~prd).sum(),'FP':(b&prd).sum(),'TN':(b&~prd).sum()}.items():assert m[key]==int(value)
        assert m['recall']==(float((pos&prd).sum()/pos.sum()) if pos.any() else None)
    rows=list(csv.DictReader((site/'windows.csv').open(encoding='utf-8-sig',newline='')))
    assert len(rows)==len(expected)
    for i,r in enumerate(rows):
        assert [int(r['x']),int(r['y'])]==expected[i].tolist() and float(r['score'])==float(saved['scores'][i])
    features=json.loads((site/'detections.geojson').read_text())['features']
    assert len(features)==stats['NMS_retained_boxes']
    checks.append({'id':sid,'status':'PASS','saved_window_count':len(expected),'positive_windows':int((cleanf>.05).sum()),
                   'nonoverlap_positive_windows':int(((cleanf>.05)&nonoverlap).sum()),'all_prediction_negative':bool(not saved['predicted'].any()),
                   'new_sites_vector_reference_verified':sid!='OLD_GEO','input_sha256_unchanged':True})
result={'status':'PASS','new_areas_do_not_overlap':True,'frozen_design_unchanged':True,'sites':checks,
 'checks':'All windows/fractions/eligibility/confusion counts and CSV rebuilt independently; source hashes and GIS grid checked',
 'note':'Non-overlapping windows reduce duplicate support, but still not independent landslide events.'}
(out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
