"""Read-only source inspection; never executes downloaded author code."""
from pathlib import Path
import json, hashlib, xml.etree.ElementTree as ET
import numpy as np, rasterio

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'preprocessing_audit'
OUT.mkdir(exist_ok=True)
nb = json.loads((ROOT/'sources/LS_mapping.ipynb').read_text(encoding='utf-8'))
lines = []
for i, cell in enumerate(nb['cells']):
    lines.append(f'\n### CELL {i} ({cell["cell_type"]})\n')
    lines.append(''.join(cell.get('source', [])))
(OUT/'notebook_source.txt').write_text('\n'.join(lines), encoding='utf-8')
paper = ET.parse(ROOT/'sources/paper_fulltext.xml').getroot()
paragraphs = []
for elem in paper.iter():
    if elem.tag in ['p', 'title', 'caption', 'table']:
        paragraphs.append(' '.join(''.join(elem.itertext()).split()))
(OUT/'paper_text.txt').write_text('\n\n'.join(paragraphs), encoding='utf-8')
tree = json.loads((ROOT/'sources/repository_tree.json').read_text())
(OUT/'repository_paths.txt').write_text('\n'.join(x['path'] for x in tree['tree']), encoding='utf-8')
paths = sorted((ROOT/'sources/RawData/AOI_E_S2-S2').rglob('*.tif'))
stats = []
for path in paths:
    with rasterio.open(path) as ds:
        record = {'file':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'shape':[ds.height,ds.width], 'count':ds.count, 'dtypes':list(ds.dtypes),
            'crs':str(ds.crs), 'transform':list(ds.transform)[:6], 'nodata':ds.nodata,
            'descriptions':list(ds.descriptions), 'tags':ds.tags(), 'band_tags':[], 'stats':[]}
        for b in range(1,ds.count+1):
            a=ds.read(b,masked=True).compressed();a=a[np.isfinite(a)]
            record['band_tags'].append(ds.tags(b))
            record['stats'].append({'band':b,'quantiles_0_1_2_25_50_75_98_99_100':np.percentile(a,[0,1,2,25,50,75,98,99,100]).tolist(),
                'fraction_zero':float(np.mean(a==0)), 'fraction_255':float(np.mean(a==255))})
        stats.append(record)
(OUT/'author_raster_metadata.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
print(json.dumps({'notebook_cells':len(nb['cells']),'rasters':len(stats),'output':str(OUT)}))
