"""Download a user-provided GEE pixel URL and validate exact experimental inputs."""
from pathlib import Path
import argparse
from urllib.parse import urlparse
import urllib.request
import json
import hashlib
from datetime import datetime
import rasterio
from rasterio.transform import Affine
import numpy as np

ROOT = Path(__file__).resolve().parent

def validate_raster(path, config):
    with rasterio.open(path) as src:
        assert src.crs.to_epsg() == 2193, 'Wrong output CRS'
        assert src.shape == (640,640) and src.count == 10, 'Wrong grid or band count'
        assert all(x == 'float32' for x in src.dtypes), 'Not float32'
        assert src.transform == Affine(10,0,1645530,0,-10,5305720), 'Grid differs from fixed reference'
        data = src.read()
        descriptions = src.descriptions
    if all(descriptions):
        assert list(descriptions) == config['export_band_order'], 'Unexpected band descriptions/order'
    valid = data[4] == 1
    assert np.isin(data[4], [0,1,-9999]).all(), 'Invalid coverage band values'
    assert valid.any(), 'No common valid pre/post pixels'
    assert np.isfinite(data[:4,valid]).all() and (data[:4,valid] != -9999).all()
    assert np.allclose(data[2,valid], data[0,valid]-data[5,valid], atol=2e-5, rtol=1e-5), 'diffVV is not post minus pre'
    assert np.allclose(data[3,valid], data[1,valid]-data[6,valid], atol=2e-5, rtol=1e-5), 'diffVH is not post minus pre'
    recent_valid = valid & (data[9] == 1)
    assert np.allclose(data[8,recent_valid], data[7,recent_valid]-data[5,recent_valid], atol=2e-5, rtol=1e-5)
    shifts = data[8,recent_valid]
    return {'shape':[640,640], 'band_count':10, 'band_descriptions':list(descriptions),
            'common_valid_pixels':int(valid.sum()), 'common_valid_fraction':float(valid.mean()),
            'difference_band_checks_passed':True, 'recent_vv_common_valid_pixels':int(recent_valid.sum()),
            'recent_minus_old_pre_VV_dB_quantiles': np.quantile(shifts,[0,.05,.25,.5,.75,.95,1]).tolist() if len(shifts) else None,
            'VV_stability_is_not_VH_stability':True,
            'model_band_ranges':[{'band':name,'min':float(data[i,valid].min()),'max':float(data[i,valid].max())} for i,name in enumerate(config['model_band_order'])]}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True, help='DOWNLOAD_URL printed by the prepared GEE script')
    parser.add_argument('--metadata', type=Path, required=True, help='JSON from EXPERIMENT_METADATA_JSON')
    args = parser.parse_args()
    parsed = urlparse(args.url)
    assert parsed.scheme == 'https' and parsed.hostname == 'earthengine.googleapis.com', 'Only HTTPS Earth Engine pixel downloads accepted'
    assert '/thumbnails/' in parsed.path and parsed.path.endswith(':getPixels'), 'Not a GEE pixel URL'
    config = json.loads((ROOT/'experiment_config.json').read_text(encoding='utf-8'))
    metadata = json.loads(args.metadata.read_text(encoding='utf-8-sig'))
    for key in ['crs','crs_transform','dimensions','relative_orbit','pass']:
        assert metadata[key] == config[key], f'Metadata mismatch: {key}'
    assert metadata['band_order'] == config['export_band_order']
    assert metadata['platform'] == 'A'
    assert metadata['pre_scene_ids'] and all('20160823T' in x and '1SDV' in x for x in metadata['pre_scene_ids'])
    assert metadata['post_scene_ids'] and all('20161115T' in x and '1SDV' in x for x in metadata['post_scene_ids'])
    assert metadata['recent_vv_scene_ids'] and all(any(day+'T' in x for day in ['20160916','20161010','20161103']) for x in metadata['recent_vv_scene_ids'])
    out = ROOT/'downloads'/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    dest = out/'ASCENDING_REL52.tif'
    partial = dest.with_suffix('.partial')
    with urllib.request.urlopen(args.url, timeout=120) as response, partial.open('wb') as output:
        size = 0
        while chunk := response.read(1024*1024):
            size += len(chunk)
            if size > 40_000_000:
                raise ValueError('Unexpectedly large response; stopped at 40MB')
            output.write(chunk)
    assert partial.read_bytes()[:4] in (b'II*\x00',b'MM\x00*'), 'Download is not TIFF'
    checks = validate_raster(partial,config)
    partial.replace(dest)
    (out/'earthengine_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    report = {'input_file':str(dest), 'bytes':dest.stat().st_size,
              'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(), 'validation':checks,
              'source':'Earth Engine COPERNICUS/S1_GRD user-provided download URL',
              'metadata':metadata, 'inference_performed':False, 'training_performed':False}
    (out/'input_checks.json').write_text(json.dumps(report,indent=2), encoding='utf-8')
    (ROOT/'latest_input.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2), flush=True)

if __name__ == '__main__':
    main()
