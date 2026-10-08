"""Portable VV feature preparation and explicit weak-reference sample extraction.

No model fitting, predictions, optical features, or road-safety classifications.
Paths are provided by CLI; historical experiment scripts are not imported.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy.ndimage import uniform_filter, binary_erosion

FEATURE_NAMES = ['VV_pre_filtered_db', 'VV_post_filtered_db', 'dVV_filtered_db',
                 'abs_dVV_filtered_db', 'pre_local_std_5x5_db', 'post_local_std_5x5_db',
                 'delta_local_std_5x5_db', 'elevation_m', 'slope_deg', 'aspect_sin', 'aspect_cos']
NODATA = -9999

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')

def filtered_db(db, valid):
    power = np.power(10., np.where(valid, db, 0) / 10)
    counts = uniform_filter(valid.astype('float64'), size=3, mode='constant')
    sums = uniform_filter(np.where(valid, power, 0).astype('float64'), size=3, mode='constant')
    return 10 * np.log10(np.maximum(np.divide(sums, counts, out=np.ones_like(sums), where=counts > 0), 1e-12))

def local_std(db, valid):
    counts = uniform_filter(valid.astype('float64'), size=5, mode='constant')
    mean = np.divide(uniform_filter(np.where(valid, db, 0).astype('float64'), size=5, mode='constant'),
                     counts, out=np.zeros_like(counts), where=counts > 0)
    second = np.divide(uniform_filter(np.where(valid, db ** 2, 0).astype('float64'), size=5, mode='constant'),
                       counts, out=np.zeros_like(counts), where=counts > 0)
    return np.sqrt(np.maximum(second - mean ** 2, 0))

def check(condition, message):
    if not condition:
        raise ValueError(message)

def compute_features(a, names):
    check(a.ndim == 3 and a.shape[0] == len(names), 'Input band count/order mismatch')
    b = dict(zip(names, a))
    required = {'VV_pre', 'VV_post', 'dVV', 'valid_pre', 'valid_post', 'elevation',
                'slope_deg', 'aspect_deg', 'geometry_visible_pre', 'geometry_visible_post'}
    check(required <= b.keys(), 'Missing named input bands')
    valid = ((b['valid_pre'] == 1) & (b['valid_post'] == 1) &
             np.isfinite(np.stack([b['VV_pre'], b['VV_post'], b['dVV']])).all(axis=0) &
             (np.stack([b['VV_pre'], b['VV_post'], b['dVV']]) != NODATA).all(axis=0))
    check(valid.any(), 'No valid paired VV pixels')
    check(np.allclose(b['dVV'][valid], (b['VV_post'] - b['VV_pre'])[valid], atol=2e-5, rtol=1e-5),
          'Raw dVV is not post minus pre; check units and bands')
    terrain = np.stack([b['elevation'], b['slope_deg'], b['aspect_deg']])
    terrain_valid = np.isfinite(terrain).all(axis=0) & (terrain != NODATA).all(axis=0)
    geometry = (b['geometry_visible_pre'] == 1) & (b['geometry_visible_post'] == 1)
    numerical = binary_erosion(valid, np.ones((7, 7)), border_value=0) & terrain_valid
    screen = numerical & geometry
    pre, post = filtered_db(b['VV_pre'], valid), filtered_db(b['VV_post'], valid)
    change = post - pre
    prestd, poststd = local_std(pre, valid), local_std(post, valid)
    features = np.stack([pre, post, change, np.abs(change), prestd, poststd,
                         poststd - prestd, b['elevation'], b['slope_deg'],
                         np.sin(np.radians(b['aspect_deg'])),
                         np.cos(np.radians(b['aspect_deg']))]).astype('float32')
    check(np.isfinite(features[:, numerical]).all(), 'Nonfinite feature values')
    features[:, ~numerical] = NODATA
    return features, {'data_validity_mask': valid, 'local_geometry_approx_mask': geometry,
                      'feature_numerical_validity_mask': numerical, 'screened_feature_mask': screen}

def write_raster(path, array, profile, descriptions, nodata):
    if array.ndim == 2:
        array = array[None]
    profile = dict(profile)
    profile.update(driver='GTiff', count=array.shape[0], dtype=str(array.dtype), nodata=nodata,
                   compress='deflate', predictor=3 if array.dtype.kind == 'f' else 2)
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(array)
        for band, name in enumerate(descriptions, 1):
            dst.set_band_description(band, name)
        dst.update_tags(training='false', road_passability='not_assessed')
    with rasterio.open(path) as src:
        check(np.array_equal(src.read(), array), f'Readback failed: {path.name}')

def prepare(args):
    config = load_json(args.config)
    check(config['nodata'] == NODATA, 'Unsupported input nodata value')
    selected = set(args.regions or [s['id'] for s in config['sites']])
    check(not args.regions or len(selected) == len(args.regions), 'Duplicate region IDs')
    sites = [s for s in config['sites'] if s['id'] in selected]
    check(len(sites) == len(selected), 'Unknown or duplicated region IDs')
    root = Path(args.output_dir).resolve()
    check(not root.exists(), 'Output directory already exists; choose a fresh path')
    # Check every source before creating output directories.
    for site in sites:
        path = Path(args.input_dir) / (site['id'] + '.tif')
        check(path.is_file(), f'Missing raw input: {path}')
    root.mkdir(parents=True)
    records = []
    for site in sites:
        path = Path(args.input_dir) / (site['id'] + '.tif')
        with rasterio.open(path) as src:
            check(str(src.crs) == config['crs'], f'CRS mismatch for {site["id"]}')
            check(tuple(src.shape) == tuple(site['shape']), 'Grid shape mismatch')
            check(src.transform == Affine(*site['grid']), 'Grid transform mismatch')
            check(src.count == len(config['band_order']), 'Band count mismatch')
            if any(src.descriptions):
                check(list(src.descriptions) == config['band_order'], 'Declared band names mismatch')
            a, profile = src.read(), src.profile.copy()
        features, masks = compute_features(a, config['band_order'])
        destination = root / site['id']
        destination.mkdir()
        outputs = {}
        write_raster(destination / 'VV_feature_candidates.tif', features, profile, FEATURE_NAMES, NODATA)
        outputs['VV_feature_candidates.tif'] = sha256(destination / 'VV_feature_candidates.tif')
        for name, values in masks.items():
            filename = name + '.tif'
            write_raster(destination / filename, values.astype('uint8'), profile, [name], None)
            outputs[filename] = sha256(destination / filename)
        record = {'id': site['id'], 'source_basename': path.name, 'source_sha256': sha256(path),
                  'outputs': outputs, 'screened_pixels': int(masks['screened_feature_mask'].sum())}
        records.append(record)
        print('Prepared', site['id'], flush=True)
    write_json(root / 'manifest.json', {'feature_names': FEATURE_NAMES, 'records': records,
               'config_sha256': sha256(args.config), 'training_performed': False,
               'geometry_certified': False, 'labels_created': False, 'road_passability': False})

def validate(args):
    root = Path(args.feature_dir)
    manifest = load_json(root / 'manifest.json')
    check(manifest['feature_names'] == FEATURE_NAMES, 'Feature schema mismatch')
    for record in manifest['records']:
        destination = root / record['id']
        for filename, expected in record['outputs'].items():
            check(sha256(destination / filename) == expected, f'Hash mismatch: {filename}')
        with rasterio.open(destination / 'VV_feature_candidates.tif') as src:
            f, grid = src.read(), (src.shape, src.crs, src.transform)
            check(list(src.descriptions) == FEATURE_NAMES, 'Feature names mismatch')
        masks = {}
        for name in ['data_validity_mask', 'local_geometry_approx_mask',
                     'feature_numerical_validity_mask', 'screened_feature_mask']:
            with rasterio.open(destination / (name + '.tif')) as src:
                check((src.shape, src.crs, src.transform) == grid, 'Mask grid mismatch')
                mask = src.read(1)
                check(np.isin(mask, [0, 1]).all(), 'Nonbinary mask')
                masks[name] = mask.astype(bool)
        valid = masks['feature_numerical_validity_mask']
        check((~valid | masks['data_validity_mask']).all(), 'Numerical mask includes missing data')
        check(np.isfinite(f[:, valid]).all(), 'Nonfinite valid features')
        check((f[:, ~valid] == NODATA).all(), 'Invalid feature nodata mismatch')
        check(np.array_equal(masks['screened_feature_mask'], valid & masks['local_geometry_approx_mask']),
              'Screened mask mismatch')
        check(np.allclose(f[2, valid], f[1, valid] - f[0, valid], atol=3e-6), 'dVV mismatch')
        check(np.allclose(f[3, valid], np.abs(f[2, valid])), 'Absolute change mismatch')
        check(np.allclose(f[6, valid], f[5, valid] - f[4, valid], atol=3e-6), 'Texture change mismatch')
        print('Validated', record['id'], flush=True)

def samples(args):
    check(args.allow_weak_references, 'Explicit --allow-weak-references is required; these are not approved truth labels')
    proposal = load_json(args.proposals)
    split = load_json(args.split)['assignments']
    check(proposal['status'] == 'WEAK_PROPOSALS_NOT_APPROVED', 'Unknown label status')
    check(proposal['feature_names'] == FEATURE_NAMES, 'Proposal feature schema mismatch')
    feature_root = Path(args.feature_dir)
    cache, data, roles, seen = {}, [], [], set()
    object_roles = {}
    for record in proposal['records']:
        region = record['region_id']
        check(region in split and record['role'] == split[region], 'Spatial split mismatch')
        if region not in cache:
            with rasterio.open(feature_root / region / 'VV_feature_candidates.tif') as src:
                check(list(src.descriptions) == FEATURE_NAMES, 'Feature bands mismatch')
                cache[region] = src.read()
                grid = (src.shape, src.crs, src.transform)
            with rasterio.open(feature_root / region / 'screened_feature_mask.tif') as src:
                check((src.shape, src.crs, src.transform) == grid, 'Sample mask grid mismatch')
                mask = src.read(1)
                check(np.isin(mask, [0, 1]).all(), 'Nonbinary sample mask')
                cache[region + '_screen'] = mask == 1
        row, col, label = record['row'], record['col'], record['reference_label_proposal']
        shape = cache[region].shape[1:]
        check(isinstance(row, int) and isinstance(col, int) and 0 <= row < shape[0] and 0 <= col < shape[1], 'Sample outside grid')
        check((region, row, col) not in seen, 'Duplicate sample')
        seen.add((region, row, col))
        check(label in [0, 1] and cache[region + '_screen'][row, col], 'Invalid label or excluded sample')
        check(bool(record['object_ids']) if label == 1 else bool(record['negative_review_group_ids']), 'Missing reference provenance')
        for identifier in record['object_ids'] + record['negative_review_group_ids']:
            check(identifier not in object_roles or object_roles[identifier] == record['role'], 'Object/group crosses spatial roles')
            object_roles[identifier] = record['role']
        values = cache[region][:, row, col]
        check(np.isfinite(values).all() and (values != NODATA).all(), 'Invalid sample feature values')
        data.append(values)
        roles.append(record['role'])
    check(bool(data), 'No samples supplied')
    out = Path(args.output_dir).resolve()
    check(not out.exists(), 'Choose a fresh output directory')
    out.mkdir(parents=True)
    np.savez_compressed(out / 'weak_reference_samples.npz', X=np.asarray(data, dtype='float32'),
                        y=np.asarray([r['reference_label_proposal'] for r in proposal['records']], dtype='uint8'),
                        role=np.asarray(roles), region=np.asarray([r['region_id'] for r in proposal['records']]),
                        row=np.asarray([r['row'] for r in proposal['records']]),
                        col=np.asarray([r['col'] for r in proposal['records']]), feature_names=np.asarray(FEATURE_NAMES))
    write_json(out / 'manifest.json', {'samples': len(data), 'training_performed': False, 'training_ready': False,
               'status': 'EXPLICIT_WEAK_REFERENCE_EXPLORATION_ONLY', 'proposals_sha256': sha256(args.proposals),
               'warnings': proposal['warnings'], 'spatial_split_sha256': sha256(args.split)})
    print('Extracted', len(data), 'weak-reference proposals; no model was trained')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--input-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--config', default='configs/vv_design.json')
    p.add_argument('--regions', nargs='+')
    p.set_defaults(run=prepare)
    p = sub.add_parser('validate')
    p.add_argument('--feature-dir', required=True)
    p.set_defaults(run=validate)
    p = sub.add_parser('samples')
    p.add_argument('--feature-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--proposals', default='configs/reviewed_sample_proposals.json')
    p.add_argument('--split', default='configs/spatial_split.json')
    p.add_argument('--allow-weak-references', action='store_true')
    p.set_defaults(run=samples)
    args = parser.parse_args()
    try:
        args.run(args)
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.exit(2, f'Input error: {error}\n')

if __name__ == '__main__':
    main()
