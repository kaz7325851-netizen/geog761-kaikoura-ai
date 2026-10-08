"""Run the reviewed author pipeline in a fresh directory, without main.py cleanup."""
from pathlib import Path
import sys
import json
import hashlib
import time
import traceback
import yaml
import numpy as np
import pandas as pd
import sklearn
from osgeo import gdal

ROOT = Path(__file__).resolve().parent
OFFICIAL = ROOT / 'official'
sys.path.insert(0, str(OFFICIAL))
from check_saved_model import load_reviewed, MODEL

def folder(path):
    path.mkdir(parents=True, exist_ok=True)
    return str(path) + '/'

def main():
    out = ROOT / 'results' / time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    print('OUTPUT_DIRECTORY=' + str(out), flush=True)
    metadata = {'status': 'running', 'repo_revision': '53b1daf5821350c4af7eb4ea35d9569f4ae4741d',
                'model_sha256': hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                'python': sys.version, 'sklearn': sklearn.__version__, 'numpy': np.__version__,
                'gdal': gdal.VersionInfo(), 'retrained': False, 'stage_seconds': {},
                'limitations': ['Author imagery sample has no reference labels: no accuracy/F1/recall calculated.',
                                'Not a road passability or beach landing model.',
                                'Excluded pixels remain unknown, not negative/safe.',
                                'Sample may overlap model training; independence not verified.']}
    try:
        with (OFFICIAL / 'config.yaml').open() as handle:
            config = yaml.safe_load(handle)
        config.update(dev_mode=False, save_dataset=False, plot_kea=False, coordinates=False)
        seg = config['k_means_segmentation']
        source = OFFICIAL / 'datasets/input/satellite_imagery'
        seg['input'] = {'image_difference': str(source / 'image_difference') + '/',
                        'post_event_image': str(source / 'post_event_image') + '/',
                        'training_labels': str(out / 'unused_labels') + '/'}
        seg['output'] = {k: folder(out / k) for k in ['tables', 'segments']}
        seg['temps'] = {k: folder(out / 'temps' / k) for k in ['folder', 'shapes', 'shepherd']}
        merge = config['mergin_algorithm']
        merge['input']['tables'] = seg['output']['tables']
        merge['output'] = {k: folder(out / k) for k in ['wrong', 'merged', 'regions']}
        config['model'] = {'path': str(MODEL), 'segments': merge['output']['merged'],
                           'predictions': folder(out / 'predictions'), 'regions': merge['output']['regions']}
        (out / 'resolved_config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
        from segmentation.k_means_segmentation import initial_segmentation_thesis_mh as segmentation
        from segmentation.merging_algorithm import merger
        from model import model as classifier
        for name, function in [('segmentation_and_features', segmentation.run), ('region_merging', merger.run)]:
            print('START_STAGE=' + name, flush=True)
            start = time.perf_counter()
            function(config)
            metadata['stage_seconds'][name] = time.perf_counter() - start
        clf = load_reviewed()
        classifier.load = lambda filename: clf
        predictions = []
        original_predict = classifier.make_predictions
        def capture(frame, model, cfg):
            features = frame.iloc[:, 1:].astype(float)
            assert features.shape[1] == clf.n_features_
            assert np.isfinite(features.values).all()
            saved = frame.copy()
            saved['prediction'] = model.predict(features.values)
            saved['positive_probability'] = model.predict_proba(features.values)[:, 1]
            predictions.append(saved)
            return original_predict(frame, model, cfg)
        classifier.make_predictions = capture
        start = time.perf_counter()
        classifier.run(config)
        metadata['stage_seconds']['feature_context_and_RF'] = time.perf_counter() - start
        if not predictions:
            raise RuntimeError('No regions reached classifier after author filtering.')
        saved = pd.concat(predictions, ignore_index=True)
        saved.to_csv(str(out / 'classifier_features_predictions.csv'), index=False)
        metadata['feature_order'] = list(saved.columns[1:-2])
        metadata['region_predictions'] = {str(k): int(v) for k, v in saved['prediction'].value_counts().items()}
        clumps_paths = list((out / 'segments').glob('*.kea'))
        assert len(clumps_paths) == 1
        ds = gdal.Open(str(clumps_paths[0]))
        clumps = ds.GetRasterBand(1).ReadAsArray()
        lut = np.full(int(clumps.max()) + 1, 255, dtype=np.uint8)
        mappings = {}
        for file in (out / 'regions').glob('*.json'):
            mappings.update(json.loads(file.read_text()))
        for _, row in saved.iterrows():
            for segment in mappings[row['segment_id']]:
                lut[int(segment.rsplit('_S', 1)[1])] = int(row['prediction'])
        mask = lut[clumps]
        raster = gdal.GetDriverByName('GTiff').Create(str(out / 'prediction.tif'), ds.RasterXSize,
                  ds.RasterYSize, 1, gdal.GDT_Byte, options=['COMPRESS=DEFLATE'])
        raster.SetGeoTransform(ds.GetGeoTransform())
        raster.SetProjection(ds.GetProjection())
        band = raster.GetRasterBand(1)
        band.SetNoDataValue(255)
        band.SetDescription('0=RF negative; 1=RF landslide; 255=excluded/unclassified')
        band.WriteArray(mask)
        raster.FlushCache()
        raster = None
        values, counts = np.unique(mask, return_counts=True)
        metadata['pixel_counts'] = {str(int(v)): int(c) for v, c in zip(values, counts)}
        metadata['raster_shape'] = list(mask.shape)
        metadata['geotransform'] = ds.GetGeoTransform()
        metadata['projection'] = ds.GetProjection()
        metadata['segment_table_rows'] = sum(len(pd.read_csv(str(p))) for p in (out / 'tables').glob('*.csv'))
        metadata['merged_regions_before_outlier_filter'] = sum(len(pd.read_csv(str(p))) for p in (out / 'merged').glob('*.csv'))
        metadata['input_hashes'] = {str(p.relative_to(OFFICIAL)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in source.rglob('*.tif')}
        metadata['status'] = 'actual_author_sample_inference_complete'
        print(json.dumps(metadata, indent=2), flush=True)
    except Exception as exc:
        metadata.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        (out / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
