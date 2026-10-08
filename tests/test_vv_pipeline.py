import importlib.util
from pathlib import Path
import unittest
import argparse
import json
import tempfile
import numpy as np

path = Path(__file__).resolve().parents[1] / 'scripts/vv_pipeline.py'
spec = importlib.util.spec_from_file_location('vv_pipeline', path)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)
NAMES = ['VV_pre', 'VV_post', 'dVV', 'VV_pre_median', 'dVV_median', 'angle_pre',
         'angle_post', 'elevation', 'slope_deg', 'aspect_deg', 'alpha_r_pre',
         'alpha_r_post', 'valid_pre', 'valid_post', 'valid_median',
         'geometry_visible_pre', 'geometry_visible_post']

class Features(unittest.TestCase):
    def data(self):
        a = np.zeros((17, 24, 24), dtype='float32')
        a[0], a[1], a[2] = -10, -8, 2
        a[7], a[8], a[9] = 100, 30, 90
        a[12:] = 1
        return a

    def test_constant_scene(self):
        f, masks = pipeline.compute_features(self.data(), NAMES)
        m = masks['feature_numerical_validity_mask']
        np.testing.assert_allclose(f[0, m], -10)
        np.testing.assert_allclose(f[1, m], -8)
        np.testing.assert_allclose(f[2, m], 2)
        np.testing.assert_allclose(f[4:7, m], 0, atol=1e-6)
        self.assertEqual(m.sum(), 18 * 18)
        self.assertTrue((f[:, ~m] == -9999).all())

    def test_missing_and_geometry(self):
        a = self.data()
        a[12, 12, 12] = 0
        a[15, 6, 6] = 0
        _, m = pipeline.compute_features(a, NAMES)
        self.assertFalse(m['feature_numerical_validity_mask'][10:15, 10:15].any())
        self.assertTrue(m['feature_numerical_validity_mask'][6, 6])
        self.assertFalse(m['screened_feature_mask'][6, 6])

    def test_bad_change_rejected(self):
        a = self.data()
        a[2] = -2
        with self.assertRaises(ValueError):
            pipeline.compute_features(a, NAMES)

    def test_missing_band_rejected(self):
        with self.assertRaises(ValueError):
            pipeline.compute_features(self.data()[:-1], NAMES[:-1])

    def test_no_paired_data_rejected(self):
        a = self.data()
        a[12] = 0
        with self.assertRaises(ValueError):
            pipeline.compute_features(a, NAMES)

    def test_weak_reference_consent_required(self):
        with self.assertRaisesRegex(ValueError, 'allow-weak-references'):
            pipeline.samples(argparse.Namespace(allow_weak_references=False))

    def test_unknown_region_no_output(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            config = Path(temp) / 'config.json'
            config.write_text(json.dumps({'sites': [{'id': 'A'}], 'nodata': -9999}), encoding='utf-8')
            output = Path(temp) / 'output'
            with self.assertRaisesRegex(ValueError, 'Unknown'):
                pipeline.prepare(argparse.Namespace(config=config, regions=['B'],
                                 input_dir=temp, output_dir=output))
            self.assertFalse(output.exists())

    def test_missing_input_no_output(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            config = Path(temp) / 'config.json'
            config.write_text(json.dumps({'sites': [{'id': 'A'}], 'nodata': -9999}), encoding='utf-8')
            output = Path(temp) / 'output'
            with self.assertRaisesRegex(ValueError, 'Missing raw input'):
                pipeline.prepare(argparse.Namespace(config=config, regions=None,
                                 input_dir=temp, output_dir=output))
            self.assertFalse(output.exists())

if __name__ == '__main__':
    unittest.main()
