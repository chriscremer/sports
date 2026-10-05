import unittest

import cv2
import numpy as np

from compare_ball_models import windows
from comparison_tracknet_v3 import heatmap_candidates
from comparison_wasb import heatmap_candidates as wasb_candidates


class FakeCapture:
    def __init__(self):
        self.position = 0

    def set(self, prop, value):
        self.position = int(value)

    def read(self):
        frame = np.full((8,16,3), self.position, np.uint8)
        self.position += 1
        return True, frame

    def get(self, prop):
        return (self.position-1)*1000/30


class ComparisonTests(unittest.TestCase):
    def test_temporal_padding_does_not_duplicate_output_times(self):
        groups = list(windows(FakeCapture(),10,5,3))
        self.assertEqual([g[2] for g in groups], [3,2])
        self.assertEqual([[int(f[0,0,0]) for f in g[0]] for g in groups], [[10,11,12],[13,14,14]])
        times = [t for _, timestamps, _ in groups for t in timestamps]
        np.testing.assert_allclose(times, np.arange(10,15)/30)

    def test_scaled_candidates_preserve_separate_objects(self):
        heatmap = np.zeros((8,16), np.float32)
        heatmap[2,3] = .8
        heatmap[6,12] = .9
        candidates = heatmap_candidates(heatmap, (1920,1080))
        self.assertEqual(len(candidates),2)
        np.testing.assert_allclose(candidates[0]['pixel'], [1440,810], atol=.001)
        np.testing.assert_allclose(candidates[1]['pixel'], [360,270], atol=.001)

    def test_native_candidates_do_not_rescale_or_fill_missing_ball(self):
        heatmap = np.zeros((108,192), np.float32)
        self.assertEqual(wasb_candidates(heatmap), [])
        heatmap[27,36] = .8
        np.testing.assert_allclose(wasb_candidates(heatmap)[0]['pixel'], [36,27], atol=.001)


if __name__ == '__main__':
    unittest.main()
