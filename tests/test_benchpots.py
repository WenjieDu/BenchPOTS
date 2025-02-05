"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import unittest

import torch

from benchpots.datasets import (
    preprocess_random_walk,
    preprocess_physionet2012,
    preprocess_physionet2019,
    preprocess_ett,
    preprocess_electricity_load_diagrams,
    preprocess_beijing_air_quality,
    preprocess_italy_air_quality,
    preprocess_ucr_uea_datasets,
)
from benchpots.utils import sliding_window, inverse_sliding_window


class TestBenchPOTS(unittest.TestCase):
    def test_random_walk(self):
        preprocess_random_walk(
            n_steps=8,
            n_features=5,
            n_classes=2,
            n_samples_each_class=100,
            missing_rate=0.1,
        )

    def test_physionet2012(self):
        preprocess_physionet2012(subset="set-a", rate=0.1)

    def test_physionet2019(self):
        preprocess_physionet2019(subset="training_setA", rate=0.1)

    def test_ett(self):
        preprocess_ett(subset="ETTh1", rate=0.1, n_steps=24, pattern="point")

    def test_electricity(self):
        preprocess_electricity_load_diagrams(rate=0.1, n_steps=24, pattern="point")

    def test_beijing_air(self):
        preprocess_beijing_air_quality(rate=0.1, n_steps=24, pattern="point")

    def test_italy_air(self):
        preprocess_italy_air_quality(rate=0.1, n_steps=24, pattern="point")

    def test_ucr_uea(self):
        preprocess_ucr_uea_datasets(
            dataset_name="ucr_uea_MelbournePedestrian",
            rate=0.1,
            n_steps=24,
            pattern="point",
        )

    def test_sliding(self):
        torch_tensor = torch.randn(1024, 5)
        samples = sliding_window(torch_tensor, 8)
        assert len(samples.shape) == 3
        inverse_result = inverse_sliding_window(samples, 8)
        assert len(inverse_result.shape) == 2 and inverse_result.shape[0] == 1024

        numpy_arr = torch_tensor.numpy()
        samples = sliding_window(numpy_arr, 8)
        assert len(samples.shape) == 3
        inverse_result = inverse_sliding_window(samples, 8)
        assert len(inverse_result.shape) == 2 and inverse_result.shape[0] == 1024
