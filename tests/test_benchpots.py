"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import unittest

from benchpots.datasets import (
    preprocess_physionet2012,
    preprocess_physionet2019,
    preprocess_ett,
    preprocess_electricity_load_diagrams,
    preprocess_beijing_air_quality,
    preprocess_italy_air_quality,
    preprocess_ucr_uea_datasets,
)


class TestBenchPOTS(unittest.TestCase):
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
