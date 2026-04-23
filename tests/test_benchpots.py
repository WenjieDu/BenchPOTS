"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import unittest

import numpy as np
import torch
from pygrinder import calc_missing_rate

from benchpots.datasets import (
    preprocess_random_walk,
    preprocess_physionet2012,
    preprocess_physionet2019,
    preprocess_ett,
    preprocess_electricity_load_diagrams,
    preprocess_beijing_air_quality,
    preprocess_italy_air_quality,
    preprocess_ucr_uea_datasets,
    preprocess_nl_benchmarks,
)
from benchpots.utils import sliding_window, inverse_sliding_window
from benchpots.utils import convert_processed_dataset_by_task_type


class TestBenchPOTS(unittest.TestCase):
    def test_random_walk(self):
        n_steps = 8
        n_features = 5
        n_classes = 2
        n_samples_each_class = 100
        missing_rate = 0.1
        anomaly_rate = 0.1
        dataset_wo_anomalies = preprocess_random_walk(
            n_steps=n_steps,
            n_features=n_features,
            n_classes=n_classes,
            n_samples_each_class=n_samples_each_class,
            missing_rate=missing_rate,
        )
        assert "train_anomaly_y" not in dataset_wo_anomalies.keys()
        dataset_w_anomalies = preprocess_random_walk(
            n_steps=n_steps,
            n_features=n_features,
            n_classes=n_classes,
            n_samples_each_class=n_samples_each_class,
            anomaly_rate=anomaly_rate,
            missing_rate=missing_rate,
        )
        assert "train_anomaly_y" in dataset_w_anomalies.keys()
        train_X = dataset_w_anomalies["train_X"]
        val_X = dataset_w_anomalies["val_X"]
        test_X = dataset_w_anomalies["test_X"]
        trainset_size, trainset_n_steps, trainset_n_features = train_X.shape
        valset_size, valset_n_steps, valset_n_features = val_X.shape
        testset_size, testset_n_steps, testset_n_features = test_X.shape

        assert trainset_size + valset_size + testset_size == n_classes * n_samples_each_class
        assert trainset_n_steps == valset_n_steps == testset_n_steps == n_steps
        assert trainset_n_features == valset_n_features == testset_n_features == n_features

        assert round(calc_missing_rate(train_X), 1) == missing_rate * 2  # 'cause train set was masked twice
        assert round(calc_missing_rate(val_X), 1) == missing_rate
        assert round(calc_missing_rate(test_X), 1) == missing_rate

    def test_physionet2012(self):
        preprocess_physionet2012(subset="set-a", rate=0.1)

    def test_random_walk_forecasting_shapes(self):
        n_steps = 12
        n_pred_steps = 4
        n_features = 3
        dataset = preprocess_random_walk(
            n_steps=n_steps,
            n_features=n_features,
            n_classes=2,
            n_samples_each_class=60,
            missing_rate=0,
            task_type="forecasting",
            n_pred_steps=n_pred_steps,
        )

        assert dataset["task_type"] == "forecasting"
        assert dataset["n_steps_original"] == n_steps
        assert dataset["n_steps"] == n_steps - n_pred_steps
        assert dataset["n_pred_steps"] == n_pred_steps
        assert dataset["n_pred_features"] == n_features

        assert dataset["train_X"].shape[1] == n_steps - n_pred_steps
        assert dataset["val_X"].shape[1] == n_steps - n_pred_steps
        assert dataset["test_X"].shape[1] == n_steps - n_pred_steps

        assert dataset["train_X_pred"].shape[1] == n_pred_steps
        assert dataset["val_X_pred"].shape[1] == n_pred_steps
        assert dataset["test_X_pred"].shape[1] == n_pred_steps

    def test_random_walk_forecasting_uses_ori_targets(self):
        n_pred_steps = 3
        base_X = np.arange(2 * 8 * 2, dtype="float32").reshape(2, 8, 2)
        processed = {
            "train_X": base_X.copy(),
            "val_X": base_X.copy() + 10,
            "test_X": base_X.copy() + 20,
            "train_X_ori": base_X.copy() + 100,
            "val_X_ori": base_X.copy() + 110,
            "test_X_ori": base_X.copy() + 120,
        }

        converted = convert_processed_dataset_by_task_type(
            processed,
            task_type="forecasting",
            n_pred_steps=n_pred_steps,
        )

        assert "train_X_pred_ori" not in converted
        assert "val_X_pred_ori" not in converted
        assert "test_X_pred_ori" not in converted

        np.testing.assert_allclose(converted["train_X_pred"], (base_X + 100)[:, -n_pred_steps:, :], equal_nan=True)
        np.testing.assert_allclose(converted["val_X_pred"], (base_X + 110)[:, -n_pred_steps:, :], equal_nan=True)
        np.testing.assert_allclose(converted["test_X_pred"], (base_X + 120)[:, -n_pred_steps:, :], equal_nan=True)

    def test_task_type_conversion_validation(self):
        fake_dataset = {
            "train_X": np.random.randn(4, 8, 2).astype("float32"),
            "val_X": np.random.randn(4, 8, 2).astype("float32"),
            "test_X": np.random.randn(4, 8, 2).astype("float32"),
        }

        with self.assertRaises(ValueError):
            convert_processed_dataset_by_task_type(dict(fake_dataset), task_type="unknown")

        with self.assertRaises(ValueError):
            convert_processed_dataset_by_task_type(
                dict(fake_dataset),
                task_type="forecasting",
                n_pred_steps=8,
            )

        with self.assertRaises(ValueError):
            convert_processed_dataset_by_task_type(
                dict(fake_dataset),
                task_type="forecasting",
                n_pred_steps=2,
                forecast_feature_indices=[10],
            )

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

    def test_nl_benchs(self):
        preprocess_nl_benchmarks(dataset_name="EMPS", rate=0.1, n_steps=24)
        # bypass the below datasets to cut down the testing time
        # preprocess_nl_benchmarks(dataset_name="CED", rate=0.1, n_steps=24)
        # preprocess_nl_benchmarks(dataset_name="WienerHammerBenchMark", rate=0.1, n_steps=24)
        # preprocess_nl_benchmarks(dataset_name="Silverbox", rate=0.1, n_steps=24)
        # preprocess_nl_benchmarks(dataset_name="F16", rate=0.1, n_steps=24)
        # preprocess_nl_benchmarks(dataset_name="ParWH", rate=0.1, n_steps=24)
        # preprocess_nl_benchmarks(dataset_name="Cascaded_Tanks", rate=0.1, n_steps=24)
        # preprocess_nl_benchmarks(dataset_name="BoucWen", rate=0.1, n_steps=24)
        # preprocess_nl_benchmarks(dataset_name="WienerHammerstein_Process_Noise", rate=0.1, n_steps=24)
        # preprocess_nl_benchmarks(dataset_name="Industrial_robot", rate=0.1, n_steps=24)

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
