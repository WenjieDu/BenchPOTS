"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
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
    @staticmethod
    def _build_mock_physionet2012_data(n_records=20):
        record_ids = np.arange(1000, 1000 + n_records)
        rows = []
        for record_id in record_ids:
            for time in range(48):
                rows.append(
                    {
                        "RecordID": record_id,
                        "Time": time,
                        "ICUType": (record_id % 4) + 1,
                        "Age": 40 + (record_id % 10),
                        "HR": float(record_id + time),
                    }
                )

        set_a = pd.DataFrame(rows)
        outcomes_a = pd.DataFrame(
            {"In-hospital_death": [1] * (n_records // 2) + [0] * (n_records - (n_records // 2))},
            index=record_ids,
        )
        return {
            "set-a": set_a,
            "outcomes-a": outcomes_a,
            "static_features": ["Age", "ICUType"],
        }

    @staticmethod
    def _build_mock_physionet2019_data(n_records=20):
        record_ids = np.arange(2000, 2000 + n_records)
        rows = []
        for record_id in record_ids:
            for iculos in range(1, 49):
                rows.append(
                    {
                        "RecordID": record_id,
                        "ICULOS": iculos,
                        "SepsisLabel": int(record_id % 2),
                        "Age": 50 + (record_id % 10),
                        "HR": float(record_id + iculos),
                    }
                )

        training_set_a = pd.DataFrame(rows)
        return {
            "training_setA": training_set_a,
            "training_setB": training_set_a.copy(),
            "static_features": ["Age"],
        }

    @staticmethod
    def _build_mock_ucr_uea_data(n_train=20, n_test=8, n_steps=6, n_features=2):
        X_train = np.repeat(np.arange(n_train, dtype=float).reshape(-1, 1, 1), n_steps * n_features, axis=2).reshape(
            n_train, n_steps, n_features
        )
        X_test = np.repeat(np.arange(n_test, dtype=float).reshape(-1, 1, 1), n_steps * n_features, axis=2).reshape(
            n_test, n_steps, n_features
        )
        y_train = np.arange(n_train)
        y_test = np.arange(n_test)
        return {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}

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

    def test_physionet2012_split_random_state(self):
        def extract_split_record_ids(dataset, split):
            split_X = dataset[f"{split}_X"]
            scaler = dataset["scaler"]
            unscaled = scaler.inverse_transform(split_X.reshape(-1, split_X.shape[-1])).reshape(split_X.shape)
            return np.unique(np.rint(unscaled[:, 0, 0]).astype(int))

        with patch(
            "benchpots.datasets.physionet_2012.tsdb.load",
            side_effect=lambda _: self._build_mock_physionet2012_data(),
        ):
            ds_a = preprocess_physionet2012(subset="set-a", rate=0, random_state=42)
            ds_b = preprocess_physionet2012(subset="set-a", rate=0, random_state=42)
            ds_c = preprocess_physionet2012(subset="set-a", rate=0, random_state=7)

        train_record_ids_a = extract_split_record_ids(ds_a, "train")
        val_record_ids_a = extract_split_record_ids(ds_a, "val")
        test_record_ids_a = extract_split_record_ids(ds_a, "test")

        np.testing.assert_array_equal(train_record_ids_a, extract_split_record_ids(ds_b, "train"))
        np.testing.assert_array_equal(val_record_ids_a, extract_split_record_ids(ds_b, "val"))
        np.testing.assert_array_equal(test_record_ids_a, extract_split_record_ids(ds_b, "test"))

        split_changed = (
            not np.array_equal(train_record_ids_a, extract_split_record_ids(ds_c, "train"))
            or not np.array_equal(val_record_ids_a, extract_split_record_ids(ds_c, "val"))
            or not np.array_equal(test_record_ids_a, extract_split_record_ids(ds_c, "test"))
        )
        assert split_changed, "Different random_state values should produce different splits."

    def test_physionet2019_split_random_state(self):
        def extract_split_record_ids(dataset, split):
            split_X = dataset[f"{split}_X"]
            scaler = dataset["scaler"]
            unscaled = scaler.inverse_transform(split_X.reshape(-1, split_X.shape[-1])).reshape(split_X.shape)
            return np.unique(np.rint(unscaled[:, 0, 0] - 1).astype(int))

        with patch(
            "benchpots.datasets.physionet_2019.tsdb.load",
            side_effect=lambda _: self._build_mock_physionet2019_data(),
        ):
            ds_a = preprocess_physionet2019(subset="training_setA", rate=0, random_state=42)
            ds_b = preprocess_physionet2019(subset="training_setA", rate=0, random_state=42)
            ds_c = preprocess_physionet2019(subset="training_setA", rate=0, random_state=7)

        np.testing.assert_array_equal(extract_split_record_ids(ds_a, "train"), extract_split_record_ids(ds_b, "train"))
        np.testing.assert_array_equal(extract_split_record_ids(ds_a, "val"), extract_split_record_ids(ds_b, "val"))
        np.testing.assert_array_equal(extract_split_record_ids(ds_a, "test"), extract_split_record_ids(ds_b, "test"))

        split_changed = (
            not np.array_equal(extract_split_record_ids(ds_a, "train"), extract_split_record_ids(ds_c, "train"))
            or not np.array_equal(extract_split_record_ids(ds_a, "val"), extract_split_record_ids(ds_c, "val"))
            or not np.array_equal(extract_split_record_ids(ds_a, "test"), extract_split_record_ids(ds_c, "test"))
        )
        assert split_changed, "Different random_state values should produce different splits."

    def test_ucr_uea_split_random_state(self):
        with patch("benchpots.datasets.ucr_uea_datasets.tsdb.list", return_value=["ucr_uea_mock"]), patch(
            "benchpots.datasets.ucr_uea_datasets.tsdb.load",
            side_effect=lambda _: self._build_mock_ucr_uea_data(),
        ):
            ds_a = preprocess_ucr_uea_datasets(dataset_name="ucr_uea_mock", rate=0, random_state=42)
            ds_b = preprocess_ucr_uea_datasets(dataset_name="ucr_uea_mock", rate=0, random_state=42)
            ds_c = preprocess_ucr_uea_datasets(dataset_name="ucr_uea_mock", rate=0, random_state=7)

        np.testing.assert_array_equal(np.sort(ds_a["train_y"]), np.sort(ds_b["train_y"]))
        np.testing.assert_array_equal(np.sort(ds_a["val_y"]), np.sort(ds_b["val_y"]))
        split_changed = not np.array_equal(np.sort(ds_a["train_y"]), np.sort(ds_c["train_y"]))
        assert split_changed, "Different random_state values should produce different splits."

    def test_random_walk_split_random_state(self):
        ds_a = preprocess_random_walk(
            n_steps=8,
            n_features=3,
            n_classes=3,
            n_samples_each_class=40,
            missing_rate=0,
            random_state=42,
        )
        ds_b = preprocess_random_walk(
            n_steps=8,
            n_features=3,
            n_classes=3,
            n_samples_each_class=40,
            missing_rate=0,
            random_state=42,
        )
        ds_c = preprocess_random_walk(
            n_steps=8,
            n_features=3,
            n_classes=3,
            n_samples_each_class=40,
            missing_rate=0,
            random_state=7,
        )

        np.testing.assert_allclose(ds_a["train_X"], ds_b["train_X"])
        np.testing.assert_array_equal(ds_a["train_y"], ds_b["train_y"])
        split_changed = not np.allclose(ds_a["train_X"], ds_c["train_X"])
        assert split_changed, "Different random_state values should produce different splits."

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
