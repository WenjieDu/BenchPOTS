# BenchPOTS <a href="https://github.com/WenjieDu/BenchPOTS"><img src="https://pypots.com/figs/pypots_logos/BenchPOTS/logo_FFBG.svg" width="200" align="right" alt="BenchPOTS logo"></a>

> A Python Toolbox for Benchmarking Machine Learning on Partially-Observed Time Series

<p>
    <a href="https://docs.pypots.com/en/latest/benchpots.html">
        <img alt="Python version" src="https://img.shields.io/badge/Python-v3.10+-F8C6B5?logo=python&logoColor=white">
    </a>
    <a href="https://github.com/WenjieDu/BenchPOTS/releases">
        <img alt="the latest release version" src="https://img.shields.io/github/v/release/wenjiedu/benchpots?color=EE781F&include_prereleases&label=Release&logo=github&logoColor=white">
    </a>
    <a href="https://docs.pypots.com/en/latest/benchpots.html">
        <img alt="Docs building" src="https://img.shields.io/readthedocs/pypots?logo=readthedocs&label=Docs&logoColor=white&color=395260">
    </a>
    <a href="https://pepy.tech/project/benchpots">
        <img alt="PyPI downloads" src="https://pypots.com/figs/downloads_badges/pypi_benchpots_downloads.svg">
    </a>
    <a href="https://anaconda.org/conda-forge/benchpots">
        <img alt="Conda downloads" src="https://pypots.com/figs/downloads_badges/conda_benchpots_downloads.svg">
    </a>
    <a href="https://github.com/WenjieDu/BenchPOTS/blob/main/LICENSE">
        <img alt="BSD-3 license" src="https://img.shields.io/badge/License-BSD--3-E9BB41?logo=opensourceinitiative&logoColor=white">
    </a>
    <a href="https://github.com/WenjieDu/PyPOTS#-community">
        <img alt="Community" src="https://img.shields.io/badge/join_us-community!-C8A062">
    </a>
    <a href="https://star-history.com/#wenjiedu/benchpots">
        <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/wenjiedu/benchpots?logo=None&color=6BB392&label=%E2%98%85%20Stars">
    </a>
    <a href="https://github.com/WenjieDu/BenchPOTS/network/members">
        <img alt="GitHub Repo forks" src="https://img.shields.io/github/forks/wenjiedu/benchpots?logo=forgejo&logoColor=black&label=Forks">
    </a>
    <a href="https://github.com/WenjieDu/BenchPOTS/graphs/contributors">
        <img alt="GitHub contributors" src="https://img.shields.io/github/contributors/wenjiedu/benchpots?color=D8E699&label=Contributors&logo=GitHub">
    </a>
    <a href="https://github.com/WenjieDu/BenchPOTS/actions/workflows/testing_ci.yml">
        <img alt="GitHub Testing" src="https://img.shields.io/github/actions/workflow/status/wenjiedu/benchpots/testing_ci.yml?logo=circleci&color=C8D8E1&label=CI">
    </a>
    <a href="https://coveralls.io/github/WenjieDu/BenchPOTS?branch=main">
        <img alt="Coveralls coverage" src="https://img.shields.io/coverallsCoverage/github/WenjieDu/BenchPOTS?branch=main&logo=coveralls&color=75C1C4&label=Coverage">
    </a>
    <a href="https://sonarcloud.io/component_measures?id=WenjieDu_BenchPOTS&metric=sqale_rating&view=list">
        <img alt="maintainability" src="https://sonarcloud.io/api/project_badges/measure?project=WenjieDu_BenchPOTS&metric=sqale_rating">
    </a>
    <a href="https://github.com/astral-sh/ruff">
        <img alt="Code Style" src="https://img.shields.io/badge/Code_Style-Ruff-D7FF64?logo=ruff">
    </a>
    <a href="https://arxiv.org/abs/2305.18811">
        <img alt="arXiv DOI" src="https://img.shields.io/badge/DOI-10.48550/arXiv.2305.18811-F8F7F0">
    </a>
</p>

BenchPOTS is an open-source Python toolbox for **standardized preprocessing and benchmarking of partially observed time series**.
It is the benchmarking layer of the [PyPOTS ecosystem](https://pypots.com/ecosystem/), providing reusable preprocessing
pipelines for **170+ public time-series datasets** and synthetic benchmarks.

BenchPOTS turns raw datasets into consistent train/validation/test structures for machine-learning experiments.
Its task-aware preprocessing supports imputation, forecasting, classification, clustering, and anomaly detection,
with configurable artificial missingness for controlled evaluation.

[Website](https://pypots.com) · [Documentation](https://docs.pypots.com/en/latest/benchpots.html) · [Tutorials](https://github.com/WenjieDu/BrewPOTS) · [Community](https://github.com/WenjieDu/PyPOTS#-community)

[**Key Features**](#-key-features) · [**Installation**](#-installation) · [**Usage**](#-usage) · [**Dataset Preprocessors**](#-dataset-preprocessors) · [**PyPOTS Ecosystem**](#-pypots-ecosystem) · [**Citing BenchPOTS/PyPOTS**](#-citing-benchpotspypots) · [**Contribution**](#-contribution)

## ❖ Key Features

- **Standardized preprocessing:** Build consistent train/validation/test splits, scaling, sliding windows, labels, and metadata through dataset-specific pipelines.
- **170+ public datasets:** Access a broad time-series collection through TSDB-backed preprocessing, including the UCR/UEA archives and real-world healthcare, energy, air-quality, traffic, and solar datasets.
- **Multi-task formatting:** Prepare data for imputation, forecasting, classification, clustering, and anomaly detection through a common `task_type` interface.
- **Forecasting-aware targets:** Generate `*_X_pred` targets before artificial missingness is applied, keeping forecasting ground truth separate from masked model inputs.
- **Controlled missingness:** Inject point, subsequence, or block missingness with configurable rates for reproducible benchmarking.
- **Synthetic benchmarks:** Generate random-walk datasets and preprocess nonlinear system-identification benchmarks for controlled experiments.
- **Reproducible splits:** Dataset pipelines expose explicit random-state controls where stochastic splitting or generation is involved.

## ❖ Installation

BenchPOTS requires Python 3.10 or later and is available on [PyPI](https://pypi.org/project/benchpots/)
and [Anaconda](https://anaconda.org/conda-forge/benchpots).

Choose **one** of the following installation methods.

**Option 1: pip (recommended)**

```bash
pip install benchpots
# To upgrade: pip install benchpots --upgrade
```

**Option 2: Conda**

```bash
conda install conda-forge::benchpots
# To upgrade: conda update conda-forge::benchpots
```

**Option 3: Latest source code on main (may include unreleased changes)**

```bash
pip install https://github.com/WenjieDu/BenchPOTS/archive/main.zip
```

## ❖ Usage

### Quick Start

This self-contained random-walk example requires no dataset download.

```python
from benchpots.datasets import preprocess_random_walk

data = preprocess_random_walk(
    n_steps=24,
    n_features=8,
    n_classes=3,
    n_samples_each_class=100,
    missing_rate=0.1,
    random_state=42,
)

print(data["train_X"].shape)
print(data["val_X"].shape)
print(data["test_X"].shape)
print(data["task_type"])  # imputation
```

### Forecasting Preprocessing

Set `task_type="forecasting"` to split each processed sequence into model inputs and future prediction targets.

```python
from benchpots.datasets import preprocess_random_walk

data = preprocess_random_walk(
    n_steps=30,
    n_features=4,
    n_classes=2,
    n_samples_each_class=100,
    missing_rate=0.1,
    task_type="forecasting",
    n_pred_steps=6,
    random_state=42,
)

print(data["train_X"].shape)      # input history
print(data["train_X_pred"].shape) # forecasting targets
print(data["n_pred_steps"])       # 6
```

For forecasting, task conversion happens before artificial missingness is applied to `*_X`, so `*_X_pred`
retains non-artificial target values for evaluation.

### Real-World Dataset Example

BenchPOTS uses TSDB to acquire supported public datasets and then applies standardized preprocessing.

```python
from benchpots.datasets import preprocess_physionet2012

data = preprocess_physionet2012(
    subset="all",
    rate=0.1,
    pattern="point",
    task_type="imputation",
)

print(data["n_steps"])
print(data["n_features"])
print(data["train_X"].shape)
```

The first load of a public dataset may require network access. Subsequent behavior depends on the underlying TSDB cache.

See the [BenchPOTS documentation](https://docs.pypots.com/en/latest/benchpots.html) for dataset-specific arguments and return contracts.

## ❖ Dataset Preprocessors

BenchPOTS currently exposes the following public preprocessing entry points:

| Dataset / benchmark | API | Typical tasks |
|:--|:--|:--|
| PhysioNet Challenge 2012 | `preprocess_physionet2012` | Imputation, forecasting, classification |
| PhysioNet Challenge 2019 | `preprocess_physionet2019` | Imputation, forecasting, classification |
| Beijing Multi-Site Air Quality | `preprocess_beijing_air_quality` | Imputation, forecasting |
| Italy Air Quality | `preprocess_italy_air_quality` | Imputation, forecasting |
| Electricity Load Diagrams | `preprocess_electricity_load_diagrams` | Imputation, forecasting |
| Electricity Transformer Temperature (ETT) | `preprocess_ett` | Imputation, forecasting |
| PeMS Traffic | `preprocess_pems_traffic` | Imputation, forecasting |
| Solar Alabama | `preprocess_solar_alabama` | Imputation, forecasting |
| UCR & UEA datasets | `preprocess_ucr_uea_datasets` | Classification and task-converted benchmarking |
| Nonlinear benchmarks | `preprocess_nl_benchmarks` | Imputation, forecasting, system-identification experiments |
| Synthetic random walk | `preprocess_random_walk` | Imputation, forecasting, classification, anomaly detection |

All general preprocessors accept the task conversion interface used by BenchPOTS:
`imputation`, `forecasting`, `classification`, `clustering`, and `anomaly_detection`.
The exact labels and metadata available depend on the source dataset.

Missingness patterns currently include:

- `point`: point-wise missingness;
- `subseq`: contiguous subsequence missingness;
- `block`: contiguous time-feature block missingness.

## ❖ PyPOTS Ecosystem

Welcome to the PyPOTS coffee universe! Each library plays a distinct role in the time-series analysis pipeline.

- [**TSDB**](https://github.com/WenjieDu/TSDB) makes public time-series datasets easy to discover, download, and load.
- [**PyGrinder**](https://github.com/WenjieDu/PyGrinder) simulates missingness and creates partially observed data for controlled experiments.
- **BenchPOTS** provides standardized preprocessing pipelines for fair and reproducible evaluation across tasks.
- [**PyPOTS**](https://github.com/WenjieDu/PyPOTS) provides unified modeling for imputation, forecasting, classification, clustering, anomaly detection, and representation learning.
- [**BrewPOTS**](https://github.com/WenjieDu/BrewPOTS) provides tutorials and notebooks for learning the ecosystem.

<p align="center">
<a href="https://pypots.com/ecosystem/">
    <img src="https://pypots.com/figs/pypots_logos/Ecosystem/PyPOTS_Ecosystem_Pipeline.png" width="95%" alt="PyPOTS ecosystem pipeline"/>
</a>
<br>
<b> ☕️ Welcome to the universe of PyPOTS. Enjoy it and have fun!</b>
</p>

## ❖ Citing BenchPOTS/PyPOTS

BenchPOTS is developed as part of the PyPOTS ecosystem. If it supports your research, please cite the
[PyPOTS toolkit paper](https://arxiv.org/abs/2305.18811v2) and the original datasets used in your experiments.

```bibtex
@article{du2023pypots,
title = {{PyPOTS: A Python Toolkit for Data Mining on Partially-Observed Time Series}},
author = {Wenjie Du},
journal = {SIGKDD MiLeTS Workshop},
url = {https://arxiv.org/abs/2305.18811v1},
year = {2023},
}
```

```bibtex
@article{du2025pypots,
title = {{PyPOTS: A Python Toolkit for Machine Learning on Partially-Observed Time Series}},
author = {Wenjie Du and Yiyuan Yang and Linglong Qian and Jun Wang and Qingsong Wen},
journal = {arXiv preprint arXiv:2305.18811v2},
url = {https://arxiv.org/abs/2305.18811v2},
year = {2025},
}
```

## ❖ Contribution

We welcome new dataset preprocessing pipelines, bug fixes, documentation improvements, and regression tests.
Report reproducible problems or propose improvements through [GitHub Issues](https://github.com/WenjieDu/BenchPOTS/issues).

When adding a dataset, keep acquisition in TSDB where appropriate and keep BenchPOTS focused on standardized
benchmark preprocessing rather than duplicating raw-data download infrastructure.

Before submitting a pull request, enable [pre-commit](https://pre-commit.com/) in your local clone:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

GitHub CI checks Ruff linting and formatting, package construction, dependency changes, and compatibility across
the supported Python range. If you find BenchPOTS useful, consider starring the repository to help others discover it.

## ❖ Community

Join the [PyPOTS community](https://github.com/WenjieDu/PyPOTS#-community) for questions, research discussion,
feedback, and ecosystem development. You can also open a [BenchPOTS issue](https://github.com/WenjieDu/BenchPOTS/issues)
for reproducible bugs, dataset requests, or preprocessing improvements.
