<a href="https://github.com/WenjieDu/BenchPOTS">
    <img src="https://pypots.com/figs/pypots_logos/BenchPOTS/logo_FFBG.svg" width="200" align="right">
</a>

<h3 align="center">Welcome to BenchPOTS</h3>

<p align="center"><i>a Python toolbox for benchmarking ML on POTS (Partially-Observed Time Series)</i></p>

<p align="center">
    <a href="https://docs.pypots.com/en/latest/install.html#reasons-of-version-limitations-on-dependencies">
       <img alt="Python version" src="https://img.shields.io/badge/Python-v3.8+-E97040?logo=python&logoColor=white">
    </a>
    <a href="https://github.com/WenjieDu/BenchPOTS/releases">
        <img alt="the latest release version" src="https://img.shields.io/github/v/release/wenjiedu/benchpots?color=EE781F&include_prereleases&label=Release&logo=github&logoColor=white">
    </a>
    <a href="https://github.com/WenjieDu/BenchPOTS/blob/main/LICENSE">
        <img alt="BSD-3 license" src="https://img.shields.io/badge/License-BSD--3-E9BB41?logo=opensourceinitiative&logoColor=white">
    </a>
    <a href="https://github.com/WenjieDu/PyPOTS#-community">
        <img alt="Community" src="https://img.shields.io/badge/join_us-community!-C8A062">
    </a>
    <a href="https://github.com/WenjieDu/BenchPOTS/graphs/contributors">
        <img alt="GitHub contributors" src="https://img.shields.io/github/contributors/wenjiedu/benchpots?color=D8E699&label=Contributors&logo=GitHub">
    </a>
    <a href="https://star-history.com/#wenjiedu/benchpots">
        <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/wenjiedu/benchpots?logo=None&color=6BB392&label=%E2%98%85%20Stars">
    </a>
    <a href="https://github.com/WenjieDu/BenchPOTS/network/members">
        <img alt="GitHub Repo forks" src="https://img.shields.io/github/forks/wenjiedu/benchpots?logo=forgejo&logoColor=black&label=Forks">
    </a>
    <a href="https://codeclimate.com/github/WenjieDu/BenchPOTS">
        <img alt="Code Climate maintainability" src="https://img.shields.io/codeclimate/maintainability-percentage/WenjieDu/BenchPOTS?color=3C7699&label=Maintainability&logo=codeclimate">
    </a>
    <a href="https://coveralls.io/github/WenjieDu/BenchPOTS">
        <img alt="Coveralls coverage" src="https://img.shields.io/coverallsCoverage/github/WenjieDu/BenchPOTS?branch=main&logo=coveralls&color=75C1C4&label=Coverage">
    </a>
    <a href="https://github.com/WenjieDu/BenchPOTS/actions/workflows/testing_ci.yml">
        <img alt="GitHub Testing" src="https://img.shields.io/github/actions/workflow/status/wenjiedu/benchpots/testing_ci.yml?logo=circleci&color=C8D8E1&label=CI">
    </a>
    <a href="https://docs.pypots.com/en/latest/benchpots.html">
        <img alt="Docs building" src="https://img.shields.io/readthedocs/pypots?logo=readthedocs&label=Docs&logoColor=white&color=395260">
    </a>
    <a href="https://anaconda.org/conda-forge/benchpots">
        <img alt="Conda downloads" src="https://img.shields.io/endpoint?url=https://pypots.com/figs/downloads_badges/conda_benchpots_downloads.json">
    </a>
    <a href="https://pepy.tech/project/benchpots">
        <img alt="PyPI downloads" src="https://img.shields.io/endpoint?url=https://pypots.com/figs/downloads_badges/pypi_benchpots_downloads.json">
    </a>
</p>

To evaluate the performance of algorithms on POTS datasets, a benchmarking toolkit is necessary, hence the ecosystem library BenchPOTS is developed.
BenchPOTS provides the standard and unified preprocessing pipelines of a variety of POTS datasets.
It supports a variety of evaluation tasks to help users understand the performance of different algorithms.


## ❖ Usage Examples
> [!IMPORTANT]
> BenchPOTS is available on both <a alt='PyPI' href='https://pypi.python.org/pypi/benchpots'><img align='center' src='https://img.shields.io/badge/PyPI--lightgreen?style=social&logo=pypi'></a> 
> and <a alt='Anaconda' href='https://anaconda.org/conda-forge/benchpots'><img align='center' src='https://img.shields.io/badge/Anaconda--lightgreen?style=social&logo=anaconda'></a>❗️
> 
> Install via pip:
> > pip install benchpots
> 
> or install from source code:
> > pip install `https://github.com/WenjieDu/BenchPOTS/archive/main.zip`
>
> or install via conda:
> > conda install benchpots -c conda-forge

```python
import benchpots

# Load PhysioNet2012 all three subsets and apply MCAR with 0.1 rate 
benchpots.datasets.preprocess_physionet2012(subset="all", rate="0.1")

```

## ❖ Citing BenchPOTS/PyPOTS
The paper introducing PyPOTS is available [on arXiv](https://arxiv.org/abs/2305.18811),
A short version of it is accepted by the 9th SIGKDD international workshop on Mining and Learning from Time Series ([MiLeTS'23](https://kdd-milets.github.io/milets2023/))).
**Additionally**, PyPOTS has been included as a [PyTorch Ecosystem](https://pytorch.org/ecosystem/) project.
We are pursuing to publish it in prestigious academic venues, e.g. JMLR (track for
[Machine Learning Open Source Software](https://www.jmlr.org/mloss/)). If you use PyPOTS in your work,
please cite it as below and 🌟star this repository to make others notice this library. 🤗

There are scientific research projects using PyPOTS and referencing in their papers.
Here is [an incomplete list of them](https://scholar.google.com/scholar?as_ylo=2022&q=%E2%80%9CPyPOTS%E2%80%9D&hl=en).

<p align="center">
<a href="https://github.com/WenjieDu/PyPOTS">
    <img src="https://pypots.com/figs/pypots_logos/Ecosystem/PyPOTS_Ecosystem_Pipeline.png" width="95%"/>
</a>
</p>

``` bibtex
@article{du2023pypots,
title={{PyPOTS: a Python toolbox for data mining on Partially-Observed Time Series}},
author={Wenjie Du},
journal={arXiv preprint arXiv:2305.18811},
year={2023},
}
```
or
> Wenjie Du.
> PyPOTS: a Python toolbox for data mining on Partially-Observed Time Series.
> arXiv, abs/2305.18811, 2023.



<details>
<summary>🏠 Visits</summary>
<a href="https://github.com/WenjieDu/BenchPOTS">
    <img alt="BenchPOTS visits" align="left" src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FWenjieDu%2FBenchPOTS&count_bg=%23009A0A&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visits%20since%20June%202024&edge_flat=false">
</a>
</details>
<br>
