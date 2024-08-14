"""
BenchPOTS: a Python toolbox for benchmarking machine learning on POTS (Partially-Observed Time Series),
supporting processing pipelines of 170+ public time-series datasets
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from . import utils, datasets
from .version import __version__

__all__ = [
    "__version__",
    "utils",
    "datasets",
]
