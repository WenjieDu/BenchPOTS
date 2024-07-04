from setuptools import find_packages
from setuptools import setup

from benchpots import __version__

with open("./README.md", encoding="utf-8") as f:
    README = f.read()

setup(
    name="benchpots",
    version=__version__,
    description="A Python Toolbox for Benchmarking Machine Learning on Partially-Observed Time Series",
    long_description=README,
    long_description_content_type="text/markdown",
    license="BSD-3-Clause",
    author="Wenjie Du",
    author_email="wenjay.du@gmail.com",
    url="https://github.com/WenjieDu/BenchPOTS",
    project_urls={
        "Documentation": "https://docs.pypots.com/",
        "Source": "https://github.com/WenjieDu/BenchPOTS/",
        "Tracker": "https://github.com/WenjieDu/BenchPOTS/issues/",
        "Download": "https://github.com/WenjieDu/BenchPOTS/archive/main.zip",
    },
    keywords=[
        "data mining",
        "benchmark",
        "neural networks",
        "machine learning",
        "deep learning",
        "artificial intelligence",
        "time-series analysis",
        "time series",
        "imputation",
        "classification",
        "clustering",
        "forecasting",
        "partially observed",
        "irregular sampled",
        "partially-observed time series",
        "incomplete time series",
        "missing data",
        "missing values",
    ],
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    install_requires=[
        "h5py",
        "numpy",
        "pandas",
        "scikit-learn",
        "torch >=1.10",
        "tsdb >=0.5",
        "pygrinder >=0.6",
    ],
    python_requires=">=3.8.0",
    setup_requires=["setuptools>=38.6.0"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
