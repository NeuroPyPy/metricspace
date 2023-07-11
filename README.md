
# Metric Space Analysis - A Python & Rust Implementation

<br>

![Python](https://img.shields.io/badge/python-3670A0?style=?style=plastic&logo=python&logoColor=ffdd54)
![Rust](https://img.shields.io/badge/rust-000000?style=?style=plastic&logo=rust&logoColor=white)
![GitHub](https://img.shields.io/github/license/NeuroPyPy/metricspace?style=plastic)
[![CI](https://github.com/NeuroPyPy/rs-distances/actions/workflows/CI.yml/badge.svg)](https://github.com/NeuroPyPy/rs-distances/actions/workflows/CI.yml)
[![implementation](https://img.shields.io/pypi/implementation/metricspace)](https://pypi.org/project/metricspace/)
[![PyPI version](https://badge.fury.io/py/metricspace.svg)](https://badge.fury.io/py/metricspace)
[![PyPI format](https://img.shields.io/pypi/format/metricspace.svg)](https://pypi.python.org/pypi/metricspace/)

<br>

* <a href=https://journals.physiology.org/doi/abs/10.1152/jn.1996.76.2.1310> Nature and precision of temporal coding in visual cortex: a metric-space analysis. Victor & Purpura (1996)</a>
* <a href="https://www.tandfonline.com/doi/abs/10.1088/0954-898X_8_2_003"> Metric space analysis of spike trains: theory, algorithms and application. Victor & Purpura (1997) </a>

<br>

For a full walkthrough of cost-based metrics, see Jonathon Victor's <a href="http://www-users.med.cornell.edu/~jdvicto/metricdf.html#introduction"> website: </a>

> Spike trains are considered to be points in an abstract topological space. A spike train metric is a rule which assigns a non-negative number D(Sa,Sb) to pairs of spike trains Sa and Sb which expresses how dissimilar they are.

<br>

This repository hosts a Python implementation of the metric space analysis algorithms with several optimizations:
* The more computationally intensive functions are implemented in Rust (with benchmarks for matlab, python and rust) and compiled into a shared library that can be utilized within Python.
* Spike train loops are vectorized, limiting the numpy "auto-vectorization" safety and leveraging the power of AVX2 vector instructions in modern CPUs.
* Parallelization of independent spike-trains using the multiprocessing library (multithreading in the works).

<br>

In addition to the standard approach for spike-distance calculations, this package exposes a modified "sliding window" approach that can be used to calculate spike distances for spike trains of unequal length.

<br>

----

## Installation

<br>

To install this package, run the following command:
```bash
pip install metricspace
```
**Note**: Be sure to activate a vertual env (penv or conda env) with Python 3.7 or higher before installing this package so that the Rust library can be compiled correctly and has access to your python interpreter.

<br>

### Installation with pipenv

**Ensure your pip is up-to-date, and confirm activated venv**

| MacOS/Unix                                      | Windows                                            |
|:------------------------------------------------|:---------------------------------------------------|
| `python3 -m pip install --upgrade pip`          | `py -m pip install --upgrade pip`                  |
| `python3 -m pip --version`                      | `py -m pip --version`                              |
| `python3 -m pip install --user virtualenv`      | `py -m pip install --user virtualenv`              |
| `python3 -m venv env`                           | `py -m env_metricspace env`                       |
| `source env/bin/activate`                       | `.\env\Scripts\activate`                           |
| `.../env/bin/python`                            |                                                     |

**Validate your active interpreter is in your venv and install metricspace**

| MacOS/Unix                                      | Windows                                            |
|:------------------------------------------------|:---------------------------------------------------|
| `which python`                                  | `where python`                                     |
| `.../env/bin/python`                            | `...\env_metricspace\Scripts\python.exe`           |
| `python3 -m pip install metricspace`            | `py -m pip install metricspace`                    |

<br>

----


## Usage

<br>

### Exposed Functions
The following functions are exposed by this package:
* `spkd` - Calculates the spike distance between two or more spike trains.
* `spkd_slide` - Calculates the spike distance between two or more spike trains using a sliding window approach.
* `distclust` - Uses spike distance to cluster spike trains for entropy calculations.
* `tblxinfo` -  Uses the distclust confusion matrix output (probability, not count) to calculate mutual information.
* `tblxtpbi` - Similar to tblxinfo but with Treves and Panzeri's bias correction.
* `tblxbi` - Similar to tblxinfo but with jacknife or tp bias correction.

<br>

### Example

```python
import metricspace as ms
import numpy as np

# Generate random spike trains
spike_train_A = np.sort(np.random.uniform(low=0.0, high=2, size=100))
spike_train_B = np.sort(np.random.uniform(low=0.0, high=2, size=100))

# Input spike trains into a list or array (as many or few as you want)
spike_trains = [spike_train_A, spike_train_B]

# Make array of cost values to be used in the spike-distance calculation (here we get 0 to 512)
costs = np.concatenate(([0], 2 ** np.arange(-4, 9.5, 0.5)))

spike_distance = ms.spkd(spike_trains, costs)  # Standard approach
spike_distance_slide = ms.spkd_slide(spike_trains, costs, 10e-3)  # Sliding window approach with search window of 1ms

# Cluster spike trains using spike distance and the number of samples in each class
spike_train_class_labels = np.concatenate((np.zeros(100), np.ones(100))) # 100 samples in each class, randomly generated
_, nsam = np.unique(spike_train_class_labels, return_counts=True)
clustered = ms.distclust(spike_distance, nsam)

# Calculate entropy from the confusion matrix output of distclust
mi = ms.tblxinfo(clustered)
mj = ms.tblxjabi(clustered)
mt = ms.tblxtpbi(clustered)
mij = mi + mj
mit = mi + mt

```

<br>

----

## Performance
The functions provided in this package are written in Rust and compiled into a shared library that can be utilized within Python. This approach is intended to boost the computational efficiency of metric space analysis operations.

Below is a comparative performance table of the spike-train iterator function implemented in Matlab, Python, and Rust. It should be noted that the Matlab version is not optimized using MEX (which would be comparable to Python's numba @jit), and the translations from Matlab to Python to Rust are not exact 1:1.

| Spike-train iterator   | Matlab  | Python  | Rust   |
| ---------------------- | ------- | ------- | ------ |
| `raw function`         | 30.235s | 64.992s | 2.028s |
| `with numba @jit`      | 30.235s | 25.119s | 2.028s |
| `with @jit + parralel` | 24.050s | 18.067s | 0.945s |

<br>

## Advantages of Rust Implementation
Array manipulations, particularly those performed within computationally intensive tasks, are highly sensitive to memory allocation and cleanup. Rust, with its ownership model and automatic memory management, excels in this area. Rust automatically reclaims the memory when an object (like an array or a slice) goes out of scope. This is a stark contrast to languages like Python, where a garbage collector is relied upon to perform memory cleanup. This difference provides Rust implementations with a distinct edge in performance, which is reflected in the comparative analysis shown above.

With these Rust implementations, you can achieve the high-level expressiveness of Python while benefiting from the superior performance and efficiency of Rust.


<br>


----

<br>

## Contributions

Any contributions, improvements or suggestions are welcome.

### Original Developers
Jonathan D. Victor: jdvicto@med.cornell.edu
Keith P. Purpura: kpurpura@med.cornell.edu
Dmitriy Aronov: aronov@mit.edu
