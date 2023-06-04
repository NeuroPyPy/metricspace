
## Metric Space Analysis - A Python Implementation 

![Python](https://img.shields.io/badge/python-3670A0?style=?style=plastic&logo=python&logoColor=ffdd54)
[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/NeuroPyPy/metricspace/blob/master/LICENSE)

* <a href=https://journals.physiology.org/doi/abs/10.1152/jn.1996.76.2.1310> Nature and precision of temporal coding in visual cortex: a metric-space analysis. Victor & Purpura (1996)</a>
* <a href="https://www.tandfonline.com/doi/abs/10.1088/0954-898X_8_2_003"> Metric space analysis of spike trains: theory, algorithms and application. Victor & Purpura (1997) </a>
<br>

For a full walkthrough of cost-based metrics, see Dr. Jonathon Victor's <a href="http://www-users.med.cornell.edu/~jdvicto/metricdf.html#introduction"> website: </a> 

> Spike trains are considered to be points in an abstract topological space. A spike train metric is a rule which assigns a non-negative number D(Sa,Sb) to pairs of spike trains Sa and Sb which expresses how dissimilar they are.
 
This repository hosts a Python implementation of the metric space analysis algorithms with several optimizations:
* The more computationally intensive functions are implemented in Rust and compiled into a shared library that can be utilized within Python.
* Vectorized array computation leveraging the power of numpy.
* Parallelization of independent spike-trains using the multiprocessing library.

In addition to the standard approach for spike-distance calculations, this package exposes a modified "sliding window" approach that can be used to calculate spike distances for spike trains of unequal length.

## Installation
To install this package, run the following command:
```bash
pip install metricspace
```
or with conda package manager:
```bash
conda install -c  metricspace
```
**Note**: Be sure to activate your virtual environment with Python 3.7 or higher before installing this package via pip or anaconda so that the Rust library can be compiled correctly and has access to your python interpreter.

## Exposed Functions
The following functions are exposed by this package:
* `spkd` - Calculates the spike distance between two or more spike trains.
* `spkd_slide` - Calculates the spike distance between two or more spike trains using a sliding window approach.
* `distclust` - Uses spike distance to cluster spike trains for entropy calculations.
* `tblxinfo` -  Uses the distclust confusion matrix output (probability, not count) to calculate mutual information.
* `tblxtpbi` - Similar to tblxinfo but with Treves and Panzeri's bias correction.
* `tblxjabi` - Similar to tblxinfo but with jacknife bias correction.

## Usage
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


### Original Developers
Jonathan D. Victor: jdvicto@med.cornell.edu
Keith P. Purpura: kpurpura@med.cornell.edu
Dmitriy Aronov: aronov@mit.edu 