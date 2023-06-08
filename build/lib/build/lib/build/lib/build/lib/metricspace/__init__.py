"""
Metric Space Analysis - A Python Implementation
===============================================

This module provides a Python implementation of the metric space analysis algorithms originally 
theorized in:

- `Metric-space analysis of spike trains: theory, algorithms, and application` (Victor & Purpura, 1997)
- `Nature and precision of temporal coding in visual cortex: a metric-space analysis` (Victor & Purpura, 1996)

For a complete walkthrough of cost-based metrics, refer to [Jonathan Victor's website](http://www-users.med.cornell.edu/~jdvicto/metricdf.html#introduction). 

In this repository, a Python implementation of the metric space analysis algorithms has been hosted, with several optimizations:
- More computationally intensive functions have been implemented in Rust and compiled into a shared library that can be utilized within Python.
- Spike train loops are vectorized, thus limiting the numpy "auto-vectorization" safety and leveraging the power of AVX2 vector instructions in modern CPUs.
- Parallelization of independent spike trains is achieved using the multiprocessing library.

This package also introduces a modified "sliding window" approach for calculating spike distances for spike trains of unequal length.

Installation
------------
To install this package, run the command: `pip install metricspace`

Ensure Python 3.7 or higher is used so that the Rust library can be compiled correctly and has access to your python interpreter.

Usage
-----
The following functions are exposed by this package:
- `spkd`: Calculates the spike distance between two or more spike trains.
- `spkd_slide`: Calculates the spike distance between two or more spike trains using a sliding window approach.
- `distclust`: Uses spike distance to cluster spike trains for entropy calculations.
- `tblxinfo`: Uses the distclust confusion matrix output (probability, not count) to calculate mutual information.
- `tblxtpbi`: Similar to tblxinfo but with Treves and Panzeri's bias correction.
- `tblxbi`: Similar to tblxinfo but with jacknife or tp bias correction.

See the README or module-level function docstrings for more information and usage examples.

Contributions
-------------
Any contributions, improvements or suggestions are welcome.

Original Developers
-------------------
- Jonathan D. Victor: jdvicto@med.cornell.edu
- Keith P. Purpura: kpurpura@med.cornell.edu
- Dmitriy Aronov: aronov@mit.edu
"""

from .model import *
from .entropy import *

__all__ = ['distclust', 'spkd', 'spkd_slide', 'histinfo', 'histjabi', 'histbi', 'tblxbi', 'histtpbi', 'tblxtpbi', 'tblxinfo']
