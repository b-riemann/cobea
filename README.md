# About

The _cobea_ module [1] is a Python implementation of **C**losed-**O**rbit **B**ilinear-**E**xponential **A**nalysis [2], an algorithm for studying closed-orbit response matrices of storage rings (particle accelerators).

(see [CHANGELOG.md](CHANGELOG.md) for version history)

# Usage

_cobea_ is distributed under the [Simplified BSD License](LICENSE). If you publish material using this software, please cite one or more of the references [1-2].

## Installation and Dependencies

The _cobea_ module is implemented both for Python 2 (>=2.7) and 3 (>= 3.6), using the [SciPy ecosystem](https://www.scipy.org).

The module is installed using the standard _setuptools_ process. After cloning the repository to your system, just run

    sudo python setup.py develop

and you can use your local module by `import cobea` in any directory on your system, changes being updated when you pull the repository.

## Workflow / Help

Usage of _cobea_ essentially consists of three steps:

* Create a valid `Response` object from your input data,
* Send the object to the `cobea` function,
* Receive the `Result` object, plotting it using the function `plotting.plot_result` or similar.

For details, please view the (incomplete) [manual](doc/manual.pdf), which was largely generated from Python docstrings using [Sphinx](http://www.sphinx-doc.org).

# Contact

If you have questions or need help for applying the code to your accelerator, feel free to contact me (see [setup.py](setup.py) for email address) - I will answer as soon as possible.

# References

[1] B. Riemann _et al._: ''COBEA - Optical Parameters From Response Matrices Without Knowledge of Magnet Strengths'', in [Proc. IPAC 17, MOPIK066](http://accelconf.web.cern.ch/AccelConf/ipac2017/papers/mopik066.pdf) (2017).

[2] B. Riemann: ''The Bilinear-Exponential Closed-Orbit Model and its Application to Storage Ring Beam Diagnostics'', Ph.D. Dissertation (TU Dortmund University, 2016). [doi:10.17877/DE290R-17221](https://doi.org/10.17877/DE290R-17221)
