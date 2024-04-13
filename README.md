# solar-data-tools

|       |  |
| ----------- | ----------- |
| Latest Release      | [![Latest Release PyPI](https://img.shields.io/pypi/v/solar-data-tools.svg)](https://pypi.org/project/solar-data-tools/)       [![Latest Release Anaconda](https://anaconda.org/slacgismo/solar-data-tools/badges/version.svg)](https://anaconda.org/slacgismo/solar-data-tools)[![](https://anaconda.org/slacgismo/solar-data-tools/badges/latest_release_date.svg)](https://anaconda.org/slacgismo/solar-data-tools)
|License| [![License](https://img.shields.io/pypi/l/solar-data-tools.svg)](https://github.com/slacgismo/solar-data-tools/blob/master/LICENSE)
|Build Status|[![docs](https://readthedocs.org/projects/solar-data-tools/badge/?version=stable)](https://solar-data-tools.readthedocs.io/en/stable/") [![tests](https://github.com/slacgismo/solar-data-tools/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/slacgismo/solar-data-tools/actions/workflows/test.yml) [![build](https://travis-ci.com/tadatoshi/solar-data-tools.svg?branch=development)](https://travis-ci.com/tadatoshi/solar-data-tools.svg?branch=development)
|Publications|[![DOI](https://zenodo.org/badge/171066536.svg)](https://zenodo.org/badge/latestdoi/171066536)
|PyPI Downloads|[![PyPI Download Count](https://img.shields.io/pypi/dm/solar-data-tools)](https://pepy.tech/project/solar-data-tools)
|Conda Downloads|[![Conda download count](https://anaconda.org/slacgismo/solar-data-tools/badges/downloads.svg)](https://anaconda.org/slacgismo/solar-data-tools)
|Test Coverage|![test coverage](https://img.shields.io/badge/test--coverage-45%25-yellowgreen)


Tools for performing common tasks on solar PV data signals. These tasks include finding clear days in
a data set, common data transforms, and fixing time stamp issues. These tools are designed to be
automatic and require little if any input from the user. Libraries are included to help with data IO
and plotting as well.

There is close integration between this repository and the [Statistical Clear Sky](https://github.com/slacgismo/StatisticalClearSky) repository, which provides a "clear sky model" of system output, given only measured power as an input.

See [notebooks](/notebooks) folder for examples.

## Install & Setup

### 3 ways of setting up, either approach works:

#### 1) Recommended: Install with pip

In a fresh Python virtual environment, simply run

```sh
pip install solar-data-tools
```

#### 2) Create a conda virtual environment (using make)

```sh
make make-env # Create a new conda environment
make activate-env # Activate the conda environment
make deactivate-env # Deactivate the conda environment
make update-env # Update the conda environment
```


Additional documentation on setting up the Conda environment is available [here](https://github.com/slacgismo/pvinsight-onboarding/blob/main/README.md).

#### 3) General Anaconda Package

```sh
conda install slacgismo::solar-data-tools
```

### Solvers

#### QSS & CLARABEL

By default, [QSS](https://github.com/cvxgrp/qss) and CLARABEL solvers are used for non-convex and convex problems, respectively. Both are supported by [OSD](https://github.com/cvxgrp/signal-decomposition/tree/main), the modeling language used to solve signal decomposition problems in Solar Data Tools, and both are open source.

#### MOSEK

[MOSEK](https://www.mosek.com/resources/getting-started/) is a commercial software package. It is more stable and offers faster solve times. The included YAML/requirements.txt file will install MOSEK for you, but you will still need to obtain a license.

More information is available here:
* [Free 30-day trial](https://www.mosek.com/products/trial/)
* [Personal academic license](https://www.mosek.com/products/academic-licenses/)

## Usage

Users will primarily interact with this software through the `DataHandler` class. If you would like to specify a solver, just pass the keyword argument `solver` to `dh.pipeline` with the solver of choice. Passing QSS will keep the convex problems solver as OSQP, unless `solver_convex=QSS` is passed as well. Setting `solver=MOSEK` will set the solver to MOSEK for convex and non-convex problems by default.

```python
from solardatatools import DataHandler
from solardatatools.dataio import get_pvdaq_data

pv_system_data = get_pvdaq_data(sysid=35, api_key='DEMO_KEY', year=[2011, 2012, 2013])

dh = DataHandler(pv_system_data)
dh.run_pipeline(power_col='dc_power')
```
If everything is working correctly, you should see something like the following

```
total time: 24.27 seconds
--------------------------------
Breakdown
--------------------------------
Preprocessing              11.14s
Cleaning                   0.94s
Filtering/Summarizing      12.19s
    Data quality           0.25s
    Clear day detect       1.75s
    Clipping detect        7.77s
    Capacity change detect 2.42s
```

## Contributors
> [!NOTE]
> Must enable pre-commit hook before pushing any contributions

```sh
pip install pre-commit ruff
pre-commit install
```

Run pre-commit hook on all files
```sh
pre-commit run --all-files
```

## Test Coverage

In order to view the current test coverage metrics, run:
```sh
coverage run --source solardatatools -m unittest discover && coverage html
open htmlcov/index.html
```

## Versioning

We use [Semantic Versioning](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/slacgismo/solar-data-tools/tags).

## Authors

* **Bennet Meyers** - *Initial work and Main research work* - [Bennet Meyers GitHub](https://github.com/bmeyers)

See also the list of [contributors](https://github.com/bmeyers/solar-data-tools/contributors) who participated in this project.

## License

This project is licensed under the BSD 2-Clause License - see the [LICENSE](LICENSE) file for details
