<p align="center">
  <a href="https://maopl.github.io/TransOpt-doc/">
    <img src="./docs/source/_static/figures/PrismBO.png" alt="" width="40%" align="top">
  </a>
</p>
<p align="center">
  PrismBO: A Data-Centric Benchmarking Platform for Composable Transfer Learning in Bayesian Optimization in Dynamic Environments<br>
  <a href="https://maopl.github.io/TransOpt-doc/">Docs</a> |
  <a href="https://maopl.github.io/TransOpt-doc/quickstart.html">Tutorials</a> |
  <a href="https://maopl.github.io/TransOpt-doc/usage/problems.html">Examples</a> |
  <a href="">Paper</a> |
  <a href="https://maopl.github.io/TransOpt-doc">Citation</a> |
</p>

<div align="center">

<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/python_version-3.10-purple"></a>
<a href="https://opensource.org/licenses/BSD-3-Clause">
  <img alt="License: BSD 3-Clause" src="https://img.shields.io/badge/license-BSD--3--Clause-blue.svg">
</a>
<a href="https://opensource.org/licenses/BSD-3-Clause"> <img alt="Web UI: available" src="https://img.shields.io/badge/web--ui-enabled-brightgreen"> </a>
<a href="https://www.sqlite.org/"> <img alt="Database: SQLite" src="https://img.shields.io/badge/database-sqlite-lightgrey?logo=sqlite"></a>
<a href="https://opensource.org/licenses/BSD-3-Clause"> <img alt="CLI Support" src="https://img.shields.io/badge/cli-support-green"> </a>
<a href="https://opensource.org/licenses/BSD-3-Clause"> <img alt="Multiprocessing" src="https://img.shields.io/badge/multiprocessing-enabled-blueviolet"> </a>

</div>


# Welcome to PrismBO!

**PrismBO** is an open-source software platform designed to facilitate the **design, benchmarking, and application of transfer learning for Bayesian optimization (TLBO)** algorithms through a modular, data-centric framework.

## Features

- **Contains more than 1000 benchmark problems covers diverse range of domains**.  
- **Build custom optimization algorithms as easily as stacking building blocks**.  
- **Leverage historical data to achieve more efficient and informed optimization**.  
- **Deploy experiments through an intuitive web UI and monitor results in real-time**.

PrismBO empowers researchers and developers to explore innovative optimization solutions effortlessly, bridging the gap between theory and practical application.

# [Installation: how to install PrismBO](https://maopl.github.io/TransOpt-doc/installation.html)

PrismBO is composed of two main components: the backend for data processing and business logic, and the frontend for user interaction. Each can be installed as follows:

### Prerequisites

Before installing PrismBO, you must have the following installed:

- **Python 3.10+**: Ensure Python is installed.
- **Node.js 17.9.1+ and npm 8.11.0+**: These are required to install and build the frontend. [Download Node.js](https://nodejs.org/en/download/)

Please install these prerequisites if they are not already installed on your system.

1. Clone the repository:
   ```shell
   $ git clone https://github.com/COLA-Laboratory/PrismBO
   ```

2. Install the required dependencies:
   ```shell
   $ cd prismbo
   $ python setup.py install
   $ pip install .
   ```

3. Install the frontend dependencies:
   ```shell
   $ cd webui && npm install
   ```

### Start the Backend Agent

To start the backend agent, use the following command:

```bash
$ python prismbo/agent/app.py
```

### Web User Interface Mode

When PrismBO has been started successfully, go to the webui directory and start the web UI on your local machine. Enable the user interface mode with the following command:
```bash
cd webui && npm start
```

This will open the PrismBO interface in your default web browser at `http://localhost:3000`.


### Command Line Mode

In addition to the web UI mode, PrismBO also offers a Command Line (CMD) mode for users who may not have access to a display screen, such as when working on a remote server.

To run PrismBO in CMD mode, use the following command:

```bash
python PrismBO/agent/run_cli.py -n Sphere -v 3 -o 1 -m RF -acf UCB -b 300
```

This command sets up a task named Sphere with 3 variables and 1 objectives, using a Random Forest model (RF) as surrogate model and the upper confidence bound (UCB) acquisition function, with a budget of 300 function evaluations.

For a complete list of available options and more detailed usage instructions, please refer to the [CLI documentation](https://maopl.github.io/TransOpt-doc/usage/cli.html).


# [Documentation: The PrismBO Process](https://maopl.github.io/TransOpt-doc/)

Our docs walk you through using PrismBO, web UI and key API points. For an overview of the system and workflow for project management, see our documentation [documentation](https://maopl.github.io/TransOpt-doc/).


<p align="center">
<img src="./docs/source/_static/figures/workflow_prismbo.svg" width="95%">
</p>


# Why use PrismBO?

Recent years, Bayesian optimization (BO) has been widely used in various fields, such as hyperparameter optimization, molecular design, and synthetic biology. However, conventional BO is not that efficient, where it conduct every optimization task from scratch while ignoring the experiences gained from previous problem-solving practices. To address this challenge, transfer learning (TL) has been introduced to BO, aiming to leverage auxillary data to improve the optimization efficiency and performance. Despite the potential of TLBO, the usage of TLBO is still limited due to the complexity of advanced TLBO methods. PrismBO, a system that facilitates:

- development of TLBO algorithms;
- benchmarking the performance of TLBO methods;
- applications of TLBO for downstream tasks;

<p align="center">
<img src="./docs/source/_static/figures/simplecase.svg" width="95%">
</p>


# Performance of PrismBO?

|Problem instance   |   PrismBO_RGPE |   PrismBO_MHGP |   hyperopt |   smac |   hebo |   hyperbo |
|:-----------------------|---------------:|---------------:|-----------:|-------:|-------:|----------:|
| Ackley_0                 |          17.23 |          17.98 |      17.66 |  17.93 |  `17.22` |     17.42 |
| Ackley_1                 |          `14.59` |          15.42 |      17.14 |  16.55 |  17.25 |     17.22 |
| Ackley_2                |          `16.78` |          17.12 |      17.79 |  17.24 |  16.92 |    17.89 |
| Rastrigin_0             |          72.68 |          71.48 |      72.79 |  `68.39` |  69.21 |     72.84 |
| Rastrigin_1             |          64.95 |          `64.93` |      70.79 |  71.83 |  68.59 |     69.11 |
| Rastrigin_2             |          `56.59` |          59.51 |      62.16 |  64.55 |  66.21 |     63.89 |
| ResNet18_MNIST     | `0.992`        | 0.986  | 0.993         | 0.989  | 0.990         | 0.987  |
| ResNet18_CIFAR-10  | `0.888`        | 0.887  | 0.867         | 0.871  | 0.875         | 0.868  |
| ResNet18_CIFAR-100 | `0.694`        | 0.671  | 0.662         | 0.635  | 0.648         | 0.630  |
| GCC_2mm    | 1.934        | 2.021  | `1.915`         | 2.000  | 1.922         | 2.010  |
| GCC_3mm    | `3.120`        | 3.167  | 3.142         | 3.215  | 3.133         | 3.245  |
| GCC_corr   | 2.587        | `2.559`  | 2.676         | 2.642  | 2.570         | 2.660  |




# Reference & Citation

If you find our work helpful to your research, please consider citing our:

```bibtex
@article{PrismBO,
  title = {PrismBO: A Data-Centric Benchmarking Platform for Composable Transfer Learning in Bayesian Optimization in Dynamic Environments
},
  author = {Peili Mao and Ke Li},
  url = {https://github.com/COLA-Laboratory/PrismBO/},
  year = {2025}
}
```



