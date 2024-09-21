# vAnalytics - time series analytics for vLLM

<!-- Project Status -->
[![GitHub release](https://img.shields.io/github/release/leafspark/vAnalytics.svg)](https://github.com/leafspark/vAnalytics/releases)
[![GitHub last commit](https://img.shields.io/github/last-commit/leafspark/vAnalytics.svg)](https://github.com/leafspark/vAnalytics/commits)
[![CI/CD Status](https://img.shields.io/badge/CI%2FCD-passing-brightgreen)]()

<!-- Project Info -->
[![Powered by vLLM](https://img.shields.io/badge/Powered%20by-vLLM-green.svg)](https://github.com/vllm-project/vllm)
![GitHub top language](https://img.shields.io/github/languages/top/leafspark/vAnalytics.svg)
[![Platform Compatibility](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-blue)]()
[![GitHub license](https://img.shields.io/github/license/leafspark/vAnalytics.svg)](https://github.com/leafspark/vAnalytics/blob/main/LICENSE)

<!-- Repository Stats -->
![GitHub stars](https://img.shields.io/github/stars/leafspark/vAnalytics.svg)
![GitHub forks](https://img.shields.io/github/forks/leafspark/vAnalytics.svg)
![GitHub release (latest by date)](https://img.shields.io/github/downloads/leafspark/vAnalytics/latest/total?color=green)
![GitHub repo size](https://img.shields.io/github/repo-size/leafspark/vAnalytics.svg)
![Lines of Code](https://tokei.rs/b1/github/leafspark/vAnalytics?category=code)

<!-- Contribution -->
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Issues](https://img.shields.io/github/issues/leafspark/vAnalytics)](https://github.com/leafspark/vAnalytics/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/leafspark/vAnalytics/pulls)

vAnalytics provides a web interface to help easily monitor vLLM instance metrics. It allows users to easily monitor multiple vLLM instances, as well as being easy to setup and configure.

## Features
- Specify vLLM backends easily using name and host configuration
- Uses SQLite for easy database management
- Intuitive and includes error handling
- Flexible schemas and data plotting using Plotly

## Usage
Configure your instances in monitor.py, then use `python src/monitor.py`. This will start monitoring in a `/data` folder, where it will store SQLite databases with your model name.

To start the web interface, execute `python src/graph.py`. The web interface is avaliable at `localhost:4412`.
