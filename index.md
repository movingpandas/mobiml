![MobiML](https://github.com/user-attachments/assets/aa50836e-5523-4df5-8bd1-3b829106991d)

MobiML is a Python framework for machine learning from movement data and part of the [MovingPandas](https://movingpandas.org) ecosystem. 

[![Tests](https://github.com/movingpandas/mobiml/actions/workflows/tests.yaml/badge.svg)](https://github.com/movingpandas/mobiml/actions/workflows/tests.yaml)

For API documentation and source code, visit the [GitHub repository](https://github.com/movingpandas/mobiml).


## Installation

**Note:** MobiML requires Linux (one of its main dependencies, [pymeos](https://github.com/MobilityDB/PyMEOS), is not yet available on Windows).

[Install uv](https://docs.astral.sh/uv/getting-started/installation/), clone the repository, then:

```shell
uv sync
```

In your application's `pyproject.toml`:

```toml
[tool.hatch.metadata]
allow-direct-references = true
```

Then install:

```shell
uv add ../my/local/mobiml
```


## Modules

MobiML contains modules for learning and data preprocessing from movement data:

- **datasets** — Classes for handling popular movement datasets
- **models** — Models for a variety of mobility-related ML tasks
- **preprocessing** — Tools to preprocess movement data into `mobiml.Dataset` objects, ready for ML development
- **samplers** — Tools for sampling movement data while accounting for spatiotemporal characteristics
- **transforms** — Transformation operations that convert a `mobiml.Dataset` into different data structures


## Included Models

- **GeoTrackNet** — Anomaly detection in maritime traffic patterns, based on [GeoTrackNet](https://github.com/CIA-Oceanix/GeoTrackNet). *Nguyen et al. (2022). GeoTrackNet - A Maritime Anomaly Detector Using Probabilistic Neural Network Representation of AIS Tracks and A Contrario Detection. IEEE Transactions on Intelligent Transportation Systems, 23(6). [arXiv:1912.00682](https://arxiv.org/abs/1912.00682)*

- **Nautilus** — Vessel Route Forecasting, based on [Nautilus](https://github.com/DataStories-UniPi/Nautilus). *Tritsarolis et al. (2024). On Vessel Location Forecasting and the Effect of Federated Learning. MDM 2024. [arXiv:2405.19870](https://arxiv.org/abs/2405.19870)*

- **SummarizedAISTrajectoryClassifier** — A basic example model implementing Logistic Regression for trajectory classification in a federated learning setting


## Examples

Usage examples are provided as Jupyter notebooks in the [examples directory](https://github.com/movingpandas/mobiml/tree/main/examples).


## Publications

[0] [Graser, A. & Dragaschnig, M. (2025). Learning From Trajectory Data With MobiML. Workshop on Big Mobility Data Analysis (BMDA2025) in conjunction with EDBT/ICDT 2025.](https://ceur-ws.org/Vol-3946/BMDA-6.pdf)

```bibtex
@inproceedings{graser2025learning,
  title={Learning From Trajectory Data With {MobiML}},
  author={Graser, Anita and Dragaschnig, Melitta},
  booktitle={Proceedings of the Workshop on Big Mobility Data Analysis (BMDA2025) in conjunction with EDBT/ICDT},
  year={2025},
  url={https://ceur-ws.org/Vol-3946/BMDA-6.pdf}
}
```


## Acknowledgements

This work was supported in part by the Austrian Federal Ministry for Transport, Innovation and Technology (BMVIT) within the programme ‘AI for Green 2023’ under project No. FO999910218 ([AI4PT](https://projekte.ffg.at/projekt/5121351)) as well as by the Horizon Framework Programme of the European Union under grant agreement No. 101070279 ([MobiSpaces](https://mobispaces.eu)).
