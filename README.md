# MobiML

**Framework for machine learning from movement data**

Development of this framework was inspired by https://github.com/wherobots/GeoTorchAI


## Development installation 

[Install uv](https://docs.astral.sh/uv/getting-started/installation/).

Clone this repository.

Set up the project:

```shell
uv sync
```

Run tests:

```shell
uv run pytest
```

In your application that uses mobiml, add these lines to the `pyproject.toml` file:

```yaml
[tool.hatch.metadata]
allow-direct-references = true
```

and install 

```shell
uv add  ../my/local/mobiml
```

For an introduction to uv, see [e.g. the docs](https://docs.astral.sh/uv/getting-started/features/).

## MobiML modules

MobiML contains various modules for learning and data preprocessing for movement data. 

* `datasets`: This module contains classes for handling popular movement datasets.
* `models`: This module contains models for a variety of mobility-related ML tasks.
* `preprocessing`: This module contains tools to preprocess movement data to make it ready for ML development. Preprocessing tools always return a mobiml.Dataset object. 
* `samplers`: This module contains tools for sampling movement data while accounting for its spatiotemporal characteristics. 
* `transforms`: This module contains various transformation operations that can be applied to datasets. Transforms convert a mobiml.Dataset into a different data structure. 


## Documentation

Usage examples are provided in the `examples` directory. 


## Included models

* **GeoTrackNet -- Anomaly detection in maritime traffic patterns** based on https://github.com/CIA-Oceanix/GeoTrackNet, as presented in Nguyen, D., Vadaine, R., Hajduch, G., Garello, R. (2022). GeoTrackNet - A Maritime Anomaly Detector Using Probabilistic Neural Network Representation of AIS Tracks and A Contrario Detection. In IEEE Transactions on Intelligent Transportation Systems, 23(6). arXiv:1912.00682
* **Nautilus -- Vessel Route Forecasting (VRF)** based on https://github.com/DataStories-UniPi/Nautilus, as presented in Tritsarolis, A., Pelekis, N., Bereta, K., Zissis, D., & Theodoridis, Y. (2024). On Vessel Location Forecasting and the Effect of Federated Learning. In Proceedings of the 25th Conference on Mobile Data Management (MDM). arXiv:2405.19870.
* SummarizedAISTrajectoryClassifier -- A basic example model implementing LogisticRegression for trajectory classification in a federated learning setting.


## Publications

[0] Graser, A. & Dragaschnig, M. (2025). Learning From Trajectory Data With MobiML. Workshop on Big Mobility Data Analysis (BMDA2025) in conjuction with EDBT/ICDT 2025.


## Acknowledgements

This work was supported in part by the Horizon Framework Programme of the European Union under grant agreement No. 101070279 ([MobiSpaces](https://mobispaces.eu)). 
