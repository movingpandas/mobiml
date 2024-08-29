# MobiML

**Framework for machine learning from movement data**

Development of this framework was inspired by https://github.com/wherobots/GeoTorchAI


## Development installation 

To run the tests, use the mobiml environment:

```
mamba env create -f environment.yml
```

To run the Flower examples, first create the mobiml-flwr environment:

```
mamba env create -f environment-flwr.yml
```

Then activate the environment and install mobiml [in editable mode](https://pip-python3.readthedocs.io/en/latest/reference/pip_install.html#editable-installs)

```shell
pip install -e .
```


### Environment notes

As of August 2024, pip installing the current latest Flower version (flwr-1.10.0) requires numpy<2.0.0,>=1.21.0


## MobiML modules

MobiML contains various modules for learning and data preprocessing for movement data. 

* Datasets: This module contains classes for handling popular movement datasets.
* Models: This module contains models for a variety of mobility-related ML tasks.
* Preprocessing: This module contains tools to preprocess movement data to make it ready for ML development. Preprocessing tools always return a mobiml.Dataset object. 
* Transforms: This module contains various tranformations operations that can be applied to datasets. Transforms convert a mobiml.Dataset into a different data structure. 


## Documentation

Usage examples are provided in the `examples` directory. 


## Included models

* **Nautilus** Vessel Route Forecasting (VRF) -- Based on https://github.com/DataStories-UniPi/Nautilus, as presented in Tritsarolis, A., Pelekis, N., Bereta, K., Zissis, D., & Theodoridis, Y. (2024). On Vessel Location Forecasting and the Effect of Federated Learning. In Proceedings of the 25th Conference on Mobile Data Management (MDM). arXiv preprint arXiv:2405.19870.
* SummarizedAISTrajectoryClassifier -- A basic example model implementing LogisticRegression for trajectory classification in a federated learning setting.


## Acknowledgements

This work was supported in part by the Horizon Framework Programme of the European Union under grant agreement No. 101070279 (MobiSpaces; https://mobispaces.eu). 
