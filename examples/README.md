
# MobiML Examples

**Note**: As of today (2026-03-17), one of our main dependencies, pymeos, [is not available on Windows](https://github.com/MobilityDB/PyMEOS/issues/1). Therefore we recommend using MobiML on Linux. 


## Environment Setup

To create environment for the example notebooks run:

```shell
conda env create -f environment-viz.yml
```


### Federated Learning Examples

Due to Flower's specific requirements, the `mobiml-fl-demo.ipynb` needs a different environment:

```shell
conda env create -f environment-flwr.yml
```


