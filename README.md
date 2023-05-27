# movingml



## Development installation 


```
mamba env create -f environment.yml
```


### Develop mode

To install MovingPandas in ["develop" or "editable" mode](https://python-packaging-tutorial.readthedocs.io/en/latest/setup_py.html#develop-mode) you may use: 

```
python setup.py develop
```


## Usage

```
import movingml
taxis = movingml.PortoTaxis(r"H:\Geodata\Kaggle\PortoTaxis\train.csv", nrows=100)
taxis.to_df()
taxis.to_gdf()
taxis.to_trajs()


import movingml
cy = movingml.CopenhagenCyclists(r"F:\Documents\GitHub\SimonBreum\desirelines\data\interim\df_bike.pickle", nrows=10)
cy.to_trajs()

import movingml
ais = movingml.BrestAIS(r"H:\Geodata\Zenodo\Integrated Maritime\nari_dynamic_sar.csv")
ais.to_trajs()

import movingml
gulls = movingml.MovebankGulls(r"F:\Documents\GitHub\movingpandas\movingpandas-examples\data\gulls.gpkg")
gulls.to_trajs()

```