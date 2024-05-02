# mobiml


## Development installation 

```
mamba env create -f environment.yml
```


## Usage

### Porto Taxi FCD

```
import mobiml
taxis = mobiml.PortoTaxis(r"H:\Geodata\Kaggle\PortoTaxis\train.csv", nrows=100)
taxis.to_df()
taxis.to_gdf()
taxis.to_trajs()
```

### Copenhagen Cyclists Desirelines 

```
cy = mobiml.CopenhagenCyclists(r"F:\Documents\GitHub\SimonBreum\desirelines\data\interim\df_bike.pickle", nrows=10)
cy.to_trajs()
```

### Brest Vessels AIS

```
ais = mobiml.BrestAIS(r"H:\Geodata\Zenodo\Integrated Maritime\nari_dynamic_sar.csv")
ais.to_trajs()
ex = mobiml.AISTripExtractor(ais)
ais_trips = ex.get_trips()
```

### Movebank Migrating Gulls

```
gulls = mobiml.MovebankGulls(r"F:\Documents\GitHub\movingpandas\movingpandas-examples\data\gulls.gpkg")
gulls.to_trajs()

```