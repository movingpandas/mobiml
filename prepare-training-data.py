import os
import sys 
import pickle
import dvc.api
import pandas as pd
import geopandas as gpd
import movingpandas as mpd
from datetime import datetime, timedelta
from utils import create_dir_if_not_exists
from mobiml.transformers import AISTripExtractor, TrajectoryAggregator
from mobiml.datasets._dataset import MOVER_ID

import warnings
warnings.filterwarnings('ignore')


def create_vessel_list(gdf):
    print(f"{datetime.now()} Creating vessel list ...")
    return gdf.groupby(MOVER_ID)[['ship_type','Name']].agg(pd.Series.mode)


def pickle_vessels(vessels, out_path):
    out_path = out_path.replace('training-data', 'vessels')
    print(f"{datetime.now()} Writing vessels to {out_path} ...")
    with open(out_path, 'wb') as out_file:
        pickle.dump(vessels, out_file)


def pickle_trajectories(trajs, out_path):
    print(f"{datetime.now()} Writing trajectories to {out_path} ...")
    with open(out_path, 'wb') as out_file:
        pickle.dump(trajs, out_file)


def main():
    print("Hello MobiSpacers!") 
    print("Let's turn raw AIS into trajectories for training.")

    path = sys.argv[1]
    out_path = sys.argv[2]  # 'data/prepared/trajs.pickle'
    create_dir_if_not_exists(out_path)

    params = dvc.api.params_show()
    h3_resolution = params['traj_feature_engineering']['h3_resolution']

    print(f"{datetime.now()} Loading data from {path} ...")
    gdf =  gpd.read_feather(path)
    vessels = create_vessel_list(gdf)

    print(f"{datetime.now()} Extracting trips ...")
    trajs = AISTripExtractor(gdf).get_trips(stop_duration=timedelta(minutes=60))  # create_trajs(gdf)

    print(f"{datetime.now()} Computing trajectory features ...")
    trajs = TrajectoryAggregator(trajs, vessels).aggregate_trajs(h3_resolution)    

    pickle_trajectories(trajs, out_path)
    pickle_vessels(vessels, out_path)


if __name__ == "__main__":
    main()

