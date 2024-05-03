import os
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from geopandas import GeoDataFrame
from movingpandas import TrajectoryCollection
from shapely.geometry import Point, LineString
from datetime import datetime, timedelta
from fiona.crs import from_epsg

from mobiml.datasets._dataset import Dataset, TRAJ_ID, MOVER_ID


class TestDataset:
    test_dir = os.path.dirname(os.path.realpath(__file__))

    def setup_method(self):
        df = pd.DataFrame([
            {'geometry':Point(0,0), 'txx':datetime(2018,1,1,12,0,0), 'tid':1, 'mid':'a'},
            {'geometry':Point(6,0), 'txx':datetime(2018,1,1,12,6,0), 'tid':1, 'mid':'a'},
            {'geometry':Point(6,6), 'txx':datetime(2018,1,1,12,10,0), 'tid':1, 'mid':'a'},
            {'geometry':Point(9,9), 'txx':datetime(2018,1,1,12,15,0), 'tid':1, 'mid':'a'}
            ]).set_index('txx')
        self.gdf = GeoDataFrame(df, crs=31256)
        
    def test_dataset_from_gdf(self):    
        data = Dataset(self.gdf, name='test', traj_id='tid', mover_id='mid')
        assert isinstance(data, Dataset)
        assert data.name == 'test'
        assert data.traj_id == 'tid'
        assert data.mover_id == 'mid'
        assert TRAJ_ID in data.df.columns
        assert MOVER_ID in data.df.columns
        trajs = data.to_trajs()
        assert isinstance(trajs, TrajectoryCollection)

    def test_dataset_from_csv(self):    
        path = os.path.join(self.test_dir,'data/test.csv')
        data = Dataset(path, name='test', traj_id='tid', mover_id='mid', timestamp='t', crs=31256)
        assert isinstance(data, Dataset)
        assert data.name == 'test'
        assert data.traj_id == 'tid'
        assert data.mover_id == 'mid'
        assert TRAJ_ID in data.df.columns
        assert MOVER_ID in data.df.columns
        trajs = data.to_trajs()
        assert isinstance(trajs, TrajectoryCollection)

    def test_dataset_from_zipped_csv(self):    
        path = os.path.join(self.test_dir,'data/test.zip')
        data = Dataset(path, name='test', traj_id='tid', mover_id='mid', timestamp='t', crs=31256)
        assert isinstance(data, Dataset)
        assert data.name == 'test'
        assert data.traj_id == 'tid'
        assert data.mover_id == 'mid'
        assert TRAJ_ID in data.df.columns
        assert MOVER_ID in data.df.columns
        trajs = data.to_trajs()
        assert isinstance(trajs, TrajectoryCollection)