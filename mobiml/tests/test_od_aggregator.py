import os
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import Point
from datetime import datetime

from mobiml.datasets import _Dataset
from mobiml.transforms.od_aggregator import ODAggregator


class TestODAggregator:
    test_dir = os.path.dirname(os.path.realpath(__file__))

    def test_spatial_temporal_overlap(self):
        df = pd.DataFrame(
            [
                {
                    "geometry": Point(0, 0),
                    "timestamp": datetime(2018, 1, 1, 12, 0, 0),
                    "traj_id": 1,
                    "mover_id": 1,
                },
                {
                    "geometry": Point(3, 3),
                    "timestamp": datetime(2018, 1, 1, 12, 6, 0),
                    "traj_id": 1,
                    "mover_id": 1,
                },
                {
                    "geometry": Point(0, 0),
                    "timestamp": datetime(2018, 1, 1, 12, 10, 0),
                    "traj_id": 2,
                    "mover_id": 2,
                },
                {
                    "geometry": Point(3, 3),
                    "timestamp": datetime(2018, 1, 1, 12, 15, 0),
                    "traj_id": 2,
                    "mover_id": 2,
                },
            ]
        )
        self.gdf = GeoDataFrame(df, crs=4326)

        dataset = _Dataset(self.gdf)
        assert len(dataset.to_trajs()) == 2
        aggregator = ODAggregator(dataset)
        assert isinstance(aggregator, ODAggregator)
        od = aggregator.get_od_for_h3(7, "1D")
        assert od.origin.sum() == 2
        assert len(od.t.unique()) == 1
        assert len(od.h3_cell.unique()) == 1
        assert od.t[0] == datetime(2018, 1, 1, 0, 0, 0)
        assert od.h3_cell[0] == "87754e64dffffff"

    def test_temporal_overlap(self):
        df = pd.DataFrame(
            [
                {
                    "geometry": Point(0, 0),
                    "timestamp": datetime(2018, 1, 1, 12, 0, 0),
                    "traj_id": 1,
                    "mover_id": 1,
                },
                {
                    "geometry": Point(3, 3),
                    "timestamp": datetime(2018, 1, 1, 12, 6, 0),
                    "traj_id": 1,
                    "mover_id": 1,
                },
                {
                    "geometry": Point(6, 6),
                    "timestamp": datetime(2018, 1, 1, 12, 10, 0),
                    "traj_id": 2,
                    "mover_id": 2,
                },
                {
                    "geometry": Point(9, 9),
                    "timestamp": datetime(2018, 1, 1, 12, 15, 0),
                    "traj_id": 2,
                    "mover_id": 2,
                },
            ]
        )
        self.gdf = GeoDataFrame(df, crs=4326)

        dataset = _Dataset(self.gdf)
        assert len(dataset.to_trajs()) == 2
        aggregator = ODAggregator(dataset)
        assert isinstance(aggregator, ODAggregator)
        od = aggregator.get_od_for_h3(7, "1D")
        assert od.origin.sum() == 2
        assert len(od.t.unique()) == 1
        assert len(od.h3_cell.unique()) == 2
        assert od.t[0] == datetime(2018, 1, 1, 0, 0, 0)
        h3_cell = ["87588a210ffffff", "87754e64dffffff"]
        assert od.h3_cell.tolist() == h3_cell

    def test_spatial_overlap(self):
        df = pd.DataFrame(
            [
                {
                    "geometry": Point(0, 0),
                    "timestamp": datetime(2018, 1, 1, 12, 0, 0),
                    "traj_id": 1,
                    "mover_id": 1,
                },
                {
                    "geometry": Point(3, 3),
                    "timestamp": datetime(2018, 1, 1, 12, 6, 0),
                    "traj_id": 1,
                    "mover_id": 1,
                },
                {
                    "geometry": Point(0, 0),
                    "timestamp": datetime(2018, 1, 3, 12, 10, 0),
                    "traj_id": 2,
                    "mover_id": 2,
                },
                {
                    "geometry": Point(3, 3),
                    "timestamp": datetime(2018, 1, 3, 12, 15, 0),
                    "traj_id": 2,
                    "mover_id": 2,
                },
            ]
        )
        self.gdf = GeoDataFrame(df, crs=4326)

        dataset = _Dataset(self.gdf)
        assert len(dataset.to_trajs()) == 2
        aggregator = ODAggregator(dataset)
        assert isinstance(aggregator, ODAggregator)
        od = aggregator.get_od_for_h3(7, "1D")
        assert od.origin.sum() == 2
        assert len(od.t.unique()) == 2
        assert len(od.h3_cell.unique()) == 1
        t = [datetime(2018, 1, 1, 0, 0, 0), datetime(2018, 1, 3, 0, 0, 0)]
        assert od.t.tolist() == t
        assert od.h3_cell[0] == "87754e64dffffff"

    def test_no_overlap(self):
        df = pd.DataFrame(
            [
                {
                    "geometry": Point(0, 0),
                    "timestamp": datetime(2018, 1, 1, 12, 0, 0),
                    "traj_id": 1,
                    "mover_id": 1,
                },
                {
                    "geometry": Point(3, 3),
                    "timestamp": datetime(2018, 1, 1, 12, 6, 0),
                    "traj_id": 1,
                    "mover_id": 1,
                },
                {
                    "geometry": Point(6, 6),
                    "timestamp": datetime(2018, 1, 3, 12, 10, 0),
                    "traj_id": 2,
                    "mover_id": 2,
                },
                {
                    "geometry": Point(9, 9),
                    "timestamp": datetime(2018, 1, 3, 12, 15, 0),
                    "traj_id": 2,
                    "mover_id": 2,
                },
            ]
        )
        self.gdf = GeoDataFrame(df, crs=4326)

        dataset = _Dataset(self.gdf)
        assert len(dataset.to_trajs()) == 2
        aggregator = ODAggregator(dataset)
        assert isinstance(aggregator, ODAggregator)
        od = aggregator.get_od_for_h3(7, "1D")
        assert od.origin.sum() == 2
        assert len(od.t.unique()) == 2
        assert len(od.h3_cell.unique()) == 2
        t = [datetime(2018, 1, 1, 0, 0, 0), datetime(2018, 1, 3, 0, 0, 0)]
        assert od.t.tolist() == t
        h3_cell = ["87754e64dffffff", "87588a210ffffff"]
        assert od.h3_cell.tolist() == h3_cell
