import pandas as pd
import geopandas as gpd
import movingpandas as mpd
from datetime import datetime, timedelta
from multiprocessing import Pool
from itertools import repeat, cycle
from pymeos import pymeos_initialize, TGeogPointInst, TGeogPointSeq
from mobiml.datasets import AISDK
from mobiml.datasets._dataset import SPEED, TIMESTAMP, MOVER_ID


class MobileClientExtractor():
    def __init__(self, dataset: AISDK, clients: AISDK, antenna_radius_meters, n_threads=4) -> None:
        pymeos_initialize()  # Don't remove. Necessary for the correct functioning of PyMEOS
        self.n_threads = n_threads
        self.antenna_radius_meters = antenna_radius_meters
        
        print(f"{datetime.now()} Creating PyMEOS points ...")  
        ais = dataset.to_gdf()
        ais['pymeos_pt'] = ais.apply(lambda row: 
                                     TGeogPointInst(point=row['geometry'], timestamp=row[TIMESTAMP]), axis=1) 
        
        print(f"{datetime.now()} Creating client trajectories ...")  
        mpd_tc = clients.to_trajs()
        client_trajs = self.create_pymeos_trajectories(mpd_tc)

        print(f"{datetime.now()} Calculating spatiotemporal intersections ...")  
        
        results = []
        n = len(client_trajs)
        i = 1
        for traj_id, traj in client_trajs.items():
            print(f"{i}/{n}: {traj_id}")
            mmsi = int(traj_id.split('_')[0])
            tmp = ais.copy()
            tmp['dist'] = tmp.apply(lambda row: self.distance_to_traj(row, traj, mmsi), axis=1)
            tmp = tmp[(tmp.dist > 0) & (tmp.dist < self.antenna_radius_meters)]
            tmp['client'] = mmsi
            results.append(tmp)
            i = i+1

        self.gdf = pd.concat(results)
    
    def create_pymeos_trajectories(self, tc):
        print(f"{datetime.now()} Splitting trajectories ...")
        tc = mpd.ObservationGapSplitter(tc).split(gap=timedelta(minutes=60))
        print(f"{datetime.now()} Creating PyMEOS trajectories ...")
        pymeos_traj = {}
        for traj in tc.trajectories:
            wkt = self.extract_wkt_from_traj_vectorized(traj)
            pymeos_traj[traj.id] = TGeogPointSeq(string=wkt, normalize=False)
        return pymeos_traj

    def extract_wkt_from_traj_vectorized(self, traj):
        series = traj.df.geometry.to_wkt()+'@'+traj.df.geometry.index.astype(str)
        return str(list(series)).replace("'","")
    
    def distance_to_traj(self, row, traj, mmsi):
        tpt = row['pymeos_pt']
        d = traj.distance(tpt)
        if d:  # and row.MMSI != mmsi:
            return d.value()
        return None

    def to_feather(self, out_path) -> None:
        tmp = self.gdf.copy()
        tmp.drop(columns=['pymeos_pt'], inplace=True)
        tmp.to_feather(out_path)