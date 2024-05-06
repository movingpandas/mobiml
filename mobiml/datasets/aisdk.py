import pandas as pd
from datetime import datetime
from zipfile import ZipFile

from mobiml.datasets._dataset import Dataset, SPEED, TIMESTAMP, MOVER_ID, DIRECTION

SHIPTYPE = 'ship_type'

class AISDK(Dataset):
    name = "Danish AIS (AISDK)"
    file_name = "aisdk-2018-02.zip"
    source_url = "http://web.ais.dk/aisdata/aisdk-2018-02.zip"
    traj_id = 'MMSI'
    mover_id = 'MMSI'
    crs = 4326

    COLS = ['# Timestamp','Latitude','Longitude','MMSI','Navigational status','SOG','COG','Name','Ship type']
    TIME_FORMAT = "%d/%m/%Y %H:%M:%S"

    def __init__(self, path, min_lon=None, min_lat=None, max_lon=None, max_lat=None, *args, **kwargs) -> None:
        self.min_lon = min_lon
        self.min_lat = min_lat
        self.max_lon = max_lon
        self.max_lat = max_lat
        super().__init__(path, *args, **kwargs)
        if (self.min_lat is not None) & (self.max_lat is not None) & (self.min_lon is not None) & (self.max_lon is not None):
            self.df = self.df[(self.df.Latitude >= self.min_lat) & (self.df.Latitude <= self.max_lat) &
                            (self.df.Longitude >= self.min_lon) & (self.df.Longitude <= self.max_lon)]
        self.df = self.df[self.df.SOG>0]
        self.df.rename(columns={
            '# Timestamp': 't', 'Longitude':'x', 'Latitude': 'y', 'SOG': SPEED, 'COG': DIRECTION, 'Navigational status':'nav_status', 'Ship type': SHIPTYPE},
            inplace=True)
        self.df[TIMESTAMP] = pd.to_datetime(self.df['t'], format=self.TIME_FORMAT)
        self.df.drop(columns=['t'], inplace=True)
        print(f"{datetime.now()} Loaded Dataframe with {len(self.df)} rows.")

    def load_df_from_zip_archive(self, path) -> pd.DataFrame:
        """Load the AIS records from all CSVs in the provided ZIP archive path"""

        def load_single_csv(csv_name) -> pd.DataFrame:
            print(f"{datetime.now()} Loading {csv_name} ...")
            tmp_df = pd.read_csv(ZipFile(path).open(csv_name), usecols=self.COLS) 
            if (self.min_lat is not None) & (self.max_lat is not None) & (self.min_lon is not None) & (self.max_lon is not None):
                tmp_df = tmp_df[(tmp_df.Latitude >= self.min_lat) & (tmp_df.Latitude <= self.max_lat) &
                                (tmp_df.Longitude >= self.min_lon) & (tmp_df.Longitude <= self.max_lon)]
            tmp_df = tmp_df[tmp_df.SOG>0]
            return tmp_df
            
        df = pd.concat(
            [load_single_csv(csv_name) for csv_name in ZipFile(path).namelist()],
            ignore_index=True
        )
        return df


class PreprocessedAISDK(Dataset):
    name = "Danish AIS (AISDK)"
    file_name = "ais-antenna.feather"
    crs = 4326

    def __init__(self, path, *args, **kwargs) -> None:
        super().__init__(path, *args, **kwargs)    
