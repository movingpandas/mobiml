from mobiml.datasets._dataset import SPEED, TIMESTAMP, Dataset


import pandas as pd


class BrestAIS(Dataset):
    name = "Brest AIS"
    file_name = "nari_dynamic_sar.csv or nari_dynamic.csv"
    source_url = "https://zenodo.org/record/1167595/files/%5BP1%5D%20AIS%20Data.zip?download=1"
    traj_id = 'sourcemmsi'
    mover_id = 'sourcemmsi'
    crs = 4326

    def __init__(self, path, *args, **kwargs) -> None:
        super().__init__(path, *args, **kwargs)
        self.df.rename(columns={
            'ts': 't', 'lon':'x', 'lat': 'y', 'speedoverground': SPEED},
            inplace=True)
        self.df[TIMESTAMP] = pd.to_datetime(self.df['t'], unit='s')
        self.df.drop(columns=['t'], inplace=True)
        print(f"Loaded Dataframe with {len(self.df)} rows.")