import os 
import pandas as pd

from mobiml.datasets._dataset import Dataset, SPEED, TIMESTAMP, MOVER_ID, DIRECTION


class BrestAIS(Dataset):
    name = "Brest AIS"
    file_name = "nari_dynamic_sar.csv or nari_dynamic.csv"
    source_url = "https://zenodo.org/record/1167595/files/%5BP1%5D%20AIS%20Data.zip?download=1"
    traj_id = 'sourcemmsi'
    mover_id = 'sourcemmsi'
    crs = 4326

    def __init__(self, path, *args, **kwargs) -> None:
        apply_mid_filter = kwargs.pop('filter_invalid_mmsis', False)

        super().__init__(path, *args, **kwargs)
        self.df.rename(columns={
            'ts': 't', 'lon':'x', 'lat': 'y', 'speedoverground': SPEED, 'courseoverground': DIRECTION},
            inplace=True)
        self.df[TIMESTAMP] = pd.to_datetime(self.df['t'], unit='s')
        self.df.drop(columns=['t'], inplace=True)

        if apply_mid_filter:
            self.filter_by_mid()

        print(f"Loaded Dataframe with {len(self.df)} rows.")

    def filter_by_mid(self):
        wd = os.path.dirname(os.path.realpath(__file__))
        mid_whitelist = pd.read_csv(os.path.join(wd,'ais_mid_whitelist.csv'))
        self.df = self.df.loc[
            self.df[MOVER_ID].astype(str).str.zfill(9).str[:3].isin(
                mid_whitelist.MID.astype(str)
            )
        ].copy()


class PreprocessedBrestAIS(Dataset):
    name = "Brest AIS"
    file_name = "preprocessed.csv"
    crs = 4326

    def __init__(self, path, *args, **kwargs) -> None:
        super().__init__(path, *args, **kwargs)    
        self.df[TIMESTAMP] = pd.to_datetime(self.df[TIMESTAMP])
        