from datetime import timedelta

from mobiml.datasets import (
    Dataset,
    TIMESTAMP,
    COORDS,
    ROWNUM,
    unixtime_to_datetime,
)


class PortoTaxis(Dataset):
    name = "Porto Taxi"
    file_name = "train.csv or test.csv"
    source_url = "https://www.kaggle.com/competitions/pkdd-15-predict-taxi-service-trajectory-i/data"  # noqa E501
    traj_id = "TRIP_ID"
    mover_id = "TAXI_ID"
    crs = 4326

    def __init__(self, path, *args, **kwargs) -> None:
        super().__init__(path, *args, **kwargs)

        def _compute_datetime(row):
            t0 = unixtime_to_datetime(row["TIMESTAMP"])
            offset = row[ROWNUM] * timedelta(seconds=15)
            return t0 + offset

        self.df.POLYLINE = self.df.POLYLINE.apply(eval)  # string to list
        self.df.rename(columns={"POLYLINE": COORDS}, inplace=True)
        self.explode_coordinate_list()
        self.df[TIMESTAMP] = self.df.apply(_compute_datetime, axis=1)
        self.df.drop(columns=["TIMESTAMP"], inplace=True)
        print(f"Loaded Dataframe with {len(self.df)} rows.")
