from mobiml.datasets import _Dataset, TIMESTAMP


class DelhiAirPollution(_Dataset):
    name = "Delhi Air Pollution"
    file_name = "2021-01-30_all.csv"
    source_url = "http://cse.iitd.ac.in/pollutiondata/delhi/download"
    traj_id = "deviceId"
    mover_id = "deviceId"
    crs = 4326

    def __init__(self, path, *args, **kwargs) -> None:
        super().__init__(path, *args, **kwargs)

        self.df.rename(
            columns={
                "dateTime": TIMESTAMP,
                "long": "x",
                "lat": "y",
            },
            inplace=True,
        )
        self.df.drop(columns=["Unnamed: 0"], inplace=True)

        print(f"Loaded Dataframe with {len(self.df)} rows.")
