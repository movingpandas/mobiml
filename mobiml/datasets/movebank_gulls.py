from mobiml.datasets import Dataset, SPEED, TIMESTAMP, MOVER_ID


class MovebankGulls(Dataset):
    name = "Movebank Migrating Gulls"
    file_name = "gulls.gpkg"
    source_url = "https://github.com/movingpandas/movingpandas-examples/blob/main/data/gulls.gpkg"
    traj_id = "individual-local-identifier"
    mover_id = "individual-local-identifier"
    crs = 4326

    def __init__(self, path, drop_extra_cols=True, *args, **kwargs) -> None:
        super().__init__(path, *args, **kwargs)
        self.df.rename(columns={"timestamp": TIMESTAMP})
        if drop_extra_cols:
            self.df.drop(
                columns=[
                    "individual-taxon-canonical-name",
                    "study-name",
                    "location-long",
                    "location-lat",
                    "event-id",
                    "visible",
                ],
                inplace=True,
            )
        print(f"Loaded Dataframe with {len(self.df)} rows.")
