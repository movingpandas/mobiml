from datetime import datetime, timedelta

from mobiml.datasets import Dataset, TIMESTAMP, ROWNUM


class CopenhagenCyclists(Dataset):
    name = "Copenhagen Cyclists"
    file_name = "df_bike.pickle"
    source_url = "https://zenodo.org/record/7288616"
    traj_id = "id"
    mover_id = None
    crs = None

    def __init__(self, path, drop_extra_cols=True, *args, **kwargs) -> None:
        super().__init__(path, *args, **kwargs)

        def _compute_datetime(row) -> datetime:
            # some educated guessing going on here:
            # the paper states that the video covers 2021-06-09 07:00-08:00
            t0 = datetime(2021, 6, 9, 7, 0, 0)
            offset = (row["frame_in"] + row[ROWNUM]) * timedelta(seconds=2)
            return t0 + offset

        self.merge_xcol_and_ycol_to_xycol("xs_640x360", "ys_640x360")
        self.explode_coordinate_list()
        self.df[TIMESTAMP] = self.df.apply(_compute_datetime, axis=1)
        self.df.drop(columns=["frame_in"], inplace=True)
        if drop_extra_cols:
            self.df.drop(
                columns=[
                    "frame_out",
                    "num_frames",
                    "time_on_screen_s",
                    "x_start_640x360",
                    "x_end_640x360",
                    "y_start_640x360",
                    "y_end_640x360",
                    "class",
                ],
                inplace=True,
            )
        print(f"Loaded Dataframe with {len(self.df)} rows.")
