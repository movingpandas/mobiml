from mobiml.datasets import Dataset, SPEED, DIRECTION


class Normalizer:
    def __init__(self, data: Dataset) -> None:
        self.data = data

    def normalize(self, speed_max=None, replace=False) -> Dataset:
        """
        Normalizes latitude, longitude, speed, direction values in dataset.

        Inspired by: https://github.com/CIA-Oceanix/GeoTrackNet

        Parameters
        ----------
        speed_max : int | float
                Desired maximum speed.
        replace : boolean
                When ``replace=False`` (default) writes output to new columns.
                When ``replace=True`` overwrites values in columns.

        Returns
        ----------
        Dataset

        Examples
        ----------
        >>> ais = BrestAIS(r"../examples/data/nari_dynamic.csv", nrows=1000)
        >>> ais = Normalizer(ais).normalize(speed_max=5.0, replace=True)
        """
        LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = self.data.get_bounds()
        data = self.data.to_df()

        if replace is False:
            x_col_name = "norm_x"
            y_col_name = "norm_y"
            speed_col_name = "norm_speed"
            direction_col_name = "norm_direction"
        else:
            x_col_name = "x"
            y_col_name = "y"
            speed_col_name = SPEED
            direction_col_name = DIRECTION

        data[x_col_name] = data["x"].apply(
            lambda x: (x - LON_MIN) / (LON_MAX - LON_MIN)
        )
        data[y_col_name] = data["y"].apply(
            lambda x: (x - LAT_MIN) / (LAT_MAX - LAT_MIN)
        )

        if SPEED in data.columns:
            if speed_max is None:
                speed_max = data[SPEED].max()
            else:
                pass
            SPEED_MAX = speed_max
            data.loc[data[SPEED] > SPEED_MAX, speed_col_name] = SPEED_MAX
            data[speed_col_name] = data[SPEED] / SPEED_MAX
        else:
            pass

        if DIRECTION in data.columns:
            data[direction_col_name] = data[DIRECTION] / 360.0
        else:
            pass

        return Dataset(data)
