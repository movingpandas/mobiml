import numpy as np
import geopandas as gpd
import shapely
import math
import warnings

from mobiml.datasets import Dataset


class RandomTrajSampler:

    def __init__(self, data: Dataset) -> None:
        self.data = data

    def random_sample(
        self, n_cells, n_sample=None, percent_sample=None, random_state=None, **kwargs
    ) -> Dataset:
        """
        Randomly samples trajectories in a region of interest.
        Based on: https://github.com/microsoft/torchgeo
        and https://james-brennan.github.io/posts/fast_gridding_geopandas/

        Parameters
        ----------
        n_cells : int | (int, int)
                Desired number of cells per row and column.
                Square layout if only one value provided.
        n_sample : int
                Desired sample number from dataset.
        percent_sample : float
                Desired percentage from dataset for sample.
        random_state : int
                Seed for random number generator.

        Returns
        ----------
        Dataset

        Examples
        ----------
        >>> data = AISDK(r"../examples/data/aisdk_20180208_sample.zip")
        >>> sample = RandomTrajSampler(data).random_sample(n_cells=2, n_sample=100)
        """

        trajs = self.data.to_trajs()
        start = trajs.get_start_locations()
        start.set_crs("EPSG:4326", inplace=True)

        xmin, ymin, xmax, ymax = start.total_bounds
        xmin = xmin - 0.01
        ymin = ymin - 0.01
        xmax = xmax + 0.01
        ymax = ymax + 0.01

        def stride(value):
            if isinstance(value, tuple):
                cell_size_x = (xmax - xmin) / value[0]
                cell_size_y = (ymax - ymin) / value[1]
            elif isinstance(value, int):
                cell_size_x = (xmax - xmin) / value
                cell_size_y = (ymax - ymin) / value
            else:
                print("Please provide a tuple or int.")
            return cell_size_x, cell_size_y

        cell_size_x, cell_size_y = stride(n_cells)

        grid_cells = []
        for x0 in np.arange(xmin, xmax, cell_size_x):
            for y0 in np.arange(ymin, ymax, cell_size_y):
                x1 = x0 + cell_size_x
                y1 = y0 + cell_size_y
                grid_cells.append(shapely.geometry.box(x0, y0, x1, y1))

        cell = gpd.GeoDataFrame(grid_cells, columns=["geometry"], crs="EPSG:4326")
        cell["cell"] = cell.index

        merged = gpd.sjoin(start, cell, how="left", predicate="within")
        merged = merged.drop(columns="index_right")

        filled_cells = merged.cell.unique()
        all_cells = merged.merge(cell, how="outer").cell.unique().tolist()
        diff = [cell for cell in all_cells if cell not in filled_cells]
        if not diff:
            print("All cells can be used for sampling.")
        else:
            warnings.warn(
                "There are empty cells that will not be used for sampling.", UserWarning
            )
            print("Empty cells:", diff)

        print("Number of cells used for sampling:", len(filled_cells))

        def get_cell_sample(
            n_sample=None, percent_sample=None, random_state=None, **kwargs
        ):
            if percent_sample:
                n_sample = percent_sample * len(merged)

            if n_sample > len(start):
                try:
                    raise ValueError("Sample too big.")
                except ValueError:
                    print(
                        "Your sample of",
                        n_sample,
                        "cannot be greater than the dataset:",
                        len(start),
                    )
                    raise

            sample_size = n_sample / len(filled_cells)
            n_sample = math.ceil(sample_size)
            print("Number of samples per cell:", n_sample)

            if n_sample > merged.cell.value_counts().min():
                num = merged["cell"].value_counts()
                count = 0
                for n in num:
                    if n < n_sample:
                        count += 1
                    else:
                        pass
                warnings.warn(
                    "Not enough points in some cells, all points in these cells used for sampling.",
                    UserWarning,
                )
                print("Not enough points in", count, "cell(s).")

            df_sample = merged.groupby(
                ["cell"], as_index=False, group_keys=False
            ).apply(
                lambda x: x.sample(min(n_sample, len(x)), random_state=random_state),
                include_groups=False,
            )
            df_sample["split"] = 2
            df_sample = df_sample[["traj_id", "split"]]
            combined = merged.merge(df_sample, how="left")
            combined.loc[combined["split"] != 2, "split"] = 1
            return combined

        combined = get_cell_sample(n_sample, percent_sample, random_state)

        dataset = Dataset(combined)
        return dataset

    def get_sample_data(
        self, n_cells, n_sample=None, percent_sample=None, random_state=None, **kwargs
    ) -> Dataset:
        data = self.random_sample(
            n_cells, n_sample, percent_sample, random_state, **kwargs
        )

        gdf = data.to_gdf()

        df_sample = gdf.loc[gdf["split"] == 2]
        df_sample = df_sample.drop(columns="split")
        print("Your sample contains", len(df_sample), "records.")

        dataset = Dataset(df_sample)
        return dataset
