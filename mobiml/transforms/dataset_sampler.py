import numpy as np
import geopandas as gpd
import shapely

from mobiml.datasets import Dataset


class RandomTrajSampler:

    def __init__(self, data: Dataset) -> None:
        self.data = data

    def random_sample(self, n_cells, n_sample) -> Dataset:
        """
        Randomly samples trajectories in a region of interest.
        Based on https://github.com/microsoft/torchgeo
        and https://james-brennan.github.io/posts/fast_gridding_geopandas/

        Parameters
        ----------
        n_cells : int | float
                Desired number of cells per row
        n_sample : int
                Desired sample number from each cell

        Returns
        ----------
        Dataset

        Examples
        ----------
        >>> data = AISDK(r"../examples/data/aisdk_20180208_sample.zip")
        >>> random_sample = RandomTrajSampler(data).random_sample(2, 20)
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
            elif isinstance(value, float):
                cell_size_x = (xmax - xmin) / value
                cell_size_y = (ymax - ymin) / value
            else:
                print("Please provide a tuple, int or float.")
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

        def calc_sample_size(n_sample):
            if n_sample < 1:
                n_sample = int(n_sample * len(merged))
                return n_sample
            else:
                n_sample = int(n_sample)
                return n_sample

        n_sample = calc_sample_size(n_sample)

        def get_cell_sample(n_sample):
            if n_sample > merged.cell.value_counts().min():
                print(
                    "Your cell sample of",
                    n_sample,
                    "cannot be greater than the minimum number of points in a cell:",
                    merged.cell.value_counts().min(),
                )
                print("Setting the cell sample to:", merged.cell.value_counts().min())
                n_sample = merged.cell.value_counts().min()
            df_sample = merged.groupby("cell").sample(n=n_sample)
            df_sample["split"] = 2
            df_sample = df_sample[["traj_id", "split"]]
            combined = merged.merge(df_sample, how="left")
            combined.loc[combined["split"] != 2, "split"] = 1
            return combined

        combined = get_cell_sample(n_sample)

        dataset = Dataset(combined)
        return dataset

    def get_sample_data(self, n_cells, n_sample) -> Dataset:
        data = self.random_sample(n_cells, n_sample)

        gdf = data.to_gdf()

        df_sample = gdf.loc[gdf["split"] == 2]
        df_sample = df_sample.drop(columns="split")
        print("Your sample contains", len(df_sample), "records.")

        dataset = Dataset(df_sample)
        return dataset
