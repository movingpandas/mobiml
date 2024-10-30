import numpy as np
import geopandas as gpd
import shapely
import math
import warnings

from mobiml.datasets import Dataset, TRAJ_ID


class RandomTrajSampler:

    def __init__(self, data: Dataset) -> None:
        self.data = data

    def split(
        self, n_cells, n_sample=None, percent_sample=None, random_state=None
    ) -> Dataset:
        """
        Randomly samples trajectories, targetting an equal spatial distribution based
        on user-defined grid (i.e. equal number of trajecties per cell, based on traj start points)
        
        Inspired by: https://github.com/microsoft/torchgeo
        and https://james-brennan.github.io/posts/fast_gridding_geopandas/

        Parameters
        ----------
        n_cells : int | (int, int)
                Desired number of cells per row and column.
                Square layout if only one value is provided.
        n_sample : int
                Desired number of trajectories to sample.
        percent_sample : float
                Desired percentage of trajectories to sample.
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
        start_pts = trajs.get_start_locations()
        start_pts.set_crs(trajs.get_crs(), inplace=True)

        xmin, ymin, xmax, ymax = start_pts.total_bounds
        grid = self._create_grid(n_cells, xmin, ymin, xmax, ymax, trajs.get_crs())

        joined = gpd.sjoin(start_pts, grid, how="left", predicate="within")
        joined = joined.drop(columns="index_right")

        filled_cells = joined.cell.unique()
        all_cells = joined.merge(grid, how="outer").cell.unique().tolist()
        diff = [cell for cell in all_cells if cell not in filled_cells]

        if not diff:
            print("All cells can be used for sampling.")
        else:
            warnings.warn(
                "There are empty cells that will not be used for sampling.", UserWarning
            )
            print(f"Empty cells: {diff}")

        print(f"Number of cells used for sampling: {len(filled_cells)}")

        if percent_sample:
            n_sample = math.ceil(percent_sample * len(joined))

        sampled = self._sample_trajs(joined, filled_cells, n_sample, random_state)
        sampled = sampled[[TRAJ_ID, 'split']]
        sampled.set_index(TRAJ_ID, inplace=True)

        if len(sampled[sampled.split==2]) > n_sample:
            sampled = sampled.sample(n_sample, random_state=random_state)
        
        result = self.data.to_gdf().copy()
        result = result.join(sampled, on=TRAJ_ID, rsuffix='r')
        result['split'] = result['split'].fillna(1)
        return Dataset(result)

    def sample(
        self, n_cells, n_sample=None, percent_sample=None, random_state=None
    ) -> Dataset:
        data = self.split(
            n_cells, n_sample, percent_sample, random_state
        )

        gdf = data.to_gdf()

        df_sample = gdf.loc[gdf["split"] == 2]
        df_sample = df_sample.drop(columns="split")
        print(f"Your sample contains {len(df_sample)} records.")

        dataset = Dataset(df_sample)
        return dataset

    def _create_grid(self, n_cells, xmin, ymin, xmax, ymax, crs, buffer=0.1):
        xmin -= buffer
        ymin -= buffer
        xmax += buffer
        ymax += buffer      
        cell_size_x, cell_size_y = self._get_cell_size(n_cells, xmin, xmax, ymin, ymax)

        grid_cells = []
        for x0 in np.arange(xmin, xmax, cell_size_x):
            for y0 in np.arange(ymin, ymax, cell_size_y):
                x1 = x0 + cell_size_x
                y1 = y0 + cell_size_y
                grid_cells.append(shapely.geometry.box(x0, y0, x1, y1))

        grid = gpd.GeoDataFrame(grid_cells, columns=["geometry"], crs=crs)
        grid["cell"] = grid.index
        return grid

    def _sample_trajs(self, merged, filled_cells, n_sample, random_state=None):
        if n_sample > len(merged):
            try:
                raise ValueError("Sample too big.")
            except ValueError:
                print(
                    f"Your sample of {n_sample} ",
                    f"cannot be greater than the dataset: {len(merged)}"
                )
                raise

        n_per_cell = math.ceil(n_sample / len(filled_cells))
        print("Number of samples per cell:", n_per_cell)

        if n_per_cell > merged.cell.value_counts().min():
            num = merged["cell"].value_counts()
            count = 0
            for n in num:
                if n < n_per_cell:
                    count += 1
                else:
                    pass
            warnings.warn(
                "Not enough points in some cells.",
                UserWarning,
            )
            print(
                f"Not enough points in {count} cell(s). All points used in these cells.",
            )

        df_sample = merged.groupby(
            ["cell"], as_index=False, group_keys=False
        ).apply(
            lambda x: x.sample(min(n_per_cell, len(x)), random_state=random_state),
            include_groups=False,
        )
        df_sample["split"] = 2
        df_sample = df_sample[["traj_id", "split"]]
        combined = merged.merge(df_sample, how="left")
        combined.loc[combined["split"] != 2, "split"] = 1
        return combined
    
    def _get_cell_size(self, value, xmin, xmax, ymin, ymax):
        if isinstance(value, tuple):
            cell_size_x = (xmax - xmin) / value[0]
            cell_size_y = (ymax - ymin) / value[1]
        elif isinstance(value, int):
            cell_size_x = (xmax - xmin) / value
            cell_size_y = (ymax - ymin) / value
        else:
            raise(ValueError("Please provide the number of cells as int or (int, int)."))
        return cell_size_x, cell_size_y