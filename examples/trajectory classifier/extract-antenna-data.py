import os
import sys 
import dvc.api
import warnings
import utils 
import pandas as pd 
import geopandas as gpd
from datetime import datetime
from mobiml.datasets import AISDK
from mobiml.transforms import StationaryClientExtractor

warnings.filterwarnings('ignore')


def create_client_gdf(clients, client_radius_meters) -> gpd.GeoDataFrame:
    ids =  [{'client': i} for i in range(len(clients))]
    df = pd.DataFrame(ids)
    df['geometry'] = gpd.GeoSeries.from_wkt(clients)
    gdf = gpd.GeoDataFrame(df, geometry=df.geometry, crs=4326)
    gdf = gdf.to_crs(3857)
    gdf['geometry'] = gdf.buffer(client_radius_meters)
    return gdf.to_crs(4326)


def main():
    utils.print_logo()
    print(f"{datetime.now()} Starting data extraction for stationary clients (AIS antennas) ...") 
    
    params = dvc.api.params_show()
    antennas = params['extract']['antennas']
    antenna_radius_meters = params['extract']['antenna_radius_meters']

    buffered_antennas = create_client_gdf(antennas, antenna_radius_meters)
    min_lon, min_lat, max_lon, max_lat = buffered_antennas.geometry.total_bounds

    out_path = sys.argv[2]  
    out_dir = os.path.dirname(out_path)

    path = sys.argv[1]
    if not os.path.exists(out_dir):
        print(f"{datetime.now()} Creating output directory {out_dir} ...")
        os.makedirs(out_dir)

    print(f"{datetime.now()} Loading data from {path}")
    aisdk = AISDK(path, min_lon, min_lat, max_lon, max_lat)

    print(f"{datetime.now()} Extracting client data ...")
    client_gdf = StationaryClientExtractor(aisdk, buffered_antennas) 

    print(f"{datetime.now()} Writing output to {out_path}")
    client_gdf.to_feather(out_path)


if __name__ == "__main__":
    main()


