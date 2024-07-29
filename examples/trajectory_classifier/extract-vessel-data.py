import os
import sys
import dvc.api
import utils
import warnings
from copy import deepcopy
from datetime import datetime

import sys
sys.path.append("../mobiml")
from mobiml.datasets import AISDK
from mobiml.transforms import MobileClientExtractor

# warnings.filterwarnings('ignore')


def main():
    utils.print_logo()
    print(f"{datetime.now()} Starting data extraction for mobile clients (vessels) ...")

    #params = dvc.api.params_show()
    ship_type = 'Towing'  # params["extract"]["vessels"]
    antenna_radius_meters = 25000  # params["extract"]["vessels_radius_meters"]
    bbox = [57.273, 11.196, 57.998, 12.223]  # params["extract"]["bbox"]
    min_lat, min_lon, max_lat, max_lon = bbox

    out_path = sys.argv[2]
    out_dir = os.path.dirname(out_path)

    path = sys.argv[1]
    if not os.path.exists(out_dir):
        print(f"{datetime.now()} Creating output directory {out_dir} ...")
        os.makedirs(out_dir)

    print(f"{datetime.now()} Loading data from {path}")
    aisdk = AISDK(path, min_lon, min_lat, max_lon, max_lat)
    vessels = deepcopy(
        aisdk
    )  # AISDK(path, min_lon, min_lat, max_lon, max_lat, vessel_type)
    vessels.df = vessels.df[vessels.df.ship_type == ship_type]

    print(f"{datetime.now()} Extracting client data ...")
    client_gdf = MobileClientExtractor(aisdk, vessels, antenna_radius_meters)

    print(f"{datetime.now()} Writing output to {out_path}")
    client_gdf.to_feather(out_path)


if __name__ == "__main__":
    main()
