import warnings
import os
import dvc.api
from datetime import datetime

warnings.filterwarnings("ignore")


def create_dir_if_not_exists(out_path):
    out_dir = os.path.dirname(out_path)
    if not os.path.exists(out_dir):
        print(f"{datetime.now()} Creating output directory {out_dir} ...")
        os.makedirs(out_dir)


def print_logo():
    print(
        f"""
         %%%%%%%%%%%%
      %%%%%           ###
    %%%%               ##
   %%%                             %%%      %%%            %%        %%        
  %%%      ###################     %%%%%   %%%%            %%             %%%%%              %%%%%%             %%%%%     %%%%%
                                   %%%%%%%%%%%%    %%%%    %%%%%%    %%  %%        %%%%%%         %%   %%%%%%  %%    %%  %%    
 ###################      %%%      %%%  %%  %%%  %%    %%  %%    %%  %%    %%%%   %%    %%   %%%%%%%  %%       %%%%%%%%    %%%%
                         %%%       %%%      %%%  %%    %%  %%    %%  %%       %%  %%    %%  %%%   %%  %%       %%             %%
     ##                %%%         %%%      %%%    %%%%    %%%%%%%   %%  %%%%%%   %%%%%%%    %%%%%%%   %%%%%%   %%%%%%%  %%%%%%
     ###           %%%%%%                                                         %% 
         %%%%%%%%%%%%%                                                            %% 
    """
    )


def get_dvc_params():
    dvc_params = dvc.api.params_show()
    vessel_types = dvc_params["base_model"]["vessel_types"]  # AIS has 10 classes
    n_features = dvc_params["base_model"]["n_features"]  # Number of features in dataset
    traj_features = dvc_params["base_model"]["traj_features"]
    test_size = dvc_params["base_model"]["test_size"]
    return vessel_types, n_features, traj_features, test_size
