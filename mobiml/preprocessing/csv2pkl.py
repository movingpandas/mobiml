# coding: utf-8

# MIT License
#
# Copyright (c) 2018 Duong Nguyen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

"""
A script to merge AIS messages into AIS tracks.
"""
import numpy as np
import os
import pickle
from datetime import datetime
from tqdm import tqdm
import pandas as pd

# PARAMS
# ======================================
# AISDK dataset
LAT_MIN = 57.0
LAT_MAX = 58.0
LON_MIN = 11.0
LON_MAX = 13.0

# D2C_MIN = 2000 #meters

# Pkl filenames
pkl_filename = "aisdk_20180208.pkl"
pkl_filename_train = "aisdk_20180208_train.pkl"
pkl_filename_valid = "aisdk_20180208_valid.pkl"
pkl_filename_test = "aisdk_20180208_test.pkl"

# Path to csv files
dataset_path = "../../examples/data/"
l_csv_filename = ["aisdk_20180208_sample.csv"]

# Training/validation/test/total period
EPOCH = datetime(1970, 1, 1)
t_train_min = (
    (pd.to_datetime("08/02/2018 00:00:00", format="%d/%m/%Y %H:%M:%S"))
    - pd.Timestamp(EPOCH)
) // pd.Timedelta("1s")
t_train_max = (
    (pd.to_datetime("08/02/2018 07:59:59", format="%d/%m/%Y %H:%M:%S"))
    - pd.Timestamp(EPOCH)
) // pd.Timedelta("1s")
t_valid_min = (
    (pd.to_datetime("08/02/2018 08:00:00", format="%d/%m/%Y %H:%M:%S"))
    - pd.Timestamp(EPOCH)
) // pd.Timedelta("1s")
t_valid_max = (
    (pd.to_datetime("08/02/2018 15:59:59", format="%d/%m/%Y %H:%M:%S"))
    - pd.Timestamp(EPOCH)
) // pd.Timedelta("1s")
t_test_min = (
    (pd.to_datetime("08/02/2018 16:00:00", format="%d/%m/%Y %H:%M:%S"))
    - pd.Timestamp(EPOCH)
) // pd.Timedelta("1s")
t_test_max = (
    (pd.to_datetime("08/02/2018 23:59:59", format="%d/%m/%Y %H:%M:%S"))
    - pd.Timestamp(EPOCH)
) // pd.Timedelta("1s")
t_min = (
    (pd.to_datetime("08/02/2018 00:00:00", format="%d/%m/%Y %H:%M:%S"))
    - pd.Timestamp(EPOCH)
) // pd.Timedelta("1s")
t_max = (
    (pd.to_datetime("08/02/2018 23:59:59", format="%d/%m/%Y %H:%M:%S"))
    - pd.Timestamp(EPOCH)
) // pd.Timedelta("1s")

# Output path
out_path = "../../examples/data/"

# ======================================
LAT_RANGE = LAT_MAX - LAT_MIN
LON_RANGE = LON_MAX - LON_MIN
SOG_MAX = 30.0  # SOG is truncated to 30.0 knots max

TIMESTAMP, MMSI, LAT, LON, NAV_STT, ROT, SOG, COG, HEADING, SHIPTYPE = list(range(10))
cols = [
    "Latitude",
    "Longitude",
    "SOG",
    "COG",
    "Heading",
    "ROT",
    "Navigational status",
    "# Timestamp",
    "MMSI",
    "Ship type",
]

CARGO_TANKER_ONLY = False
if CARGO_TANKER_ONLY:
    pkl_filename = "ct_" + pkl_filename
    pkl_filename_train = "ct_" + pkl_filename_train
    pkl_filename_valid = "ct_" + pkl_filename_valid
    pkl_filename_test = "ct_" + pkl_filename_test

cargo_tanker_filename = "aisdk_20180208_cargo_tanker.npy"
TYPE = "Fishing"


# LOADING CSV FILES
# ======================================
l_l_msg = []  # list of AIS messages, each row is a message (list of AIS attributes)
for csv_filename in l_csv_filename:
    data_path = os.path.join(dataset_path, csv_filename)
    df = pd.read_csv(data_path, header=0, usecols=cols)
dates = pd.to_datetime(df["# Timestamp"], format="%d/%m/%Y %H:%M:%S")
df["# Timestamp"] = (dates - pd.Timestamp(EPOCH)) // pd.Timedelta("1s")
df = df.drop(df[df["Ship type"] == "Undefined"].index)
df = df.reset_index()
df = df.drop("index", axis=1)
df = df.dropna()
l_l_msg = df
m_msg = np.array(l_l_msg)

print("Total number of AIS messages: ", m_msg.shape[0])
print("Lat min: ", np.min(m_msg[:, LAT]), "Lat max: ", np.max(m_msg[:, LAT]))
print("Lon min: ", np.min(m_msg[:, LON]), "Lon max: ", np.max(m_msg[:, LON]))
print("Ts min: ", np.min(m_msg[:, TIMESTAMP]), "Ts max: ", np.max(m_msg[:, TIMESTAMP]))
print("Time min: ", pd.to_datetime((np.min(m_msg[:, TIMESTAMP])), unit="s"))
print("Time max: ", pd.to_datetime((np.max(m_msg[:, TIMESTAMP])), unit="s"))

# Vessel Type
# ======================================
print("Selecting vessel type ...")


def sublist(lst1, lst2):
    ls1 = [element for element in lst1 if element in lst2]
    ls2 = [element for element in lst2 if element in lst1]
    return (len(ls1) != 0) and (ls1 == ls2)


VesselTypes = dict()
l_mmsi = []
n_error = 0
for v_msg in tqdm(m_msg):
    try:
        mmsi_ = v_msg[MMSI]
        type_ = v_msg[SHIPTYPE]
        if mmsi_ not in l_mmsi:
            VesselTypes[mmsi_] = [type_]
            l_mmsi.append(mmsi_)
        elif type_ not in VesselTypes[mmsi_]:
            VesselTypes[mmsi_].append(type_)
    except Exception as e:
        n_error += 1
        continue
# print(n_error)

for mmsi_ in tqdm(list(VesselTypes.keys())):
    VesselTypes[mmsi_] = np.sort(VesselTypes[mmsi_])

l_cargo_tanker = []
l_shiptype = []
for mmsi_ in list(VesselTypes.keys()):
    if (VesselTypes[mmsi_] == ["Cargo"]) or (VesselTypes[mmsi_] == ["Tanker"]):
        l_cargo_tanker.append(mmsi_)
    if VesselTypes[mmsi_] == [TYPE]:
        l_shiptype.append(mmsi_)

print("Total number of vessels: ", len(VesselTypes))
print("Total number of cargos/tankers: ", len(l_cargo_tanker))
print("Total number of", TYPE, ": ", len(l_shiptype))

print("Saving vessels' type list with cargo/tankers to:", cargo_tanker_filename)
np.save(os.path.join(out_path, cargo_tanker_filename), l_cargo_tanker)
np.save(
    os.path.join(
        out_path, cargo_tanker_filename.replace("_cargo_tanker.npy", "_shiptype.npy")
    ),
    l_shiptype,
)
print(
    "Saving vessels' type list with",
    TYPE,
    "to: ",
    cargo_tanker_filename.replace("_cargo_tanker.npy", "_shiptype.npy"),
)

# FILTERING
# ======================================
# Selecting AIS messages in the ROI and in the period of interest.
# LAT LON
m_msg = m_msg[m_msg[:, LAT] >= LAT_MIN]
m_msg = m_msg[m_msg[:, LAT] <= LAT_MAX]
m_msg = m_msg[m_msg[:, LON] >= LON_MIN]
m_msg = m_msg[m_msg[:, LON] <= LON_MAX]
# SOG
m_msg = m_msg[m_msg[:, SOG] >= 0]
m_msg = m_msg[m_msg[:, SOG] <= SOG_MAX]
# COG
m_msg = m_msg[m_msg[:, SOG] >= 0]
m_msg = m_msg[m_msg[:, COG] <= 360]
# D2C
# m_msg = m_msg[m_msg[:,D2C]>=D2C_MIN]

# TIME
m_msg = m_msg[m_msg[:, TIMESTAMP] >= 0]
m_msg = m_msg[m_msg[:, TIMESTAMP] >= t_min]
m_msg = m_msg[m_msg[:, TIMESTAMP] <= t_max]
m_msg_train = m_msg[m_msg[:, TIMESTAMP] >= t_train_min]
m_msg_train = m_msg_train[m_msg_train[:, TIMESTAMP] <= t_train_max]
m_msg_valid = m_msg[m_msg[:, TIMESTAMP] >= t_valid_min]
m_msg_valid = m_msg_valid[m_msg_valid[:, TIMESTAMP] <= t_valid_max]
m_msg_test = m_msg[m_msg[:, TIMESTAMP] >= t_test_min]
m_msg_test = m_msg_test[m_msg_test[:, TIMESTAMP] <= t_test_max]

print("Total msgs: ", len(m_msg))
print("Number of msgs in the training set: ", len(m_msg_train))
print("Number of msgs in the validation set: ", len(m_msg_valid))
print("Number of msgs in the test set: ", len(m_msg_test))


# MERGING INTO DICT
# ======================================
# Creating AIS tracks from the list of AIS messages.
# Each AIS track is formatted by a dictionary.
print("Convert to dicts of vessel's tracks...")

# Training set
Vs_train = dict()
for v_msg in tqdm(m_msg_train):
    mmsi = int(v_msg[MMSI])
    if not (mmsi in list(Vs_train.keys())):
        Vs_train[mmsi] = np.empty((0, 10))
    Vs_train[mmsi] = np.concatenate(
        (Vs_train[mmsi], np.expand_dims(v_msg[:10], 0)), axis=0
    )
for key in tqdm(list(Vs_train.keys())):
    if CARGO_TANKER_ONLY and (key not in l_cargo_tanker):
        del Vs_train[key]
    else:
        Vs_train[key] = np.array(
            sorted(Vs_train[key], key=lambda m_entry: m_entry[TIMESTAMP])
        )

# Validation set
Vs_valid = dict()
for v_msg in tqdm(m_msg_valid):
    mmsi = int(v_msg[MMSI])
    if not (mmsi in list(Vs_valid.keys())):
        Vs_valid[mmsi] = np.empty((0, 10))
    Vs_valid[mmsi] = np.concatenate(
        (Vs_valid[mmsi], np.expand_dims(v_msg[:10], 0)), axis=0
    )
for key in tqdm(list(Vs_valid.keys())):
    if CARGO_TANKER_ONLY and (key not in l_cargo_tanker):
        del Vs_valid[key]
    else:
        Vs_valid[key] = np.array(
            sorted(Vs_valid[key], key=lambda m_entry: m_entry[TIMESTAMP])
        )

# Test set
Vs_test = dict()
for v_msg in tqdm(m_msg_test):
    mmsi = int(v_msg[MMSI])
    if not (mmsi in list(Vs_test.keys())):
        Vs_test[mmsi] = np.empty((0, 10))
    Vs_test[mmsi] = np.concatenate(
        (Vs_test[mmsi], np.expand_dims(v_msg[:10], 0)), axis=0
    )
for key in tqdm(list(Vs_test.keys())):
    if CARGO_TANKER_ONLY and (key not in l_cargo_tanker):
        del Vs_test[key]
    else:
        Vs_test[key] = np.array(
            sorted(Vs_test[key], key=lambda m_entry: m_entry[TIMESTAMP])
        )


# PICKLING
# ======================================
for filename, filedict in zip(
    [pkl_filename_train, pkl_filename_valid, pkl_filename_test],
    [Vs_train, Vs_valid, Vs_test],
):
    print("Writing to ", os.path.join(out_path, filename), "...")
    with open(os.path.join(out_path, filename), "wb") as f:
        pickle.dump(filedict, f)
    print("Total number of tracks: ", len(filedict))
