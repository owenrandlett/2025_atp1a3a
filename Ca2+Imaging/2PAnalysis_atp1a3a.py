#%%

import os
import glob
from natsort import natsorted
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
#%%
import os
import glob
from natsort import natsorted

raw_data_fldrs_path = r'/media/BigBoy/ciqle/2p/20250902-11_atp1a3a_experiments'
raw_data_fldrs = natsorted(glob.glob(raw_data_fldrs_path + '/*/*_func-000'))

processed_data_flds_path = r'/media/FastDrive/atp1a3a_data'
processed_data_s2pfld = natsorted(glob.glob(processed_data_flds_path + '/*/suite2p'))
processed_data_fldrs = [os.path.split(f)[0] for f in processed_data_s2pfld]


fish_dict = {
    "-/-": [
        "20250902_atp1a3a_Fish4_susMut_func-000",
        "20250903_atp1a3a_Fish01_func-000",
        "20250903_atp1a3a_Fish04_func-000",
        "20250904_atp1a3a_Fish01_func-000",
        "20250904_atp1a3a_Fish08_func-000",
        "20250909_atp1a3a_Fish05_func-000",
        "20250909_atp1a3a_Fish09_func-000",
        "20250910_atp1a3a_Fish7_func-000",
        "20250911_atp1a3a_Fish2_func-000",
        "20250911_atp1a3a_Fish10_func-000",
        "20250911_atp1a3a_Fish11_func-000",
    ],
    "+/-": [
        "20250902_atp1a3a_Fish1_susMut_func-000",
        "20250902_atp1a3a_Fish3_susMut_func-000",
        "20250902_atp1a3a_Fish5_susMut_func-000",
        "20250903_atp1a3a_Fish05_func-000",
        "20250903_atp1a3a_Fish06_func-000",
        "20250904_atp1a3a_Fish02_func-000",
        "20250904_atp1a3a_Fish03_func-000",
        "20250904_atp1a3a_Fish04_func-000",
        "20250904_atp1a3a_Fish05_func-000",
        "20250904_atp1a3a_Fish06_func-000",
        "20250904_atp1a3a_Fish07_func-000",
        "20250909_atp1a3a_Fish02_func-000",
        "20250909_atp1a3a_Fish03_func-000",
        "20250909_atp1a3a_Fish07_func-000",
        "20250909_atp1a3a_Fish08_func-000",
        "20250910_atp1a3a_Fish2_func-000",
        "20250910_atp1a3a_Fish4_func-000",
        "20250910_atp1a3a_Fish5_func-000",
        "20250910_atp1a3a_Fish8_func-000",
        "20250910_atp1a3a_Fish9_func-000",
        "20250911_atp1a3a_Fish4_func-000",
        "20250911_atp1a3a_Fish6_func-000",
        "20250911_atp1a3a_Fish8_func-000",
        "20250911_atp1a3a_Fish9_func-000",
    ],
    "+/+": [
        "20250903_atp1a3a_Fish03_func-000",
        "20250909_atp1a3a_Fish01_func-000",
        "20250909_atp1a3a_Fish04_func-000",
        "20250909_atp1a3a_Fish10_func-000",
        "20250910_atp1a3a_Fish1_func-000",
        "20250910_atp1a3a_Fish3_func-000",
        "20250910_atp1a3a_Fish6_func-000",
        "20250911_atp1a3a_Fish3_func-000",
        "20250911_atp1a3a_Fish5_func-000",
        "20250911_atp1a3a_Fish7_func-000",
    ],
}

# --- build quick lookup: folder -> category ---
category_lookup = {}
for cat, flist in fish_dict.items():
    for f in flist:
        category_lookup[f] = cat

# --- match based on final folder name ---
raw_map = {os.path.basename(f): f for f in raw_data_fldrs}
processed_map = {os.path.basename(f): f for f in processed_data_fldrs}

common_keys = sorted(set(raw_map.keys()) & set(processed_map.keys()))

# build matched pairs with category
matched_pairs = []
for k in common_keys:
    cat = category_lookup.get(k, "UNKNOWN")
    matched_pairs.append((raw_map[k], processed_map[k], cat))

# --- check for missing matches ---
missing_in_processed = set(raw_map.keys()) - set(processed_map.keys())
missing_in_raw = set(processed_map.keys()) - set(raw_map.keys())

if missing_in_processed:
    print("⚠️ No processed match for these raw folders:")
    for m in sorted(missing_in_processed):
        print("   ", raw_map[m])

if missing_in_raw:
    print("⚠️ No raw match for these processed folders:")
    for m in sorted(missing_in_raw):
        print("   ", processed_map[m])

# --- optional: check for unmatched to fish_dict ---
not_in_fish_dict = [k for k in common_keys if k not in category_lookup]
if not_in_fish_dict:
    print("⚠️ These matched folders are not assigned to any category in fish_dict:")
    for m in not_in_fish_dict:
        print("   ", m)


#%%
def get_fish_category(fish_type):
    if fish_type == "-/-":
        return 2
    elif fish_type == "+/-":
        return 1
    elif fish_type == "+/+":
        return 0
    else:
        return -1  # Unknown category
    
cell_thresh = 0.3 # classifier probability threshold
ops = {}
k = 6
data_path =  matched_pairs[k][1]
fish_type = get_fish_category(matched_pairs[k][2]) # 0=WT, 1=het, 2=hom

print(f"Processing fish: {data_path}, type: {fish_type}")
#%%
fish_name = os.path.split(data_path)[1]
planes = natsorted(glob.glob(os.path.join(data_path,'*plane*_data.npy')))


# load in planes for that fish
for i in range(len(planes)):
    plane_data = np.load(planes[i], allow_pickle=True).item()
    #print(plane_data['plane'])
    roi_stats_temp = plane_data['roi_stats']
    iscell = plane_data['iscell']
    # cells = iscell[:,0] == 1
    cells = iscell[:,1] > cell_thresh
    n_cells = np.sum(cells)
    roi_stats_temp = roi_stats_temp[cells]
    F_temp = stats.zscore(plane_data['F'][cells,:], axis=1)
    F_temp[~np.isfinite(F_temp)] = 0
    fish_data_temp = np.stack((k*np.ones(n_cells), fish_type*np.ones(n_cells))).T.astype('uint8')

    for roi in range(len(roi_stats_temp)):
        roi_stats_temp[roi]['fish_name']=fish_name
    if i == 0 and k == 0: 
        roi_stats = roi_stats_temp
        F_norm = F_temp
        fish_data = fish_data_temp
    else:
        roi_stats = np.hstack((roi_stats, roi_stats_temp))
        F_norm = np.vstack((F_norm, F_temp))
        fish_data = np.vstack((fish_data, fish_data_temp))
    
    if i == 5:
        ops[fish_name] = plane_data['ops']


#%% load stimulus data
stim_file = glob.glob(matched_pairs[k][0] + '/*/exp_params.csv')
stim_data = pd.read_csv(stim_file[0])