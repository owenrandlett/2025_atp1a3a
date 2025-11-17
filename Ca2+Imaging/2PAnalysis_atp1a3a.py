#%%

import os
import glob
from natsort import natsorted
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
import pynwb
from pynwb import NWBHDF5IO
import tifffile
from scipy.ndimage import zoom, morphology
import nrrd


def ffill_cols(a, startfillval=0):
    
    ### fill NaN values with previous value
    mask = np.isnan(a)
    tmp = a[0].copy()
    a[0][mask[0]] = startfillval
    mask[0] = False
    idx = np.where(~mask,np.arange(mask.shape[0])[:,None],0)
    out = np.take_along_axis(a,np.maximum.accumulate(idx,axis=0),axis=0)
    a[0] = tmp
    return out

def rolling_window(a, window):
    pad = np.ones(len(a.shape), dtype=np.int32)
    pad[-1] = window-1
    pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
    a = np.pad(a, pad,mode='reflect')
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def draw_hit_volume(hits_inds, values = [1], draw_centroid=False, add_write=True, proj_mean=True, draw_outline=False, save_name = None, normalize=True):
    hits_inds_shuf = hits_inds.copy()
    np.random.shuffle(hits_inds_shuf)
    IM_roi = np.zeros((Zs, height, width))
    for j in range(len(hits_inds)):
        roi_coords_y = roi_stats[hits_inds[j]]['ypix_refbrain'].astype('int')
        roi_coords_x = roi_stats[hits_inds[j]]['xpix_refbrain'].astype('int')
        roi_coords_z = roi_stats[hits_inds[j]]['centroid_refbrain'][2].astype('int')
        roi_coords_z = np.arange(roi_coords_z-2, roi_coords_z+2) # take a 5 z-planes to make it more comparable with xy size
        roi_coords_y[roi_coords_y > height-1] = height-1
        roi_coords_x[roi_coords_x > width-1] = width-1
        roi_coords_z[roi_coords_z > Zs-1] = Zs-1
        # if roi_coords_z > Zs-1:
        #     roi_coords_z = Zs-1
        if draw_centroid:
            roi_coords_y = np.mean(roi_coords_y).astype('int')
            roi_coords_x = np.mean(roi_coords_x).astype('int')
            roi_coords_z = np.mean(roi_coords_z).astype('int')
        if add_write:
            if len(values) == 1:
                for z in roi_coords_z:  
                    IM_roi[z, roi_coords_y, roi_coords_x]  += values
            else:
                for z in roi_coords_z:  
                    IM_roi[z, roi_coords_y, roi_coords_x]  += values[j]
        else:
            if len(values) == 1:  
                for z in roi_coords_z:  
                    IM_roi[z, roi_coords_y, roi_coords_x]  = values
            else:
                for z in roi_coords_z:  
                    IM_roi[z, roi_coords_y, roi_coords_x]  = values[j]


    if proj_mean:
        im_proj_z = np.mean(IM_roi[:,:, :], axis=0)
        im_proj_x = zoom(np.mean(IM_roi[:,:, :], axis=2).T, [1, z_rez/xy_rez])
    else:
        im_proj_z = np.max(IM_roi[:,:, :], axis=0)
        im_proj_x = zoom(np.max(IM_roi[:,:, :], axis=2).T, [1, z_rez/xy_rez])
    
    if normalize:
        im_proj = np.hstack((im_proj_z/np.max(im_proj_z), im_proj_x/np.max(im_proj_x)))
    else:
        im_proj = np.hstack((im_proj_z, im_proj_x))

    if draw_outline:
        im_proj[outline > 0.01] = np.max(im_proj)

    # if not save_name==None:
    #     imsave(os.path.join(analysis_out, save_name+'_proj_image.tif'), im_proj)
    return IM_roi, im_proj


re_analyze = False # set to True to re-process all data from raw
raw_data_fldrs_path = r'/media/BigBoy/ciqle/2p/20250902-11_atp1a3a_experiments'
processed_data_flds_path = r'/media/FastDrive/atp1a3a_data'
out_dir = os.path.join(processed_data_flds_path, 'Outputs_2pAnalysis')
os.makedirs(out_dir, exist_ok=True)



raw_data_fldrs = natsorted(glob.glob(raw_data_fldrs_path + '/*/*_func-000'))
processed_data_s2pfld = natsorted(glob.glob(processed_data_flds_path + '/*/suite2p'))
processed_data_fldrs = [os.path.split(f)[0] for f in processed_data_s2pfld]


fish_dict = {
    "+/+": [
        "20250903_atp1a3a_Fish03_func-000",
        # "20250909_atp1a3a_Fish01_func-000", # z position unstable
        "20250909_atp1a3a_Fish04_func-000",
        "20250909_atp1a3a_Fish10_func-000",
        # "20250910_atp1a3a_Fish1_func-000",  # z position unstable
        "20250910_atp1a3a_Fish3_func-000",
        "20250910_atp1a3a_Fish6_func-000",
        "20250911_atp1a3a_Fish3_func-000",
        # "20250911_atp1a3a_Fish5_func-000", # no neural activity or behaviour, fish presumably dead
        "20250911_atp1a3a_Fish7_func-000",
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
        # "20250911_atp1a3a_Fish4_func-000", # z position unstable
        "20250911_atp1a3a_Fish6_func-000",
        "20250911_atp1a3a_Fish8_func-000",
        "20250911_atp1a3a_Fish9_func-000",
    ],
    "-/-": [
        "20250902_atp1a3a_Fish4_susMut_func-000",
        # "20250903_atp1a3a_Fish01_func-000", # z position unstable
        "20250903_atp1a3a_Fish04_func-000",
        # "20250904_atp1a3a_Fish01_func-000", # z position unstable
        "20250904_atp1a3a_Fish08_func-000",
        "20250909_atp1a3a_Fish05_func-000",
        "20250909_atp1a3a_Fish09_func-000",
        "20250910_atp1a3a_Fish7_func-000",
        "20250911_atp1a3a_Fish2_func-000",
        "20250911_atp1a3a_Fish10_func-000",
        "20250911_atp1a3a_Fish11_func-000",
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
#%


def get_fish_category(fish_type):
    if fish_type == "-/-":
        return 2
    elif fish_type == "+/-":
        return 1
    elif fish_type == "+/+":
        return 0
    else:
        return -1  # Unknown category
    
if re_analyze:
    cell_thresh = 0.3 # classifier probability threshold
    ops = {}
    for k in range(len(matched_pairs)):
        data_path =  matched_pairs[k][1]
        fish_type = get_fish_category(matched_pairs[k][2]) # 0=WT, 1=het, 2=hom

        print(f"Processing fish: {data_path}, type: {fish_type}")
        
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
            F_raw = plane_data['F'][cells,:]
            F_temp = stats.zscore(plane_data['F'][cells,:], axis=1)
            F_temp[~np.isfinite(F_temp)] = 0
            fish_data_temp = np.stack((k*np.ones(n_cells), fish_type*np.ones(n_cells))).T.astype('uint8')

            for roi in range(len(roi_stats_temp)):
                roi_stats_temp[roi]['fish_name']=fish_name
            if i == 0 and k == 0: 
                roi_stats = roi_stats_temp
                F = F_raw
                F_norm = F_temp
                fish_data = fish_data_temp
            else:
                roi_stats = np.hstack((roi_stats, roi_stats_temp))
                F = np.vstack((F, F_raw))
                F_norm = np.vstack((F_norm, F_temp))
                fish_data = np.vstack((fish_data, fish_data_temp))
            
            # if i == 5:
            ops[fish_name] = plane_data['ops']
            ops[fish_name]['fish_ind'] = k

        #% load timestamp data
        try:
            nwb_filename = glob.glob(data_path + r'/*.nwb')[0]

            twophoton_series_names = []
            frame_rates = []
            with NWBHDF5IO(nwb_filename, 'r') as io:
                nwbfile = io.read()
                
                    # Get all TwoPhotonSeries objects
                for name, obj in nwbfile.acquisition.items():
                    if isinstance(obj, pynwb.ophys.TwoPhotonSeries):
                
                        twophoton_series_names.append(name)
                        # Access timestamps
                        timestamps = np.copy(obj.timestamps)
                        if timestamps is not None:
                            frame_periods = np.diff(timestamps[:])
                            frame_rates.append(1000 / np.mean(frame_periods))
            ops[fish_name]['frame_rates'] = frame_rates
            ops[fish_name]['timestamps'] = (timestamps - timestamps[0])/1000 # in seconds
        except:
            print(f"Could not load NWB file for {data_path}")
        
        # load behaviour data:

        stim_file = glob.glob(matched_pairs[k][0] + '/*/exp_params.csv')
        stim_data = pd.read_csv(stim_file[0])

        coords_file = glob.glob(matched_pairs[k][0] + '/*/coords.txt')
        tstamps_file = glob.glob(matched_pairs[k][0] + '/*/tstamps.txt')



        coords = np.loadtxt(coords_file[0], delimiter=",")
        coords = ffill_cols(coords)
        t_stamps = np.loadtxt(tstamps_file[0], delimiter=',')

        #%

        microscope_frames = t_stamps[1::2] 
        microscope_frames = microscope_frames - microscope_frames[0]
        time = t_stamps[::2]
        x_coords = coords[::2, :]
        x_coords = x_coords - np.nanmean(x_coords[:,0], axis=0)
        y_coords = coords[1::2, :] 
        y_coords = y_coords - np.nanmean(y_coords[:,0], axis=0)
        #%
        min_coords = min(x_coords.shape[0], y_coords.shape[0])
        x_coords = x_coords[:min_coords, :]
        y_coords = y_coords[:min_coords, :]

        angles = np.arctan2(np.diff(y_coords, axis=1), np.diff(x_coords, axis=1))
        angles = np.unwrap(angles)

        orients = np.nanmean(angles, axis=1)
        diff_angles = np.diff(angles, axis=1)

        bend_amps = np.nanmean(diff_angles, axis=1)
        bend_amps[np.isnan(bend_amps)] = 0
        bend_amps_filt = savgol_filter(bend_amps, 11, 5)
        orients_filt = savgol_filter(orients, 11, 1)
        orients_filt = np.rad2deg(orients_filt - np.median(orients_filt))

        min_len = min(len(time), len(orients_filt)) # something weird here - we might be dropping frames. Behaviour may not be synched properly...
        plt.plot(time[:min_len], orients_filt[:min_len], label='orientation')
        plt.ylim([-35, 35])
        plt.xlabel('time(sec)')
        plt.ylabel('delta orientation (deg)')
        plt.title(fish_name + ' : type : ' + matched_pairs[k][2])
        plt.show()


        ops[fish_name]['pi_tailtrack'] = {}
        ops[fish_name]['pi_tailtrack']['microscope_frames'] = microscope_frames
        ops[fish_name]['pi_tailtrack']['time'] = time
        ops[fish_name]['pi_tailtrack']['x_coords'] = x_coords
        ops[fish_name]['pi_tailtrack']['y_coords'] = y_coords
        ops[fish_name]['pi_tailtrack']['orients'] = orients
        ops[fish_name]['pi_tailtrack']['orients_filt'] = orients_filt
        ops[fish_name]['pi_tailtrack']['bend_amps'] = bend_amps
        ops[fish_name]['pi_tailtrack']['bend_amps_filt'] = bend_amps_filt
        ops[fish_name]['pi_tailtrack']['stim_data'] = stim_data

    np.savez(os.path.join(processed_data_flds_path, 'ImagingData_allFish.npz'),
                        roi_stats=roi_stats,
                        F=F, 
                        F_norm=F_norm, 
                        fish_data=fish_data, 
                        ops=ops
                    )

all_fish_data = np.load(os.path.join(processed_data_flds_path, 'ImagingData_allFish.npz'), allow_pickle=True)

roi_stats = all_fish_data['roi_stats']
F = all_fish_data['F']
F_norm = all_fish_data['F_norm']
fish_data = all_fish_data['fish_data']
ops = all_fish_data['ops'].item()

#% calculate dF/F for each neuron
Fo = np.nanmedian(F, axis=1, keepdims=True)
F_dff = (F - Fo) / Fo
# Remove NaNs from F_dff (set them to zero)
F_dff[~np.isfinite(F_dff)] = 0

ref_brain_path = r'/media/FastDrive/atp1a3a_data/registration/telen_template_allfish_HuC-H2BGCaMP.nrrd' 
ref_brain, ref_meta = nrrd.read(ref_brain_path)
width, height, Zs = ref_brain.shape
ref_brain = np.moveaxis(ref_brain, [0,1,2], [2,1,0])


xy_rez = ref_meta['space directions'][0][0]
z_rez = ref_meta['space directions'][-1][-1]

#%% analze behaviour + imaging traces
from scipy.signal import medfilt
def safe_filename(s: str, replacement: str = "_", max_length: int = 255) -> str:
    import re
    """
    Make a string safe for use as a filename on most OSes.
    - Replaces invalid characters with `replacement`
    - Strips leading/trailing whitespace
    - Truncates to max_length
    """
    # Replace invalid characters
    s = re.sub(r'[<>:"/\\|?*]', replacement, s)
    # Replace whitespace with underscore
    s = re.sub(r'\s+', replacement, s)
    # Remove leading dots (avoid hidden files / special names)
    s = s.lstrip(".")
    # Truncate to maximum filename length
    return s[:max_length]

out_dir_behavPlots = os.path.join(out_dir, 'BehaveTraces_Fnorm')
os.makedirs(out_dir_behavPlots, exist_ok=True)

if re_analyze:
    for fish_ind in range(len(ops)):
        
        plt.figure(figsize=(25,17))

        keys_fish = list(ops.keys())
        fish_name = keys_fish[fish_ind]
        pi_tailtrack = ops[fish_name]['pi_tailtrack']
        mic_timestamps = ops[fish_name]['timestamps']

        stim_data = pi_tailtrack['stim_data']

        # --- OMR ---
        mask = stim_data["omr_cycle_rate"] == 0.6
        OMR_start_sec = stim_data.loc[mask, "time (sec)"].values
        OMR_end_sec   = stim_data.loc[mask.shift(fill_value=False), "time (sec)"].values

        OMR_vec = np.zeros_like(mic_timestamps, dtype=int)
        for start, end in zip(OMR_start_sec, OMR_end_sec):
            mask = (mic_timestamps >= start) & (mic_timestamps <= end)
            OMR_vec[mask] = 1

        # --- DF ---
        mask = stim_data["stim_brighness"] == 0
        DF_start_sec = stim_data.loc[mask, "time (sec)"].values
        DF_end_sec   = stim_data.loc[mask.shift(fill_value=False), "time (sec)"].values

        DF_vec = np.zeros_like(mic_timestamps, dtype=int)
        for start, end in zip(DF_start_sec, DF_end_sec):
            mask = (mic_timestamps >= start) & (mic_timestamps <= end)
            DF_vec[mask] = 1

        bend_amps_filt = pi_tailtrack['bend_amps_filt']
        orients_filt = pi_tailtrack['orients_filt']
        behav_time = pi_tailtrack['time']

        frame_rate_behav = int(1/np.median(np.diff(behav_time)))
        tail_power = np.std(rolling_window(bend_amps_filt, frame_rate_behav), -1)
        tail_power = tail_power - np.median(tail_power)

        swim_bursting = medfilt(tail_power, frame_rate_behav*20+1)

        max_inds_behav = min(len(bend_amps_filt), len(behav_time))
        # --- Plot ---
        
        plt.plot(behav_time[:max_inds_behav], bend_amps_filt[:max_inds_behav], label="Bend Amps")
        lowpass_orients = medfilt(orients_filt, frame_rate_behav*13+1) 
        # plt.plot(behav_time[:max_inds_behav], (orients_filt[:max_inds_behav] - np.mean(orients_filt))/50, label="Orientations")  
        plt.plot(behav_time[:max_inds_behav], (lowpass_orients[:max_inds_behav] - np.mean(lowpass_orients))/50, linewidth=5, label="Orientations_lowpass")
        plt.plot(behav_time[:max_inds_behav], tail_power[:max_inds_behav], linewidth=3, label="Swimming Power")
        plt.plot(behav_time[:max_inds_behav], swim_bursting[:max_inds_behav], linewidth = 3, label="Swimg Bursting")
        plt.plot(mic_timestamps, np.mean(F_norm[fish_data[:,0]==fish_ind, :], axis=0), linewidth = 2, label="Mean F_norm")
        # plt.plot(mic_timestamps, np.mean(F_norm[:, :], axis=0), label="Mean F_norm all cells")
        ylim_max = 0.5
        plt.plot(mic_timestamps, OMR_vec*0.1 - ylim_max, linewidth = 3, label="OMR")
        plt.plot(mic_timestamps, DF_vec*0.15 - ylim_max, linewidth = 3, label="DF")  # offset a bit for visibility
        plot_title = fish_name + ' : type : ' + matched_pairs[fish_ind][2]
        plt.title(plot_title, fontsize=30)

        plt.legend(fontsize=30)

        plt.ylim([-ylim_max, ylim_max])

        plt.savefig(os.path.join(out_dir_behavPlots, safe_filename(plot_title + '.png')))
        plt.savefig(os.path.join(out_dir_behavPlots, safe_filename(plot_title + '.svg')))
        plt.show()

        # Save relevant behavioral traces in ops[fish_name]
        ops[fish_name]['behav_traces'] = {
            'bend_amps_filt': bend_amps_filt,
            'orients_filt': orients_filt,
            'tail_power': tail_power,
            'swim_bursting': swim_bursting,
            'lowpass_orients': lowpass_orients,
            'OMR_vec': OMR_vec,
            'DF_vec': DF_vec,
            'mic_timestamps': mic_timestamps,
            'behav_time': behav_time,
            'stim_data': stim_data
        }

    ops_save_path = os.path.join(processed_data_flds_path, 'ops_updated_wBehav.npy')
    np.save(ops_save_path, ops)
    print(f"Updated ops file saved to: {ops_save_path}")

# Reload the updated ops file for further analysis
ops_reload_path = os.path.join(processed_data_flds_path, 'ops_updated_wBehav.npy')
ops = np.load(ops_reload_path, allow_pickle=True).item()
print(f"Reloaded updated ops file from: {ops_reload_path}")



# %%

import seaborn as sns
from numba import njit, prange



def GCaMPConvolve(trace, ker):
    if np.sum(trace) == 0:
        return trace
    else:
        trace_conv = np.convolve(trace, ker, 'full')
        trace_conv = trace_conv[1:trace.shape[0]+1] 
        trace_conv[np.logical_not(np.isfinite(trace_conv))] = 0
        trace_conv = trace_conv/max(trace_conv)
        return trace_conv



def pearsonr_2D(x, y):
    """computes pearson correlation coefficient
       where x is a 1D and y a 2D array
       from https://stackoverflow.com/questions/30143417/computing-the-correlation-coefficient-between-two-multi-dimensional-arrays """

    upper = np.sum((x - np.mean(x)) * (y - np.mean(y, axis=1)[:,None]), axis=1)
    lower = np.sqrt(np.sum(np.power(x - np.mean(x), 2)) * np.sum(np.power(y - np.mean(y, axis=1)[:,None], 2), axis=1))
    
    rho = upper / lower
    
    return rho

@njit
def pearsonr_numba2(x, y):
    """computes pearson correlation coefficient
       where x is a 1D and y a 2D array
       from https://stackoverflow.com/questions/30143417/computing-the-correlation-coefficient-between-two-multi-dimensional-arrays """
    n_var = y.shape[1]
    y_mean = np.sum(y, axis=1) / n_var
    y_mean = y_mean.repeat(n_var).reshape((-1, n_var))

    upper = np.sum((x - np.mean(x)) * (y - y_mean), axis=1)
    

    lower = np.sqrt(np.sum(np.power(x - np.mean(x), 2)) * np.sum(np.power(y - y_mean, 2), axis=1))
    
    rho = upper / lower
    
    return rho

@njit
def pearsonr_vec_2Dnumb(x,y):
    # computes the pearson correlation coefficient between a a vector (x) and each row in 2d matrix (y), using numba acceleration
    
    n_rows_y = int(y.shape[0])
    corr = np.zeros((n_rows_y))
    for row_y in prange(n_rows_y):
        corr[row_y] = np.corrcoef(x, y[row_y,:])[0,1]
    return corr


@njit
def pearsonr_2Dnumb(x,y, print_progress = False):

    # computes the pearson correlation coefficient between a each row in 2d matrix (x) and each row in 2d matrix (y), using numba acceleration

    n_rows_y = int(y.shape[0])
    n_rows_x = x.shape[0]
    corr = np.zeros((n_rows_x, n_rows_y))

    for row_x in prange(n_rows_x):
        for row_y in prange(n_rows_y):
            y[row_y,:]
            x[row_x, :]
            corr[row_x, row_y] = np.corrcoef(x[row_x, :], y[row_y,:])[0,1]
        if print_progress:
            print('done correlations on row ' + str(row_x) + ' in x, out of ' + str(n_rows_x))

    return corr


col_map = sns.diverging_palette(360,180, s=100, l=50, sep=30, as_cmap=True, center="dark")

start_analyze_frame = 200  # ignore the first frames for correlation analyses, scanning artifact should be done by then

# parameters for GCaMP kernel
DecCnst = 0.3
RiseCnst = 0.5
frame_rate = 1.976
DecCnst = DecCnst*frame_rate # now in frames
RiseCnst = RiseCnst*frame_rate

KerRise = np.power(2, (np.arange(0,5)*RiseCnst)) - 1
KerRise= KerRise[KerRise < 1]
KerRise = KerRise/max(KerRise)

KerDec = np.power(2, (np.arange(20, 0, -1)*DecCnst))
KerDec = (KerDec - min(KerDec))/(max(KerDec) - min(KerDec));

KerDec = KerDec[KerDec > 0]
KerDec = KerDec[1:]
KerTotal = np.concatenate([KerRise, KerDec])
plt.plot(np.arange(len(KerTotal))/frame_rate, KerTotal)
plt.xlabel('seconds')
plt.ylabel('predicted GCaMP\nresponse')
plt.show()

# z-brain dimensions
height = 1406
width = 621
Zs = 138

color_fish = ['#2258e0', '#22e061', '#e02222']

DF_vec = ops[list(ops.keys())[0]]['behav_traces']['DF_vec']
OMR_vec = ops[list(ops.keys())[0]]['behav_traces']['OMR_vec']


stim_df_conv = GCaMPConvolve(DF_vec, KerTotal)
stim_omr_conv = GCaMPConvolve(OMR_vec, KerTotal)

plt.plot(stim_df_conv)
plt.plot(stim_omr_conv)
plt.show()

regressor_names = [
'Dark Flashes',
'OMR',
'Tail Power',
'Swim Bursting',
'Lowpass Orientation'
]

def resample_to_reference(high_t, high_y, low_t, method="linear", fill_value="extrapolate"):
    """
    Resample a signal sampled at high_t to the time base of low_t.

    Parameters
    ----------
    high_t : array-like
        Timestamps of the high-rate signal.
    high_y : array-like
        Signal values at high_t (1D).
    low_t : array-like
        Target timestamps (usually lower rate).
    method : str, optional
        Interpolation method ("linear", "nearest", "cubic", etc.).
    fill_value : str or float, optional
        What to do outside the range of high_t.
        Default "extrapolate", can also be a float (e.g. 0).

    Returns
    -------
    low_y : np.ndarray
        Resampled signal matching low_t.
    """
    from scipy.interpolate import interp1d
    
    f = interp1d(high_t, high_y, kind=method, fill_value=fill_value, bounds_error=False)
    return f(low_t)


if re_analyze:
    regressors_per_fish = []
    behav_data_per_fish = []
    for fish_ind in range(len(ops)):

        fish_name = keys_fish[fish_ind]
        fish_IDs = np.where(fish_data[:,0] == fish_ind)[0]
        F_norm_fish = F_norm[fish_IDs, :]
        nROIs = len(fish_IDs)
        pi_tailtrack = ops[fish_name]['pi_tailtrack']
        microscope_timestamps = ops[fish_name]['timestamps']
        bend_amps_filt = pi_tailtrack['bend_amps_filt']
        orients_filt = pi_tailtrack['orients_filt']
        behav_time = pi_tailtrack['time']

        max_inds_behav = min(len(bend_amps_filt), len(behav_time))
        bend_amps_filt = bend_amps_filt[:max_inds_behav]
        orients_filt = orients_filt[:max_inds_behav]
        behav_time = behav_time[:max_inds_behav]


        frame_rate_behav = int(1/np.median(np.diff(behav_time)))
        tail_power = np.std(rolling_window(bend_amps_filt, frame_rate_behav), -1)
        tail_power = tail_power - np.median(tail_power)
        tail_power_resample = resample_to_reference(behav_time, tail_power, microscope_timestamps)
        tail_power_conv = GCaMPConvolve(tail_power_resample, KerTotal)
        
        swim_bursting = medfilt(tail_power, frame_rate_behav*20+1)
        swim_bursting_resample = resample_to_reference(behav_time, swim_bursting, microscope_timestamps)
        swim_bursting_conv = GCaMPConvolve(swim_bursting_resample, KerTotal)
        
        lowpass_orients = medfilt(orients_filt, frame_rate_behav*13+1)
        lowpass_orients_resample = resample_to_reference(behav_time, lowpass_orients, microscope_timestamps)
        lowpass_orients_conv = GCaMPConvolve(lowpass_orients_resample, KerTotal)

        regressors = np.vstack((
            stim_df_conv, 
            stim_omr_conv,
            tail_power_conv,
            swim_bursting_conv,
            lowpass_orients_conv
        ))

        behav_data = np.vstack((
            DF_vec,
            OMR_vec,
            tail_power_resample,
            swim_bursting_resample,
            lowpass_orients_resample
        ))
        regressors_per_fish.append(regressors)
        behav_data_per_fish.append(behav_data)

        n_regressors = regressors.shape[0]
        corrMat_temp = np.zeros([nROIs, n_regressors])
        for regr in range(n_regressors):
            corrMat_temp[:, regr] = pearsonr_vec_2Dnumb(regressors[regr, start_analyze_frame:], F_norm_fish[:, start_analyze_frame:])

        corrMat_temp[np.isnan(corrMat_temp)] = 0 # set invalid correlations to 0

        if fish_ind == 0:
            corrMat = np.copy(corrMat_temp)
        else:
            corrMat = np.vstack((corrMat, corrMat_temp))

        

    # Compile relevant correlation analysis results into a dict
    correlation_results = {
        "fish_names": list(ops.keys()),
        "fish_data": fish_data,                # (N_cells, 2) [fish_ind, fish_type]
        "regressors_per_fish": regressors_per_fish,              # list of (N_regressors, N_timepoints) regressor traces per fish
        "behav_data_per_fish": behav_data_per_fish,              # list of (N_regressors, N_timepoints) behavioral data traces per fish
        "regressor_names": regressor_names,    # list of regressor labels
        "corrMat": corrMat,                    # (N_cells, N_regressors) correlation matrix
        "F_norm": F_norm,                      # normalized fluorescence traces
        "F_dff": F_dff,                      # dF/F fluorescence traces
        "roi_stats": roi_stats,                # ROI metadata
    }

    # Save to disk in the same folder as other outputs
    corr_save_path = os.path.join(out_dir, "correlation_results.npz")
    np.savez(corr_save_path, **correlation_results)
    print(f"Saved correlation results to: {corr_save_path}")


# Reload correlation results for subsequent analyses
corr_load_path = os.path.join(out_dir, "correlation_results.npz")
corr_data = np.load(corr_load_path, allow_pickle=True)

fish_names = corr_data["fish_names"]
fish_data = corr_data["fish_data"]
regressor_names = corr_data["regressor_names"]
behav_data_per_fish = corr_data["behav_data_per_fish"]
corrMat = corr_data["corrMat"]
F_norm = corr_data["F_norm"]
F_dff = corr_data["F_dff"]
roi_stats = corr_data["roi_stats"]
regressors_per_fish = corr_data["regressors_per_fish"]

print(f"Reloaded correlation results from: {corr_load_path}")



#%%
corr_thresh = 0.1
inds_hits = []
for regr in range(corrMat.shape[1]):
    inds_hits.append(np.where(corrMat[:,regr] >= corr_thresh)[0])
    plt.plot(np.mean(F_norm[inds_hits[regr], :], axis=0), label=regressor_names[regr])
plt.ylabel('Mean z-scored fluorescence of ROIs with\ncorrelation > ' + str(corr_thresh))
plt.xlabel('Frame number')
plt.title('Mean activity of ROIs correlated with each regressor')

plt.legend()

#%% clustering of functional responses
from sklearn.cluster import AffinityPropagation, KMeans, SpectralClustering, AgglomerativeClustering


F_dff_std = np.nanstd(F_dff, axis=1)
#%
std_thresh = 0.6
std_above_thresh = F_dff_std >= std_thresh

corr_thresh = 0.1
corr_above_thresh = np.max(abs(corrMat), axis=1) >= corr_thresh

active_neurons = np.where(np.logical_or(std_above_thresh, corr_above_thresh))[0]

plt.hist(F_dff_std, bins=np.arange(0, 2, 0.01))
plt.vlines(std_thresh, 0, 5000, colors='r', linestyles='dashed')

print(f'Number of neurons above std threshold: {np.sum(std_above_thresh)}')
print(f'Number of neurons above corr threshold: {np.sum(corr_above_thresh)}')
print(f'Number of active neurons selected for clustering: {len(active_neurons)}')

#%
IM_roi, im_show = draw_hit_volume(active_neurons, draw_outline=False)
plt.figure(figsize=(10,20))
plt.imshow(im_show, vmin = 0, vmax=0.7, cmap='inferno')
plt.axis('off')
plt.title('units selected for clustering')
plt.show()



#%% 

out_dir_heatmaps = os.path.join(out_dir, 'clustered_heatmaps')
os.makedirs(out_dir_heatmaps, exist_ok=True)
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
# Select first fish

cluster_results = []
for fish_ind in range(len(fish_names)):
    # locate all ROIs belonging to this fish
    fish_IDs = np.where(fish_data[:, 0] == fish_ind)[0]
    # keep only ROIs that passed global activity filters
    active_neurons_in_fish = np.intersect1d(fish_IDs, active_neurons)

    # pull normalized traces (drop initial frames to avoid artefacts)
    traces_to_cluster = F_norm[active_neurons_in_fish, start_analyze_frame:]
    # build similarity matrix (cosine/correlation via dot product)
    corr_m_fish = np.dot(traces_to_cluster, traces_to_cluster.T) / traces_to_cluster.shape[1]

    # run affinity propagation on the similarity matrix
    af = AffinityPropagation(
        preference=-9,
        damping=0.9,
        max_iter=500,
        random_state=1,
        affinity="precomputed",
        verbose=True,
    ).fit(corr_m_fish)
    labels = af.labels_

    # compute mean trace (centroid) for each cluster
    unique_labels = np.unique(labels)
    centroids = np.vstack([traces_to_cluster[labels == lbl].mean(axis=0) for lbl in unique_labels])

    # order clusters so that similar centroids appear next to each other
    if centroids.shape[0] > 1:
        centroid_order = leaves_list(linkage(centroids, method="single"))
        ordered_labels = unique_labels[centroid_order]
    else:
        ordered_labels = unique_labels

    ordered_members = []
    for lbl in ordered_labels:
        # collect neuron indices for the current cluster
        cluster_members = np.where(labels == lbl)[0]
        if cluster_members.size > 1:
            # compute within-cluster correlation matrix
            cluster_traces = traces_to_cluster[cluster_members, :]
            cluster_corr = np.corrcoef(cluster_traces)
            cluster_corr[~np.isfinite(cluster_corr)] = 0
            cluster_corr = np.clip(cluster_corr, -1, 1)
            # convert to condensed distance form for hierarchical ordering
            condensed = squareform(np.clip(1 - cluster_corr, 0, None), checks=False)
            if np.any(condensed > 0):
                # order neurons along the dendrogram leaves for smooth transitions
                member_order = leaves_list(linkage(condensed, method="single"))
                cluster_members = cluster_members[member_order]
            else:
                # fallback: keep original index order
                cluster_members = cluster_members[np.argsort(cluster_members)]
        ordered_members.append(cluster_members)

    # flatten per-cluster order into a single index array
    final_inds = np.concatenate(ordered_members)
    traces_to_cluster_sorted = traces_to_cluster[final_inds, :]
    labels_sorted = labels[final_inds]

    # identify start/end rows for each cluster (for plotting dividers)
    unique_labels_sorted, label_starts = np.unique(labels_sorted, return_index=True)
    label_ends = np.append(label_starts[1:], traces_to_cluster_sorted.shape[0])
    # fetch regressors for this fish (already convolved & resampled)
    reg_signals = regressors_per_fish[fish_ind]
    if isinstance(reg_signals, np.ndarray) and reg_signals.dtype == object:
        reg_signals = np.stack(reg_signals)
    else:
        reg_signals = np.asarray(reg_signals)

    heatmap_cmap = LinearSegmentedColormap.from_list(
        "black_green",
        ["white", "black"],
        N=256,
    )
    heatmap_vmin, heatmap_vmax = 0, 1


    with plt.rc_context({"font.size": 28}):
        fig, axes = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(30, 20),
            gridspec_kw={"height_ratios": [4, 1]},
        )
        ax_heatmap = axes[0]

        sns.heatmap(
            traces_to_cluster_sorted,
            cmap=heatmap_cmap,
            vmin=heatmap_vmin,
            vmax=heatmap_vmax,
            # cbar_kws={"label": "z-score"},
            cbar = False,
            ax=ax_heatmap,
        )

        ax_heatmap.set_title(f"{fish_names[fish_ind]}\nFish Type = {matched_pairs[fish_ind][-1]}")
        for start, end in zip(label_starts, label_ends):
            ax_heatmap.hlines(start, xmin=0, xmax=traces_to_cluster_sorted.shape[1], colors="black", linestyles="--", linewidth=2.5)

        title_str = fish_names[fish_ind] + "\nFish Type = " + matched_pairs[fish_ind][-1]
        ax_heatmap.set_title(title_str)
        ax_heatmap.set_ylabel("Neuron (continuum-ordered clusters)")
        ax_heatmap.collections[0].set_rasterized(True)

        frame_idx = np.arange(traces_to_cluster_sorted.shape[1])
        x_coords = frame_idx + 0.5

        ax = axes[1]
        
        behav_data = behav_data_per_fish[fish_ind].copy()
        behav_data[0,:] = behav_data[0,:] * 0.15  # scale dark flash for visibility
        behav_data[1,:] = behav_data[1,:] * 0.1   # scale OMR for visibility
        behav_data[4,:] = behav_data[4,:] / 50  # scale lowpass orientation
        
        for i in range(len(behav_data)):
            ax.plot(x_coords, behav_data[i, start_analyze_frame : start_analyze_frame + len(frame_idx)], linewidth=2.5, label=regressor_names[i])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        tick_positions = np.linspace(0, frame_idx[-1], 5, dtype=int)
        tick_positions_shifted = tick_positions + 0.5
        axes[-1].set_xlabel("Frame Number")
        axes[-1].set_ylabel("Regressor Signal")
        axes[-1].set_xticks(tick_positions_shifted)
        axes[-1].set_xticklabels((tick_positions + start_analyze_frame).astype(int))
        axes[-1].legend(fontsize=14)


        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_heatmaps, safe_filename(title_str) + '.png'), dpi=300)
        plt.savefig(
            os.path.join(out_dir_heatmaps, safe_filename(title_str) + ".svg"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()
    
    cluster_results.append(
        {
            "fish_index": fish_ind,
            "fish_name": fish_names[fish_ind],
            "active_neuron_ids": active_neurons_in_fish.copy(),
            "cluster_labels": labels_sorted.copy(),
            "cluster_order": ordered_labels.copy(),
            "label_starts": label_starts.copy(),
            "label_ends": label_ends.copy(),
            "final_roi_indices": final_inds.copy(),
            "cluster_centroids": centroids.copy(),
            "traces_sorted": traces_to_cluster_sorted.copy(),
            "regressors_window": reg_signals[:, start_analyze_frame : start_analyze_frame + len(frame_idx)].copy(),
        }
    )

np.save(
    os.path.join(out_dir, "cluster_results.npy"),
    np.array(cluster_results, dtype=object),
)
#%%
# run affinity propagation
af = AffinityPropagation(preference=-9, damping=0.9, max_iter=500, random_state=1, affinity='precomputed', verbose=True).fit(corr_m)

#%%

cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_
n_clusters = len(cluster_centers_indices)
print(n_clusters)



outline = tifffile.imread('/media/BigBoy/ciqle/ref_brains/ZBrain2_0_outline_proj.tif')
#%%


IM_rois, im_rois_proj = draw_hit_volume(np.arange(len(roi_stats)))
plt.figure(figsize=(20,20))
plt.imshow(im_rois_proj, cmap='inferno')
plt.title('Density of ROIs detected')
plt.axis('off')
plt.show()

#%%
import matplotlib.cm as cm
ref_proj_z = np.max(ref_brain[:,:, :], axis=0)
ref_proj_x = zoom(np.max(ref_brain[:,:, :], axis=2).T, [1, z_rez/xy_rez])
ref_proj = np.hstack((ref_proj_z, ref_proj_x))

def to_rgb(image, cmap_name="gray", vmin=None, vmax=None):
    """Convert scalar image to RGB using a colormap."""
    cmap = cm.get_cmap(cmap_name)
    normed = np.clip((image - (vmin if vmin is not None else image.min())) /
                     ((vmax if vmax is not None else image.max()) -
                      (vmin if vmin is not None else image.min()) + 1e-8), 0, 1)
    return cmap(normed)[..., :3]  # drop alpha channel

common_normalize = True
norm_value = 1 # set max value for normalization across all overlays

IM_rois_regrs = []
im_rois_proj_regrs = []

for regr in range(n_regressors):
    IM_rois, im_rois_proj = draw_hit_volume(inds_hits[regr], normalize=False)
    IM_rois_regrs.append(IM_rois)
    im_rois_proj_regrs.append(im_rois_proj)

    # im_rois_proj = im_rois_proj[:-5, :]


    # normalize
    ref_rgb = to_rgb(ref_proj, cmap_name="gray", vmin=0, vmax=np.percentile(ref_proj, 95))

    if common_normalize:
        rois_rgb = to_rgb(im_rois_proj, cmap_name="magma", vmin=0, vmax=norm_value)
    else:
        rois_rgb = to_rgb(im_rois_proj, cmap_name="magma", vmin=0, vmax=np.percentile(im_rois_proj, 95))

    # weighted additive blending
    w_ref = 0.5   # weight for anatomy
    w_rois = 1.0  # weight for ROI overlay
    blended = np.clip(w_ref * ref_rgb + w_rois * rois_rgb, 0, 1)

    plt.figure(figsize=(20, 20))
    plt.imshow(blended)
    title_str = f"Neurons tuned to {regressor_names[regr]}"
    plt.title(title_str, fontsize=30)
    plt.savefig(os.path.join(out_dir, safe_filename(title_str + '.png')))
    plt.axis("off")
    plt.show()

    tifffile.imwrite(os.path.join(out_dir, safe_filename(title_str + 'stack.tif')), IM_rois)





#%% now do per fish category

corr_thresh = 0.1
inds_hits = []

n_fish_in_category = []

for fish_type in list(fish_dict.keys()):
    n_fish_in_category.append(len(fish_dict[fish_type]))


for regr in range(n_regressors):
    inds_hits.append([])
    for fish_type in [0, 1, 2]: # WT, het, hom
        hits_reg_type = np.where((corrMat[:,regr] >= corr_thresh) & (fish_data[:,1] == fish_type))[0]
        inds_hits[regr].append(hits_reg_type)


common_normalize = True
norm_value = 0.0325 # set max value for normalization across all overlays

IM_rois_regrs = []
im_rois_proj_regrs = []

for regr in range(n_regressors):
    for fish_type in [0, 1, 2]: # WT, het, hom
        if fish_type == 0:
            fish_type_str = 'WT'
        elif fish_type == 1:
            fish_type_str = 'HET'
        else:
            fish_type_str = 'MUT' 

        IM_rois, im_rois_proj = draw_hit_volume(inds_hits[regr][fish_type], normalize=False)
        IM_rois = IM_rois / n_fish_in_category[fish_type] # normalize to number of fish in that category
        im_rois_proj = im_rois_proj / n_fish_in_category[fish_type]
        # im_rois_proj = im_rois_proj[:-5, :]


        # normalize
        ref_rgb = to_rgb(ref_proj, cmap_name="gray", vmin=0, vmax=np.percentile(ref_proj, 95))

        if common_normalize:
            rois_rgb = to_rgb(im_rois_proj, cmap_name="magma", vmin=0, vmax=norm_value)
        else:
            rois_rgb = to_rgb(im_rois_proj, cmap_name="magma", vmin=0, vmax=np.percentile(im_rois_proj, 95))

        # weighted additive blending
        w_ref = 0.5   # weight for anatomy
        w_rois = 1.0  # weight for ROI overlay
        blended = np.clip(w_ref * ref_rgb + w_rois * rois_rgb, 0, 1)

        plt.figure(figsize=(20, 20))
        plt.imshow(blended)
        title_str = f"Neurons tuned to {regressor_names[regr]}, fish type: {fish_type_str}"
        plt.title(title_str, fontsize=30)
        plt.savefig(os.path.join(out_dir, safe_filename(title_str + '.png')))
        plt.axis("off")
        plt.show()

        tifffile.imwrite(os.path.join(out_dir, safe_filename(title_str + 'stack.tif')), IM_rois)


#%% Plot per fish category: std and mean fluorescence

common_normalize = True

n_fish_in_category = [len(fish_dict[ft]) for ft in fish_dict.keys()]
fish_type_labels = ['WT', 'HET', 'MUT']



# Define all metrics to plot
metrics = [
    (np.nansum(abs(F_norm), axis=1), "Sum z_score", "viridis"),
    (np.nanmean(F, axis=1), "Mean_F", "viridis"),
    (np.nanstd(F_dff, axis=1), "Std_dF/F", "viridis"),
]

# --- Find global maximum for each metric ---
global_max_per_metric = []
for measure, _, _ in metrics:
    # For each metric, draw hit volume for all neurons and get max value in projection
    IM_rois, im_rois_proj = draw_hit_volume(np.arange(len(measure)), values=measure, normalize=False)
    # Normalize by total number of fish for fair comparison
    im_rois_proj = im_rois_proj / sum(n_fish_in_category)
    global_max_per_metric.append(np.nanmax(im_rois_proj))

# --- Plot per fish category, normalized to global max ---
stacks_by_measure = {}   # dict -> { measure_name: { 'WT': {'stack': IM_rois, 'proj': im_rois_proj, 'stack_norm': ..., 'proj_norm': ...}, ... } }

for (measure, measure_name, cmap), norm_value in zip(metrics, global_max_per_metric):
    # prepare container for this metric
    stacks_by_measure[measure_name] = {}
    metric_dir = os.path.join(out_dir, 'stacks_by_measure', safe_filename(measure_name))
    os.makedirs(metric_dir, exist_ok=True)

    for fish_type in [0, 1, 2]:  # WT, het, hom
        fish_type_str = fish_type_labels[fish_type]
        inds_type = np.where(fish_data[:,1] == fish_type)[0]
        values = measure[inds_type]

        # produce stacks
        IM_rois, im_rois_proj = draw_hit_volume(inds_type, values=values, normalize=False)

        # normalize by number of fish in category for fair comparison
        IM_rois_norm = IM_rois / n_fish_in_category[fish_type]
        im_rois_proj_norm = im_rois_proj / n_fish_in_category[fish_type]

        # store in dict
        stacks_by_measure[measure_name][fish_type_str] = {
            'stack_raw': IM_rois.astype(np.float32),
            'proj_raw': im_rois_proj.astype(np.float32),
            'stack_norm': IM_rois_norm.astype(np.float32),
            'proj_norm': im_rois_proj_norm.astype(np.float32),
            'values_inds': inds_type,   # ROI indices used
            'values_vec': values.astype(np.float32)  # per-ROI measure values
        }

        # save to disk for downstream analyses
        fname_base = safe_filename(f"{measure_name}_{fish_type_str}")
        np.save(os.path.join(metric_dir, f"{fname_base}_stack_raw.npy"), IM_rois.astype(np.float32))
        np.save(os.path.join(metric_dir, f"{fname_base}_proj_raw.npy"), im_rois_proj.astype(np.float32))
        np.save(os.path.join(metric_dir, f"{fname_base}_stack_norm.npy"), IM_rois_norm.astype(np.float32))
        np.save(os.path.join(metric_dir, f"{fname_base}_proj_norm.npy"), im_rois_proj_norm.astype(np.float32))
        try:
            tifffile.imwrite(os.path.join(metric_dir, f"{fname_base}_stack_norm.tif"), IM_rois_norm.astype(np.float32))
        except Exception:
            pass

        # Normalize overlay for visualization (use metric global norm if available)
        ref_rgb = to_rgb(ref_proj, cmap_name="gray", vmin=0, vmax=np.percentile(ref_proj, 95))
        v_max = norm_value if (norm_value is not None and not np.isnan(norm_value)) else np.nanmax(im_rois_proj_norm)
        rois_rgb = to_rgb(im_rois_proj_norm, cmap_name=cmap, vmin=0, vmax=v_max * 0.3)

        # Weighted additive blending
        w_ref = 0.4
        w_rois = 1.0
        blended = np.clip(w_ref * ref_rgb + w_rois * rois_rgb, 0, 1)

        plt.figure(figsize=(20, 20))
        plt.imshow(blended)
        title_str = f"Neuron {measure_name} fluorescence, fish type: {fish_type_str}"
        plt.title(title_str, fontsize=30)
        plt.savefig(os.path.join(out_dir, safe_filename(title_str + '.png')))
        plt.axis("off")
        plt.show()

        # write normalized stack for quick inspection (also keep original on disk above)
        tifffile.imwrite(os.path.join(metric_dir, f"{fname_base}_stack_norm_inspect.tif"), IM_rois_norm.astype(np.float32))
#%%

from scipy.ndimage import gaussian_filter, zoom as ndi_zoom

# 3D gaussian blur params (radii in pixels)
blurr_size = 30 # in microns
blur_radius = (blurr_size/z_rez, blurr_size/xy_rez, blurr_size/xy_rez)   # (Z, Y, X) as you requested ~20x20x10 px (Z first)
truncate = 4.0
sigma = tuple(r / truncate for r in blur_radius)

comparisons_dir = os.path.join(out_dir, "group_comparisons")
os.makedirs(comparisons_dir, exist_ok=True)

def project_stack(stack, proj_mean=True):
    # stack: (Z, H, W) -> produce same 2D projection used elsewhere (axial + sagittal hstack)
    if proj_mean:
        im_proj_z = np.mean(stack, axis=0)
        im_proj_x = ndi_zoom(np.mean(stack, axis=2).T, [1, z_rez/xy_rez])
    else:
        im_proj_z = np.max(stack, axis=0)
        im_proj_x = ndi_zoom(np.max(stack, axis=2).T, [1, z_rez/xy_rez])
    # avoid division by zero
    zmax = np.nanmax(im_proj_z) if np.nanmax(im_proj_z) != 0 else 1.0
    xmax = np.nanmax(im_proj_x) if np.nanmax(im_proj_x) != 0 else 1.0
    return np.hstack((im_proj_z / zmax, im_proj_x / xmax))

# compute blurred stacks and pairwise diffs; plot subplots:
pairs = [("HET", "WT"), ("MUT", "WT"), ("MUT", "HET")]
groups_order = ("WT","HET","MUT")

# create magenta -> black -> green diverging colormap for diffs
from matplotlib.colors import LinearSegmentedColormap

cmap_mag_black_green = LinearSegmentedColormap.from_list(
    "mag_black_green", ["magenta", "black", "green"], N=256
)
cmap_mag_black_green.set_bad("black")

for measure_name, groups in stacks_by_measure.items():
    # ensure storage containers
    groups.setdefault("blurred", {})
    groups.setdefault("diffs", {})

    # compute blurred + normalized stacks per group
    proj_dict = {}
    for g in groups_order:
        if g not in groups:
            continue
        stack_in = groups[g].get("stack_norm", groups[g]["stack_raw"]).astype(np.float32)
        # apply gaussian blur
        blurred = gaussian_filter(stack_in, sigma=sigma, truncate=truncate)

        groups["blurred"][g] = blurred
        # compute 2D projection for plotting
        proj_dict[g] = project_stack(blurred, proj_mean=True)
        # save blurred stacks
        np.save(os.path.join(comparisons_dir, f"{safe_filename(measure_name)}_{g}_blurred.npy"), blurred.astype(np.float32))
        try:
            tifffile.imwrite(os.path.join(comparisons_dir, f"{safe_filename(measure_name)}_{g}_blurred.tif"), blurred.astype(np.float32))
        except Exception:
            pass

    # compute pairwise diffs (blurred a - blurred b) and projections
    diff_projs = {}
    for a,b in pairs:
        if a not in groups["blurred"] or b not in groups["blurred"]:
            continue
        diff = (groups["blurred"][a] - groups["blurred"][b]).astype(np.float32)
        key = f"{a}_minus_{b}"
        groups["diffs"][key] = diff
        diff_projs[key] = project_stack(diff, proj_mean=True)
        # save diffs
        np.save(os.path.join(comparisons_dir, f"{safe_filename(measure_name)}_{key}.npy"), diff)
        try:
            tifffile.imwrite(os.path.join(comparisons_dir, f"{safe_filename(measure_name)}_{key}.tif"), diff)
        except Exception:
            pass

