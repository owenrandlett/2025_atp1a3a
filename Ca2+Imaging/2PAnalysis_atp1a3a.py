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


#%%
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
    

    tail_power_conv = GCaMPConvolve(resample_to_reference(behav_time, tail_power, microscope_timestamps), KerTotal)
    
    swim_bursting = medfilt(tail_power, frame_rate_behav*20+1)
    swim_bursting_conv = GCaMPConvolve(resample_to_reference(behav_time, swim_bursting, microscope_timestamps), KerTotal)
    
    lowpass_orients = medfilt(orients_filt, frame_rate_behav*13+1)
    lowpass_orients_conv = GCaMPConvolve(resample_to_reference(behav_time, lowpass_orients, microscope_timestamps), KerTotal)

    regressors = np.vstack((
        stim_df_conv, 
        stim_omr_conv,
        tail_power_conv,
        swim_bursting_conv,
        lowpass_orients_conv
    ))


    n_regressors = regressors.shape[0]
    corrMat_temp = np.zeros([nROIs, n_regressors])
    for regr in range(n_regressors):
        corrMat_temp[:, regr] = pearsonr_vec_2Dnumb(regressors[regr, start_analyze_frame:], F_norm_fish[:, start_analyze_frame:])

    corrMat_temp[np.isnan(corrMat_temp)] = 0 # set invalid correlations to 0

    if fish_ind == 0:
        corrMat = np.copy(corrMat_temp)
    else:
        corrMat = np.vstack((corrMat, corrMat_temp))

#%%
    
corr_thresh = 0.1
inds_hits = []
for regr in range(n_regressors):
    inds_hits.append(np.where(corrMat[:,regr] >= corr_thresh)[0])
    plt.plot(np.mean(F_norm[inds_hits[regr], :], axis=0), label=regressor_names[regr])
plt.ylabel('Mean z-scored fluorescence of ROIs with\ncorrelation > ' + str(corr_thresh))
plt.xlabel('Frame number')
plt.title('Mean activity of ROIs correlated with each regressor')

plt.legend()
#%%
import tifffile
from scipy.ndimage import zoom, morphology
import nrrd

ref_brain_path = r'/media/FastDrive/atp1a3a_data/registration/telen_template_allfish_HuC-H2BGCaMP.nrrd' 
ref_brain, ref_meta = nrrd.read(ref_brain_path)
width, height, Zs = ref_brain.shape
ref_brain = np.moveaxis(ref_brain, [0,1,2], [2,1,0])


xy_rez = ref_meta['space directions'][0][0]
z_rez = ref_meta['space directions'][-1][-1]

outline = tifffile.imread('/media/BigBoy/ciqle/ref_brains/ZBrain2_0_outline_proj.tif')
#%%
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

for measure, measure_name, cmap, norm_value in [
    (np.std(F, axis=1), "STD", "viridis", None),      # std of raw fluorescence
    (np.mean(F, axis=1), "Mean", "plasma", None)      # mean of raw fluorescence
]:
    for fish_type in [0, 1, 2]:  # WT, het, hom
        fish_type_str = fish_type_labels[fish_type]
        # Get indices for this fish type
        inds_type = np.where(fish_data[:,1] == fish_type)[0]
        # Get measure values for these neurons
        values = measure[inds_type]
        # Draw hit volume weighted by measure
        IM_rois, im_rois_proj = draw_hit_volume(inds_type, values=values, normalize=False)
        IM_rois = IM_rois / n_fish_in_category[fish_type]
        im_rois_proj = im_rois_proj / n_fish_in_category[fish_type]

        # Normalize overlay
        ref_rgb = to_rgb(ref_proj, cmap_name="gray", vmin=0, vmax=np.percentile(ref_proj, 95))
        if common_normalize:
            rois_rgb = to_rgb(im_rois_proj, cmap_name=cmap, vmin=0, vmax=np.percentile(im_rois_proj, 99))
        else:
            rois_rgb = to_rgb(im_rois_proj, cmap_name=cmap, vmin=0, vmax=np.percentile(im_rois_proj, 95))

        # Weighted additive blending
        w_ref = 0.5
        w_rois = 1.0
        blended = np.clip(w_ref * ref_rgb + w_rois * rois_rgb, 0, 1)

        plt.figure(figsize=(20, 20))
        plt.imshow(blended)
        title_str = f"Neuron {measure_name} fluorescence, fish type: {fish_type_str}"
        plt.title(title_str, fontsize=30)
        plt.savefig(os.path.join(out_dir, safe_filename(title_str + '.png')))
        plt.axis("off")
        plt.show()

        tifffile.imwrite(os.path.join(out_dir, safe_filename(title_str + 'stack.tif')), IM_rois)


#%%
#% crop to imaged volume
n_rois_min = 10
x_trace = np.max(np.max(IM_rois, axis=0) , axis=0)>n_rois_min
y_trace = np.max(np.max(IM_rois, axis=0) , axis=1)>n_rois_min
z_trace =  np.max(np.max(IM_rois, axis=1) , axis=1)>n_rois_min
#plt.plot(z_trace)

xlims = np.where(x_trace)[0][0], np.where(x_trace)[0][-1]
print(xlims)

ylims = np.where(y_trace)[0][0], np.where(y_trace)[0][-1]
print(ylims)

zlims = np.where(z_trace)[0][0], np.where(z_trace)[0][-1]
print(zlims)

zbrain_crop = zbrain[zlims[0]:zlims[1], ylims[0]:ylims[1], xlims[0]:xlims[1]]
Zs_crop, height_crop, width_crop = zbrain_crop.shape


zbrain_outline_z = np.zeros((height_crop, width_crop))
zbrain_outline_x = zoom(np.zeros((Zs_crop, height_crop)).T, [1, 2/0.798])
mask_3d = np.zeros(zbrain_crop.shape)
mask_3d[:] = 0


IDs = [
    #np.where((zbrain_crop >=27) & (zbrain_crop <= 28)), # thalamus
    np.where((zbrain_crop >=29) & (zbrain_crop <= 31)), # telencephalon
    np.where((zbrain_crop >=48) & (zbrain_crop <= 110) | (zbrain_crop == 119)), # hindbrain
    np.where((zbrain_crop >=111) & (zbrain_crop <= 112)), # tectum
    np.where((zbrain_crop == 50) ), # inferior olive
    np.where((zbrain_crop == 23) ), # pretectum  
    #np.where((zbrain_crop == 114) ), # nucMLF
    #np.where((zbrain_crop == 70) |  (zbrain_crop == 71) | (zbrain_crop == 72)| (zbrain_crop == 77) | (zbrain_crop == 78)| (zbrain_crop == 79) | (zbrain_crop == 91)| (zbrain_crop == 94)| (zbrain_crop == 95)| (zbrain_crop == 101)| (zbrain_crop == 102) | (zbrain_crop == 107)| (zbrain_crop == 108)), # reticulospinal 
    #np.where((zbrain_crop >=84) & (zbrain_crop <= 89)), # more RS
    #np.where((zbrain_crop == 79) | (zbrain_crop == 89) | (zbrain_crop == 95) ), # v cells

]

for ids in IDs:
    mask_3d = np.zeros(zbrain_crop.shape)
    mask_3d[ids] = 1
    mask = np.max(mask_3d, axis=0)
    outline = morphology.distance_transform_edt(1-mask) == 1
    #outline = morphology.binary_dilation(outline, iterations=1)
    zbrain_outline_z[outline==1] =1

    mask = zoom(np.max(mask_3d, axis=2).T, [1, 2/0.798], order=0)
    outline = morphology.distance_transform_edt(1-mask) == 1
    #outline = morphology.binary_dilation(outline, iterations=1)
    zbrain_outline_x[outline==1] =1




proj = np.hstack((zbrain_outline_z, zbrain_outline_x))
proj = proj * 2
proj = proj.astype(np.uint8)
proj[proj>0] = 255
proj[proj < 255] = 0

# tifffile.imsave('/media/BigBoy/ciqle/ref_brains/ZBrain2_0_outline_proj_areas_crop.tif', data=proj)
outline = proj
IM_rois, im_rois_proj = draw_hit_volume(np.arange(len(roi_stats)), draw_outline=True, save_name = 'roi_density_cropped')
plt.figure(figsize=(20,20))
plt.imshow(im_rois_proj, cmap='inferno')
plt.title('Density of ROIs detected')
plt.axis('off')
IM_rois=None


fish_names = []
for fish_name in ops.keys():
    fish_names.append(fish_name)
