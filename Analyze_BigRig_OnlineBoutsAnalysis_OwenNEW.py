#%%

# warning -- this untested for a full 2 plate exeriment, but should work!
import FishTrack, h5py, os, glob, pickle, gspread, sys, types
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas.core.indexes.base as idx_base
idx_base.Int64Index = pd.Index

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu
import itertools


from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
from scipy.stats import mannwhitneyu, kruskal
current_dir = os.path.dirname(__file__)

sys.path.append(current_dir)
sys.path.append(os.path.realpath(current_dir + r'/ExtraFunctions/glasbey-master/'))

from glasbey import Glasbey


#%
big_rig3 = True
#%
exp_dir = r'/media/BigBoy/MultiTracker/20250113_113921_atp1a3a_BR2_Phin'
# exp_dir = r'/media/BigBoy/MultiTracker/20250127_110626_atp1a3a_BR3_Phin'

os.chdir(exp_dir)

graph_dir = os.path.join(exp_dir, 'boutsAnalysis_graphs')

if not os.path.exists(graph_dir):
    os.mkdir(graph_dir)
    

camera_rez = 180/2290 # camera resolution in mm/pixel
    
exp_data_file =glob.glob(exp_dir+'/exp_data.pkl')[0]
online_tracking_file = glob.glob(exp_dir+'/online_tracking.hdf5')[0]
plate_tracking_data = glob.glob(exp_dir+'/*trackdata_twoMeasures.pkl')
n_plates = len(plate_tracking_data)
#%
with open(exp_data_file, 'rb') as f:
    exp_data = pickle.load(f)


#%
online_tracking = h5py.File(online_tracking_file)
print(list(online_tracking.keys()))
#%
# load datasets from hdf5 file
frames_plate = np.array(online_tracking['plate']) # 0 or 1, depending on if its tracking plate 0 or plate 1
t_stamps = np.array(online_tracking['time_stamp']) # clock time of the experiment
t_stamps_intervals = np.diff(t_stamps) # time intervals between frames
t_stamps_hrs = (t_stamps-t_stamps[0])/(60*60)
t_stamps_min = t_stamps_hrs*60
t_stamps_sec = t_stamps_min*60

frame_rate_tracking = 1/np.mean(np.diff(t_stamps)) # frame rate of the tracking data
frame_idex = np.array(online_tracking['frame_index']) # frame index of the experiment
camera_fps = np.mean(np.diff(frame_idex))/np.mean(np.diff(t_stamps)) # caluclating the FPS of that experiment



coords = online_tracking['tail_coords'] # tracking coordinates. 4d array: order = [timepoints, x/y, fish, tail coordinate]
head_coords = coords[:,:,:,0]
print("DATA loaded")



plates = np.arange(n_plates)

gb = Glasbey()


# Old pickles may reference pandas.core.indexes.numeric.Int64Index
numeric_mod = types.ModuleType("pandas.core.indexes.numeric")
numeric_mod.Int64Index = pd.Index  # alias to modern Index
sys.modules["pandas.core.indexes.numeric"] = numeric_mod


rois = []
names = []
stim_frame_inds = []
stim_given = []

curv_events = []
for plate in range(n_plates):
    with open(plate_tracking_data[plate], 'rb') as f:
        tracking_data = pickle.load(f)
        names.append(tracking_data['names'])
        rois.append(tracking_data['rois'])
        stim_frame_inds.append(tracking_data['TiffFrameInds'])
        stim_given.append( tracking_data['stim_given'])
        curv_events.append(tracking_data['MaxCurvatureOBendEvents'])


# note that this logic assumes the standard "big rig" habituation assay protocol
plot_frame_interval_names = [
    'Acclimitaization Period',
    'Block 1',
    'Block 2',
    'Block 3',
    'Block 4',
    'Free Swimming',
    'Taps',
    'Motion stimulus',
    'Block 5',
] 

plot_frame_intervals = []
for plate in range(n_plates):
    plot_frame_intervals.append([
        (np.where(frames_plate == plate)[0][0], np.where(frame_idex >= stim_frame_inds[plate][0])[0][0]), # acclim
        (np.where(frame_idex >= stim_frame_inds[plate][0]- camera_fps*60)[0][0], np.where(frame_idex >= stim_frame_inds[plate][59] +  camera_fps*60)[0][0]),
        (np.where(frame_idex >= stim_frame_inds[plate][60] - camera_fps*60)[0][0], np.where(frame_idex >= stim_frame_inds[plate][119] +  camera_fps*60)[0][0]),
        (np.where(frame_idex >= stim_frame_inds[plate][120] - camera_fps*60)[0][0], np.where(frame_idex >= stim_frame_inds[plate][179] +  camera_fps*60)[0][0]),
        (np.where(frame_idex >= stim_frame_inds[plate][180] - camera_fps*60)[0][0], np.where(frame_idex >= stim_frame_inds[plate][239] +  camera_fps*60)[0][0]),
        (np.where(frame_idex >= stim_frame_inds[plate][240] - camera_fps*60*25)[0][0], np.where(frame_idex >= stim_frame_inds[plate][240] - camera_fps)[0][0]),
        (np.where(frame_idex >= stim_frame_inds[plate][240] - camera_fps*60)[0][0], np.where(frame_idex >= stim_frame_inds[plate][269] +  camera_fps*60)[0][0]),
        (np.where(frame_idex >= stim_frame_inds[plate][270])[0][0], np.where(frame_idex >= stim_frame_inds[plate][270] +  camera_fps*60*59)[0][0]),
        (np.where(frame_idex >= stim_frame_inds[plate][271] - camera_fps*60)[0][0], np.where(frame_idex >= stim_frame_inds[plate][330] +  camera_fps*1)[0][0]),
    ])




plt.rcParams.update({
    'font.size': 30,           # Increase overall font size
    'axes.titlesize': 36,      # Title font size
    'axes.labelsize': 32,      # Axis label font size
    'xtick.labelsize': 28,     # X tick label font size
    'ytick.labelsize': 28,     # Y tick label font size
    'legend.fontsize': 28      # Legend font size
})

sav_sz_sec = 0.3  # sav_sz_sec is the size of the window to smooth over in seconds
sav_sz = np.round(frame_rate_tracking*sav_sz_sec).astype(int) 
speeds = FishTrack.get_speeds(head_coords, sav_sz=sav_sz, sav_ord=1, speed_thresh=4) 
speeds = speeds*camera_rez # convert speeds to mm/frame

speeds = speeds/t_stamps_intervals[:, None] # convert speeds to mm/s

#%

speed_thresh = 0.5
bouts = speeds > speed_thresh # binary array where 1 is when the fish is "in a bout". This could be used to find bouts starts and ends, for example use difference of this. 


bout_st_end = np.diff(bouts.astype(int), axis=0)
bout_starts = []
bout_ends = []

    
for i in range(bouts.shape[1]):
    bout_starts.append(np.where(bout_st_end[:, i] > 0)[0])
    bout_ends.append(np.where(bout_st_end[:, i] < 0)[0])

#%
print("Speeds calculated")
#%%


speed_thresh_start = 1
speed_thresh_end = 0.5


bouts_start = speeds > speed_thresh_start
below_end = speeds < speed_thresh_end  # (n_frames, n_fish)

bout_st_array = np.diff(bouts_start.astype(int), axis=0)

min_gap = 2  # frames; merge bouts if next start is within this many frames of previous end

bout_starts = []
bout_ends = []

for i in range(speeds.shape[1]):
    # Rising edges = bout starts
    starts = np.where(bout_st_array[:, i] > 0)[0] + 1

    # Candidate ends
    end_candidates = np.where(below_end[:, i])[0]

    # Match each start to the first end after it
    ends_idx = np.searchsorted(end_candidates, starts)
    valid = ends_idx < len(end_candidates)

    starts = starts[valid]
    ends   = end_candidates[ends_idx[valid]]

    # --- enforce 1-to-1 sequential bouts ---
    matched_starts = []
    matched_ends = []
    last_end = -1

    for st, en in zip(starts, ends):
        if st > last_end:   # only accept if bout has closed
            matched_starts.append(st)
            matched_ends.append(en)
            last_end = en

    # --- enforce minimum inter-bout gap (merge close bouts) ---
    merged_starts = []
    merged_ends = []
    for st, en in zip(matched_starts, matched_ends):
        if len(merged_starts) == 0:
            merged_starts.append(st)
            merged_ends.append(en)
        else:
            if st - merged_ends[-1] <= min_gap:
                # merge with previous bout: extend its end
                merged_ends[-1] = en
            else:
                # start a new bout
                merged_starts.append(st)
                merged_ends.append(en)

    bout_starts.append(np.array(merged_starts, dtype=int))
    bout_ends.append(np.array(merged_ends, dtype=int))

## plot some data with bout annotations for testing thresholds

# for fish in rois[0][0][:10]:

#     bout_min_len = 0.05 # minimum bout length in seconds
#     bout_min_frames = bout_min_len * frame_rate_tracking

#     if bout_ends[fish][0] < bout_starts[fish][0]:
#         bout_ends[fish] = bout_ends[fish][1:]
#     if bout_starts[fish][-1] > bout_ends[fish][-1]:
#         bout_starts[fish] = bout_starts[fish][:-1]

#     bout_len = bout_ends[fish] - bout_starts[fish]

#     # plt.hist(ISI, bins=100)
#     #%
#     st = 580539
#     end = st + 1000

#     bouts_st_fish = np.zeros(speeds[:, fish].shape)
#     bouts_end_fish = np.zeros(speeds[:, fish].shape)
#     bouts_st_fish[:] = np.nan
#     bouts_end_fish[:] = np.nan
#     bouts_st_fish[bout_starts[fish]] = 1
#     bouts_end_fish[bout_ends[fish]] = 1
#     plt.figure(figsize=(30, 10))
#     plt.title('Fish ' + str(fish) + '; sav_size ' + str(sav_sz_sec) + ' seconds')
#     plt.plot(t_stamps_sec[st:end], speeds[st:end, fish])
#     plt.plot(t_stamps_sec[st:end], bouts_st_fish[st:end]*speed_thresh_start, 'og', markersize=10)
#     plt.plot(t_stamps_sec[st:end], bouts_end_fish[st:end]*speed_thresh_end, 'or', markersize=10)
#     plt.ylim([0, 10])
#     plt.show()

#%

all_bouts = []

for fish_id in range(speeds.shape[1]):
    starts = bout_starts[fish_id]
    ends   = bout_ends[fish_id]

    for st, en in zip(starts, ends):
        # displacement = integral of speed * dt
        displacement = np.sum(speeds[st:en, fish_id] * t_stamps_intervals[st:en])

        # duration in seconds
        duration = np.sum(t_stamps_intervals[st:en])

        # peak speed in this bout
        peak_speed = np.max(speeds[st:en+1, fish_id])

        all_bouts.append({
            "fish_id": fish_id,
            "bout_start_frame": st,
            "bout_end_frame": en,
            "bout_start_time": t_stamps[st],
            "bout_end_time": t_stamps[en],
            "duration": duration,
            "displacement": displacement,
            "peak_speed": peak_speed,
        })

bout_df = pd.DataFrame(all_bouts)

#%

bout_df["plate"] = bout_df["bout_start_frame"].map(lambda f: frames_plate[f])




cross_plate = frames_plate[bout_df["bout_start_frame"].values] != frames_plate[bout_df["bout_end_frame"].values]

if cross_plate.any():
    print("Warning: some bouts span across plates! Dropping them.")
    bout_df = bout_df.loc[~cross_plate].copy()


plate_filename = plate_tracking_data[plate]
if not plate_filename.find('_plate_0') == -1:
    bout_df_plate = bout_df[bout_df["plate"] == 0].copy()
else:
    bout_df_plate = bout_df[bout_df["plate"] == 1].copy()



#%

for plate in range(n_plates):
    # --- Step 3: assign group labels ---
    fish_to_group = {}
    for group_name, fish_inds in zip(names[plate], rois[plate]):
        # fish_inds could be list or array, possibly nested
        for f in fish_inds:  
            # f might still be an array if nested, so flatten manually
            if isinstance(f, (list, np.ndarray)):
                for ff in f:
                    fish_to_group[int(ff)] = group_name
            else:
                fish_to_group[int(f)] = group_name

    # Assign group names to your DataFrame
    bout_df_plate["group"] = bout_df_plate["fish_id"].map(fish_to_group)
    #%


    intervals = plot_frame_intervals[plate]
    intervals = [(int(s), int(e)) for s, e in intervals]

    def assign_period(frame):
        for pi, (start, end) in enumerate(intervals):
            if start <= frame <= end:
                return pi
        return None

    bout_df_plate["period"] = bout_df_plate["bout_start_frame"].apply(assign_period)


    #%
    # --- Step 6: compute per-fish averages and totals per period ---
    per_fish = (
        bout_df_plate.groupby(["fish_id", "group", "period"])
        .agg(
            mean_disp=("displacement", "mean"),
            median_disp=("displacement", "median"),
            total_disp=("displacement", "sum"),
            mean_duration=("duration", "mean"),
            median_duration=("duration", "median"),
            mean_peak=("peak_speed", "mean"),
            median_peak=("peak_speed", "median"),
            n_bouts=("fish_id", "count")
        )
        .reset_index()
    )


    # Map period index to name
    period_names_map = {i: name for i, name in enumerate(plot_frame_interval_names)}
    per_fish["period_name"] = per_fish["period"].map(period_names_map)

    # --- Compute period length using t_stamps vector ---
    period_lengths = {}
    for i, (start, end) in enumerate(intervals):
        t_start = t_stamps[start]
        t_end   = t_stamps[end]
        period_lengths[i] = t_end - t_start

        
    # Map period length to each fish
    per_fish["period_length_s"] = per_fish["period"].map(period_lengths)

    # Calculate bout frequency in Hz
    per_fish["bout_freq_hz"] = per_fish["n_bouts"] / per_fish["period_length_s"]

    # --- Group order and palette ---
    group_order = names[plate]  # list of group names for this plate

    p = gb.generate_palette(size=len(names[plate])+2) 
    col_vec = gb.convert_palette_to_rgb(p) 
    col_vec = np.array(col_vec[1:], dtype=float)/255
    group_palette = {g: tuple(col_vec[i]) for i, g in enumerate(group_order)}



    def stripplot_period_full_grid(period_name, global_y_lims=None, use_median = True):

            # --- Statistics to plot ---
        if use_median:
            stats_to_plot = [
                ("median_disp", "Median displacement per bout (mm)"),
                ("median_duration", "Median bout duration (s)"),
                ("median_peak", "Median peak bout speed (mm/s)"),
                ("total_disp", "Total displacement (mm)"),  # total is always a sum
                # ("n_bouts", "Number of bouts"),
                ("bout_freq_hz", "Bout frequency (Hz)")
            ]
        else:
            stats_to_plot = [
                ("mean_disp", "Mean displacement per bout"),
                ("mean_duration", "Mean bout duration (s)"),
                ("mean_peak", "Mean peak speed"),
                ("total_disp", "Total displacement"),
                # ("n_bouts", "Number of bouts"),
                ("bout_freq_hz", "Bout frequency (Hz)")
            ]
        data = per_fish[per_fish["period_name"] == period_name].copy()
        
        # --- Prepare text file for stats ---
        safe_period_name = period_name.replace(" ", "_")
        stats_filename = f"{graph_dir}/MovementStats_Plate{plate}_{safe_period_name}.txt"
        stats_file = open(stats_filename, 'w')
        stats_file.write(f"Statistical results for {period_name}, Plate {plate}\n\n")
        
        n_stats = len(stats_to_plot)
        n_cols = 2
        n_rows = 3
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 18))
        axes = axes.flatten()
        
        for ax, (stat, ylabel) in zip(axes, stats_to_plot):
            # Kruskal-Wallis test
            groups_data = [data[data["group"] == g][stat].values for g in group_order]
            kw_stat, kw_p = kruskal(*groups_data)
            
            # --- Write KW result to text file ---
            stats_file.write(f"{stat}:\n")
            stats_file.write(f"  Kruskal-Wallis H = {kw_stat:.3f}, p = {kw_p:.4f}\n")
            
            # --- Write group means to text file ---
            medians = data.groupby("group")[stat].median()
            stats_file.write("  Group medians:\n")
            for g in group_order:
                stats_file.write(f"    {g}: {medians[g]:.3f}\n")
            
            # Stripplot
            sns.stripplot(
                x="group",
                y=stat,
                data=data,
                order=group_order,
                palette=group_palette,
                jitter=0.3,
                alpha=0.7,
                marker="o",
                size=8,
                ax=ax
            )
            
            # Overlay group medians
            for i, g in enumerate(group_order):
                y = medians[g]
                ax.hlines(y=y, xmin=i-0.3, xmax=i+0.3, color=group_palette[g], linewidth=7)
            
            # Pairwise significance annotations
            if kw_p < 0.05:
                n_comparisons = len(group_order) * (len(group_order)-1) / 2
                sig_annotations = []
                for g1, g2 in itertools.combinations(group_order, 2):
                    u_stat, p_val = mannwhitneyu(
                        data[data["group"]==g1][stat], 
                        data[data["group"]==g2][stat],
                        alternative='two-sided'
                    )
                    p_val_corr = min(p_val * n_comparisons, 1.0)
                    if p_val_corr < 0.05:
                        if p_val_corr < 0.001:
                            p_text = "p<0.001"
                        elif p_val_corr < 0.01:
                            p_text = "p<0.01"
                        else:
                            p_text = "p<0.05"
                        sig_annotations.append((g1, g2, p_text))
                        # Write pairwise result to text file
                        stats_file.write("         Pairwise Mann Whitney Sig. Results: \n")
                        stats_file.write(f"    {g1} vs {g2}: p = {p_val_corr}\n")
                # Annotate significance on plot
                y_max = data[stat].max()
                y_offset = y_max * 0.1
                for j, (g1, g2, p_text) in enumerate(sig_annotations):
                    x1 = group_order.index(g1)
                    x2 = group_order.index(g2)
                    y = y_max + y_offset * (j+1)
                    ax.plot([x1, x1, x2, x2], [y, y+y_offset/4, y+y_offset/4, y], color='black')
                    ax.text((x1+x2)/2, y+y_offset/4, p_text, ha='center', va='bottom', fontsize=10)
            
            # Formatting
            if global_y_lims and stat in global_y_lims:
                ax.set_ylim(global_y_lims[stat])
            else:
                ax.set_ylim(bottom=0)
            ax.set_xticks(range(len(group_order)))
            ax.set_xticklabels(group_order, fontsize=14, rotation=45, ha='right')
            ax.set_xlabel("")
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels([f"{y:.2f}" for y in ax.get_yticks()], fontsize=10)
            ax.set_ylabel(ylabel, fontsize=16)
            # ax.set_title(stat.replace('_',' ').title(), fontsize=16)

        
        # Hide unused subplots
        for ax in axes[n_stats:]:
            ax.axis('off')
        
        fig.suptitle(f"{period_name}", fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure
        fig_title = f"Plate{plate}_{safe_period_name}"
        plt.savefig(f"{graph_dir}/MovementGraphs_{fig_title}.png")
        plt.savefig(f"{graph_dir}/MovementGraphs_{fig_title}.svg")
        plt.show()
        
        # Close stats file
        stats_file.close()


    # # --- Generate figures for all selected periods ---
    for period in plot_frame_interval_names:
        stripplot_period_full_grid(period)



    # global_y_lims = {}
    # for stat, _ in stats_to_plot:
    #     global_y_lims[stat] = (0, per_fish[stat].max())
    #     print(f"{stat}: y-lim = {global_y_lims[stat]}")




#%% 
# analysis of aligned bouts

from scipy.ndimage import median_filter

# --- Parameters ---
pre_window_sec = 0.5
post_window_sec = 1
pre_window_frames = int(pre_window_sec / np.mean(t_stamps_intervals))
post_window_frames = int(post_window_sec / np.mean(t_stamps_intervals))
time_axis = np.arange(-pre_window_frames, post_window_frames) * np.mean(t_stamps_intervals)
time_axis = time_axis + 0.2 # adjust for bout detection lag

# --- Bootstrapped CI function ---
def bootstrap_ci(data, n_boot=1000, ci=95):
    """Compute bootstrapped confidence interval along axis=0, ignoring NaNs."""
    n_fish, n_time = data.shape
    boot_medians = np.full((n_boot, n_time), np.nan)
    for b in range(n_boot):
        sample_idx = np.random.choice(n_fish, size=n_fish, replace=True)
        sample = data[sample_idx, :]
        boot_medians[b, :] = np.nanmedian(sample, axis=0)
    lower = np.percentile(boot_medians, (100-ci)/2, axis=0)
    upper = np.percentile(boot_medians, 100-(100-ci)/2, axis=0)
    return lower, upper

# --- Prepare group colors using Glasbey palette ---
unique_groups = names[plate]
p = gb.generate_palette(size=len(unique_groups)+2)
col_vec = gb.convert_palette_to_rgb(p)
col_vec = np.array(col_vec[1:], dtype=float)/255
group_palette = {g: tuple(col_vec[i]) for i, g in enumerate(unique_groups)}

# --- Loop over periods ---
n_periods = len(plot_frame_interval_names)
fig, axes = plt.subplots(nrows=n_periods, ncols=1, figsize=(18, 9*n_periods), sharex=True)

for period_idx, period_name in enumerate(plot_frame_interval_names):
    aligned_speed_per_fish = []

    # Align bouts for this period
    for fish_id in range(speeds.shape[1]):
        bout_mask = (bout_df_plate["fish_id"] == fish_id) & \
                    (bout_df_plate["period"] == period_idx)
        bouts_fish = bout_df_plate.loc[bout_mask]

        aligned_bouts = []

        if bouts_fish.empty:
            aligned_bouts.append(np.full(pre_window_frames + post_window_frames, np.nan))

        for _, bout in bouts_fish.iterrows():
            st_frame = bout["bout_start_frame"]
            win_start = st_frame - pre_window_frames
            win_end   = st_frame + post_window_frames

            if win_start > 0 and win_end < speeds.shape[0]:
                bout_trace = median_filter(speeds[win_start:win_end, fish_id], size=5, mode='nearest')
                aligned_bouts.append(bout_trace)

        aligned_speed_per_fish.append(np.nanmedian(np.array(aligned_bouts), axis=0))

    aligned_speed_per_fish = np.array(aligned_speed_per_fish)

    ax = axes[period_idx] if n_periods > 1 else axes

    legend_labels = []
    # --- Plot groups ---
    for group_name, fish_ids in zip(names[plate], rois[plate]):
        group_idx = list(fish_ids)
        group_data = aligned_speed_per_fish[group_idx, :]

        # Only keep fish with at least one valid bout
        valid_fish_mask = ~np.isnan(group_data).all(axis=1)
        group_data_valid = group_data[valid_fish_mask, :]
        N_fish = group_data_valid.shape[0]

        if N_fish == 0:
            continue

        median_trace = np.nanmedian(group_data_valid, axis=0)
        ci_lower, ci_upper = bootstrap_ci(group_data_valid, n_boot=1000, ci=95)

        ax.plot(time_axis, median_trace, 
                color=group_palette[group_name], lw=2)
        ax.fill_between(time_axis, ci_lower, ci_upper, color=group_palette[group_name], alpha=0.3)

        legend_labels.append(f"{group_name} (N={N_fish})")

    # ax.axvline(0, color='k', ls='--', lw=1.5)
    ax.set_ylabel("Speed (mm/s)")
    ax.set_title(f"{period_name}")
    ax.legend(legend_labels, loc='upper right')

axes[-1].set_xlabel("Time relative to bout start (s)")
plt.tight_layout()


plt.savefig(os.path.join(graph_dir, f"aligned_bouts_plate{plate}_all_periods.png"), dpi=300)
plt.savefig(os.path.join(graph_dir, f"aligned_bouts_plate{plate}_all_periods.svg"), dpi=300)
plt.show()

#%%

