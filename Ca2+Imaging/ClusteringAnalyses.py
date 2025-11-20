#%%

import numpy as np
import os
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import tqdm
import Ca2ImagingFns

raw_data_fldrs_path = r'/media/BigBoy/ciqle/2p/20250902-11_atp1a3a_experiments'
processed_data_flds_path = r'/media/FastDrive/atp1a3a_data'

# reload all fish data
all_fish_data = np.load(os.path.join(processed_data_flds_path, 'ImagingData_allFish.npz'), allow_pickle=True)
ops = all_fish_data['ops'].item()

out_dir = os.path.join(processed_data_flds_path, 'Outputs_2pAnalysis')

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

cluster_results = np.load(os.path.join(out_dir, "cluster_results.npy"),allow_pickle=True)
#%%

out_dir_orderedheatmaps = os.path.join(out_dir, 'clustered_ordered_heatmaps')
os.makedirs(out_dir_orderedheatmaps, exist_ok=True)

selected_regressor_indices = [0, 1, 2]
selected_regressor_names = [regressor_names[i] for i in selected_regressor_indices]

#%%

out_dir_regressor_association = os.path.join(out_dir, 'regressor_cluster_association')
os.makedirs(out_dir_regressor_association, exist_ok=True)

start_analyze_frame = 200
regressor_associated_signals = []
fish_ind = 7
for fish_ind in tqdm.tqdm(range(len(cluster_results))):
    fish_name = cluster_results[fish_ind]['fish_name']

                            # current fish identifier
    print(fish_name)
    active_neurons_in_fish = cluster_results[fish_ind]['active_neuron_ids']      # ROI indices used for clustering
    traces_sorted = cluster_results[fish_ind]['traces_sorted']                   # activity traces sorted by cluster order
    labels_sorted = cluster_results[fish_ind]['cluster_labels']                  # cluster label per sorted trace
    label_starts = cluster_results[fish_ind]['label_starts']                     # start index of each cluster block
    label_ends = cluster_results[fish_ind]['label_ends']                         # end index of each cluster block
    final_roi_indices = cluster_results[fish_ind]['final_roi_indices']           # ROI indices in the sorted order
    centroids = cluster_results[fish_ind]['cluster_centroids']                   # centroid trace for each cluster
    fish_type = cluster_results[fish_ind]['fish_type']                           # genotype / condition metadata
    print(fish_type)



    reg_signals = np.asarray(cluster_results[fish_ind]["regressors_window"])[selected_regressor_indices, :]  # selected regressors
    regressor_idx_map = {}  # map each regressor name to its position
    for idx, name in enumerate(selected_regressor_names):
        regressor_idx_map[name] = idx
    ordered_labels = cluster_results[fish_ind]["cluster_order"]                  # clusters arranged for plotting
    cluster_centroids = cluster_results[fish_ind]["cluster_centroids"]           # centroid traces in storage order
                                
    fish_IDs = np.where(fish_data[:, 0] == fish_ind)[0]                          # rows belonging to this fish in global array
    F_norm_fish = F_norm[fish_IDs, start_analyze_frame:]                                            # subset of traces for the current fish

                                            

    #%
    corr_threshold = 0.25 # minimum |r| to tag a cluster with a regressor
    corrs_regressors_centroids = Ca2ImagingFns.pearsonr_2Dnumb(reg_signals, cluster_centroids)   
    max_corrs = np.max(np.abs(corrs_regressors_centroids), axis=0)          # max |r| per cluster
    max_corr_idx = np.argmax(np.abs(corrs_regressors_centroids), axis=0)  # index of best regressor per cluster

    combined_signals = []
    combined_signals_names = []
    assigned_centroids = []
    assigned_centroids_set = set()

    associated_regressor = []

    for i in range(reg_signals.shape[0]):
        mask = (max_corr_idx == i) & (np.abs(max_corrs) >= corr_threshold)
        associated_clusters = np.where(mask)[0]
        if len(associated_clusters)>0: 
            print(associated_clusters)
            assigned_centroids.append(associated_clusters)
            assigned_centroids_set.update(associated_clusters.tolist())

            for j in associated_clusters:
                combined_signals.append(cluster_centroids[j, :])  # add associated centroids
                combined_signals_names.append(f"Cluster {int(ordered_labels[j])}")
                associated_regressor.append(str(selected_regressor_names[i]))
        combined_signals.append(reg_signals[i])
        combined_signals_names.append(selected_regressor_names[i])
        associated_regressor.append(str(selected_regressor_names[i]))

    remaining_centroids = sorted(
        set(range(cluster_centroids.shape[0])) - assigned_centroids_set,
        key=lambda idx: int(np.argmax(cluster_centroids[idx])),
    )

    for c_idx in remaining_centroids:
        combined_signals.append(cluster_centroids[c_idx])
        combined_signals_names.append(f"Cluster {int(ordered_labels[c_idx])}")
        associated_regressor.append("Unassigned")


    combined_signals = np.vstack(combined_signals)
    combined_signals_names = np.vstack(combined_signals_names)


    corrs = Ca2ImagingFns.pearsonr_2Dnumb(F_norm_fish, combined_signals)     
    #%           # correlate all neurons with this signal

    corr_threshold_in_cluster = 0.25    
    max_corr = np.max(corrs, axis=1)                           # max |r| per neuron
    max_corr_idx = np.argmax(corrs, axis=1) 

    signal_associations = []

    clusters_with_no_cells = []

    for k, signal_name in enumerate(combined_signals_names):
        mask = (max_corr_idx == k) & (np.abs(max_corr) >= corr_threshold_in_cluster)
        associated_neurons = np.where(mask)[0]                                      # neurons best matching this signal  
        corr_associated_neurons = max_corr[associated_neurons]                      # their correlation values
        associated_neurons_idx = fish_IDs[associated_neurons]
        F_norm_in_signal = F_norm[associated_neurons_idx, start_analyze_frame:]
        if associated_neurons.size > 1:
            corr_mat = np.corrcoef(F_norm_in_signal)
            dist = 1.0 - corr_mat
            dist = (dist + dist.T) * 0.5         # enforce symmetry
            np.fill_diagonal(dist, 0.0)          # zero diagonal
            linkage = scipy.cluster.hierarchy.linkage(
                scipy.spatial.distance.squareform(dist, checks=False),
                method="single",
            )
            order = scipy.cluster.hierarchy.leaves_list(linkage)
            order = order[-1::-1]  # reverse order for better visualization
            associated_neurons = associated_neurons[order]
            associated_neurons_idx = associated_neurons_idx[order] 
            corr_associated_neurons = corr_associated_neurons[order]
            roi_stats_in_signal = roi_stats[associated_neurons_idx]
            F_norm_in_signal = F_norm_in_signal[order, :]

            signal_associations.append({
                "signal_name": signal_name,
                "associated_neurons": associated_neurons,
                "associated_neurons_idx": associated_neurons_idx,
                "roi_stats_in_signal": roi_stats_in_signal,
                "corr_associated_neurons": corr_associated_neurons,
                "F_norm_in_signal": F_norm_in_signal,
                'associated_regressor': associated_regressor[k],
            })




    frame_rate = np.mean(ops[fish_name]['frame_rates'])
    target_minutes = np.array([5, 10, 15], dtype=float)
    minute_frames = (target_minutes * 60 * frame_rate).astype(int) - start_analyze_frame

    window_len = signal_associations[0]["F_norm_in_signal"].shape[1]
    valid = (minute_frames >= 0) & (minute_frames < window_len)

    tick_positions = minute_frames[valid]
    xticklabels_min = target_minutes[valid]
    separator = np.ones((5,window_len))
    separator[:] = np.nan


    all_categories = selected_regressor_names + ['Unassigned']

    n_unassinged = 0
    for i in range(len(signal_associations)):
        if signal_associations[i]['associated_regressor'] == 'Unassigned':
            n_unassinged += 1

    colormap_unassigned = Ca2ImagingFns.cluster_hsv_palette(n_unassinged, hue_start=0.1, hue_end=1, saturation=1)
    counts_per_category = []
    # plot per-regressor association heatmaps
    for regressor_name in all_categories:
        neuron_count = sum(
                sa['associated_neurons'].size
                for sa in signal_associations
                if sa['associated_regressor'] == regressor_name
        )
        counts_per_category.append(neuron_count)
        yticks = []
        ytick_labels = []
        F_norms_with_regressor = np.stack((separator,separator,separator),axis=2)
        k = 0
        for j in range(len(signal_associations)):
            if signal_associations[j]['associated_regressor'] == regressor_name:
                # print('match between regressor and signal:', regressor_name, signal_associations[j]['signal_name'], signal_associations[j]['associated_neurons'].size)  
                np.stack((signal_associations[j]["F_norm_in_signal"],signal_associations[j]["F_norm_in_signal"],signal_associations[j]["F_norm_in_signal"]),axis=2)
                
                new_stack = np.stack((
                    signal_associations[j]["F_norm_in_signal"],
                    signal_associations[j]["F_norm_in_signal"],
                    signal_associations[j]["F_norm_in_signal"],
                    ),axis=2)
                new_stack[new_stack>1] = 1
                new_stack[new_stack<0] = 0
                if regressor_name == 'Unassigned':
                    for color_channel in range(3):
                        new_stack[:,:,color_channel] = new_stack[:,:,color_channel] * (colormap_unassigned[k][color_channel])
                F_norms_with_regressor = np.vstack((
                    F_norms_with_regressor, 
                    new_stack,
                    np.stack((separator,separator,separator),axis=2),
                    ))
                if len(yticks) == 0:
                    yticks.append(signal_associations[j]["associated_neurons"].size)
                else:
                    yticks.append(yticks[-1] + signal_associations[j]["associated_neurons"].size + separator.shape[0])
                ytick_labels.append(str(signal_associations[j]["signal_name"]))
                k+=1


        with plt.rc_context({'font.size': 10}):
            title_str = f"GENO_{fish_type}_NAME_{fish_name}_CAT_{regressor_name}"



            fig = plt.figure(figsize=(10, F_norms_with_regressor.shape[0]/100))
            ax = fig.add_axes([0.28, 0.12, 0.7, 0.8])
            ax.imshow(F_norms_with_regressor, vmin=0, vmax=1, origin='upper', rasterized=True)
            ax.set_title(f"{title_str} (n={neuron_count})")
            
            ax.set_yticks(yticks)
            ax.set_yticklabels(ytick_labels)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(xticklabels_min)
            ax.set_xlabel("Time (min)")
            plt.savefig(
                os.path.join(
                    out_dir_regressor_association, 
                    Ca2ImagingFns.safe_filename(title_str + '.png')
                    ), 
                dpi=300
            )
            plt.savefig(
                os.path.join(
                    out_dir_regressor_association, 
                    Ca2ImagingFns.safe_filename(title_str + '.svg')
                    ), 
                dpi=300
            )
            plt.show()

    regressor_associated_signals.append(
        {
        "signal_associations": signal_associations,
        "fish_name": fish_name,
        "fish_type": fish_type, 
        "colormap_unassigned": colormap_unassigned,
        "counts_per_category": counts_per_category,
        }
    )
#%%
np.save(
    os.path.join(out_dir, 'regressor_associated_signals.npy'),
    regressor_associated_signals)
#%%
regressor_associated_signals = np.load(    
    os.path.join(out_dir, 'regressor_associated_signals.npy'),
    allow_pickle=True
)


for fish_ind in range(len(regressor_associated_signals)):
    n_in_categories = np.zeros(len(all_categories))
    fish_name = regressor_associated_signals[fish_ind]['fish_name']
    fish_type = regressor_associated_signals[fish_ind]['fish_type']
    signal_associations = regressor_associated_signals[fish_ind]['signal_associations']
    for signals in signal_associations:
        if signals['associated_regressor'] == all_categories[0]:
            n_in_categories[0] += signals['associated_neurons'].size
        elif signals['associated_regressor'] == all_categories[1]:
            n_in_categories[1] += signals['associated_neurons'].size
        elif signals['associated_regressor'] == all_categories[2]:
            n_in_categories[2] += signals['associated_neurons'].size  
        elif signals['associated_regressor'] == all_categories[3]:
            n_in_categories[3] += signals['associated_neurons'].size
    print(n_in_categories)
        

#%%
    
    # if len(F_norms_with_regressor) > 0:
    #     F_norms_with_regressor = np.vstack(F_norms_with_regressor)
    #     plt.imshow(F_norms_with_regressor, cmap='gray', vmin=0, vmax=1)
    #     plt.title(f"Neurons associated with regressor {regressor_name} (n={F_norms_with_regressor.shape[0]})")
    #     plt.show()
#%%                          # ROI stats for this fish

for i in range(len(signal_associations)):
    signal_name = signal_associations[i]["signal_name"]
    associated_neurons = signal_associations[i]["associated_neurons"]
    F_norm_in_signal = signal_associations[i]["F_norm_in_signal"]
    plt.imshow(F_norm_in_signal, cmap='gray', vmin=0, vmax=1)
    plt.title(f"Neurons associated with {signal_name} (n={associated_neurons.size}),\n associated regressor: {signal_associations[i]['associated_regressor']}")
    plt.show()






#%%
centroid_reg_corrs = {}                                                      # per-cluster correlation summary
for lbl, centroid in zip(ordered_labels, ordered_centroids):
    reg_corrs = {}
    for r_name, regr in zip(selected_regressor_names, reg_signals):
        reg_corrs[r_name] = float(np.corrcoef(regr, centroid)[0, 1])         # correlate centroid with each regressor
    centroid_reg_corrs[int(lbl)] = reg_corrs

neuron_assignments = []                                                      # per-neuron regressor hits
for neuron_idx, neuron_trace in zip(active_neurons_in_fish.tolist(), F_norm_fish):
    best_name = None
    best_corr = 0.0
    for r_name, regr in zip(selected_regressor_names, reg_signals):
        corr_val = float(np.corrcoef(regr, neuron_trace)[0, 1])              # correlation per regressor
        if abs(corr_val) > abs(best_corr):
            best_corr = corr_val
            best_name = r_name
    if abs(best_corr) >= corr_threshold_in_cluster:                          # keep neuron if above cutoff
        neuron_assignments.append(
            {"roi_index": int(neuron_idx), "regressor": best_name, "corr": best_corr}
        )


#%%
for fish_ind in tqdm.tqdm(range(len(cluster_results))):
    fish_name = cluster_results[fish_ind]['fish_name']
    print(fish_name)
    active_neurons_in_fish = cluster_results[fish_ind]['active_neuron_ids']
    traces_sorted = cluster_results[fish_ind]['traces_sorted']
    labels_sorted = cluster_results[fish_ind]['cluster_labels']
    label_starts = cluster_results[fish_ind]['label_starts']
    label_ends = cluster_results[fish_ind]['label_ends']
    final_roi_indices = cluster_results[fish_ind]['final_roi_indices']
    centroids = cluster_results[fish_ind]['cluster_centroids']
    fish_type = cluster_results[fish_ind]['fish_type']
    print(fish_type)

    corr_threshold = 0.35  # adjust as needed

    reg_signals = np.asarray(cluster_results[fish_ind]["regressors_window"])[selected_regressor_indices, :]
    regressor_idx_map = {} # links each selected regressor name to its position in selected_regressor_names
    for idx, name in enumerate(selected_regressor_names):
        regressor_idx_map[name] = idx
    ordered_labels = cluster_results[fish_ind]["cluster_order"]
    cluster_centroids = cluster_results[fish_ind]["cluster_centroids"]

    F_norm = cluster_results[fish_ind]['F_norm']
    fish_IDs = np.where(fish_data[:, 0] == fish_ind)[0]
    F_norm_fish = F_norm[fish_IDs, :]

    corr_threshold_in_cluster = 0.2


    label_indices = {}
    for lbl in ordered_labels:
        matching_idx = np.where(labels_sorted == lbl)[0]
        if matching_idx.size > 0:
            label_indices[lbl] = matching_idx

    cluster_labels_unique = np.unique(labels_sorted)
    centroid_lookup = {lbl: cluster_centroids[idx] for idx, lbl in enumerate(cluster_labels_unique)}
    ordered_centroids = np.vstack([centroid_lookup[lbl] for lbl in ordered_labels])

    n_clusters = len(ordered_labels)
    n_reg = len(selected_regressor_names)

    corrs_matrix = np.zeros((n_reg, n_clusters))
    for r_idx, r_name in enumerate(selected_regressor_names):
        regr = reg_signals[r_idx]
        corrs_matrix[r_idx] = np.array([np.corrcoef(regr, centroid)[0, 1] for centroid in ordered_centroids])

    best_reg_idx = np.argmax(np.abs(corrs_matrix), axis=0)
    best_corrs = corrs_matrix[best_reg_idx, np.arange(n_clusters)]

    cluster_hits = {r_name: {"labels": [], "corrs": [], "cluster_traces": []} for r_name in selected_regressor_names}
    unassigned_clusters = []

    for cluster_pos, lbl in enumerate(ordered_labels):
        best_idx = int(best_reg_idx[cluster_pos])
        best_corr = best_corrs[cluster_pos]
        if np.abs(best_corr) >= corr_threshold and lbl in label_indices:
            traces_block = traces_sorted[label_indices[lbl], :]
            cluster_hits[selected_regressor_names[best_idx]]["labels"].append(lbl)
            cluster_hits[selected_regressor_names[best_idx]]["corrs"].append(best_corr)
            cluster_hits[selected_regressor_names[best_idx]]["cluster_traces"].append(traces_block)
        elif lbl in label_indices:
            unassigned_clusters.append((lbl, best_corr, traces_sorted[label_indices[lbl], :]))

    behavior_panel_units = 60
    panel_height_units = []
    heat_panel_labels = []
    for r_name in selected_regressor_names:
        blocks = cluster_hits[r_name]["cluster_traces"]
        heat_units = sum(block.shape[0] for block in blocks) if blocks else 1
        panel_height_units.extend([heat_units, behavior_panel_units])
        heat_panel_labels.append(r_name)

    remaining_heat_units = (
        sum(block.shape[0] for (_, _, block) in unassigned_clusters) if unassigned_clusters else 1
    )
    panel_height_units.extend([remaining_heat_units, behavior_panel_units])

    total_units = sum(panel_height_units)
    fig_height = np.clip(0.02 * total_units, 8, 40)

    behav_data_plot = behav_data_per_fish[fish_ind][selected_regressor_indices, :].copy()
    behav_data_plot[0, :] *= 0.15
    behav_data_plot[1, :] *= 0.1
    behav_data_plot[2, :] -= np.median(behav_data_plot[2, :])

    behav_colors_full = {
        "Dark Flashes": "#8c564b",
        "OMR": "#1f77b4",
        "Tail Power": "#2ca02c",
        "Swim Bursting": "#d62728",
        "Lowpass Orientation": "#9467bd",
    }
    behav_colors = {name: behav_colors_full[name] for name in selected_regressor_names}

    heatmap_cmap = LinearSegmentedColormap.from_list("cluster_heatmap", ["white", "black"], N=256)
    global_vmin = 0
    global_vmax = 1

    fig = plt.figure(figsize=(18, fig_height))
    gs = fig.add_gridspec(len(panel_height_units), 1, height_ratios=panel_height_units, hspace=0.5)

    axes = []
    for idx in range(len(panel_height_units)):
        axes.append(fig.add_subplot(gs[idx], sharex=axes[0] if idx > 0 else None))

    axis_iter = iter(axes)
    heat_axes = {}
    behav_axes = {}
    for r_name in selected_regressor_names:
        heat_axes[r_name] = next(axis_iter)
        behav_axes[r_name] = next(axis_iter)
    remaining_heat_ax = next(axis_iter)
    remaining_behav_ax = next(axis_iter)

    window_len = traces_sorted.shape[1]
    x_coords = np.arange(window_len) + 0.5
    behav_xlim = (x_coords[0], x_coords[-1])

    target_minutes = np.array([2, 5, 10, 15], dtype=float)
    frame_rate = np.mean(ops[fish_name]['frame_rates'])
    minute_frames = (target_minutes * 60 * frame_rate).astype(int) - start_analyze_frame
    valid = (minute_frames >= 0) & (minute_frames < window_len)

    tick_positions = minute_frames[valid]
    xticklabels_min = target_minutes[valid]

    for r_name in selected_regressor_names:
        ax_heat = heat_axes[r_name]
        data = cluster_hits[r_name]
        ax_heat.set_title(f"{r_name} (max |r| ≥ {corr_threshold})")

        if not data["cluster_traces"]:
            ax_heat.text(0.5, 0.5, "No clusters assigned", transform=ax_heat.transAxes, ha="center", va="center")
            ax_heat.axis("off")
        else:
            stacked_traces = np.vstack(data["cluster_traces"])
            sns.heatmap(
                stacked_traces,
                cmap=heatmap_cmap,
                vmin=global_vmin,
                vmax=global_vmax,
                cbar=False,
                ax=ax_heat,
            )

            cluster_sizes = [block.shape[0] for block in data["cluster_traces"]]
            for boundary in np.cumsum(cluster_sizes)[:-1]:
                ax_heat.hlines(boundary, xmin=0, xmax=stacked_traces.shape[1], colors="white", linestyles="--", linewidth=1.2)

            y_centers = np.cumsum(cluster_sizes) - np.array(cluster_sizes) / 2.0
            y_labels = [f"Cluster {lbl} (n={size}, r={corr:.2f})" for lbl, size, corr in zip(data["labels"], cluster_sizes, data["corrs"])]
            ax_heat.set_yticks(y_centers)
            ax_heat.set_yticklabels(y_labels, rotation=0)
            ax_heat.tick_params(axis="x", labelbottom=False)
            ax_heat.collections[0].set_rasterized(True)

        ax_behav = behav_axes[r_name]
        behav_idx = regressor_idx_map[r_name]
        behav_trace = behav_data_plot[behav_idx, start_analyze_frame : start_analyze_frame + window_len]
        ax_behav.plot(x_coords, behav_trace, color=behav_colors.get(r_name, "black"), linewidth=2)
        ax_behav.set_xlim(behav_xlim)
        ax_behav.set_ylabel("Signal", fontsize=10)
        ax_behav.spines["top"].set_visible(False)
        ax_behav.spines["right"].set_visible(False)
        ax_behav.spines["left"].set_visible(False)
        ax_behav.set_xticks(tick_positions + 0.5)
        ax_behav.set_xticklabels(xticklabels_min)
        ax_behav.set_xlabel("Time (min)")

    remaining_sorted = []
    remaining_heat_ax.set_title("Remaining clusters")
    if not unassigned_clusters:
        remaining_heat_ax.text(0.5, 0.5, "All clusters assigned", transform=remaining_heat_ax.transAxes, ha="center", va="center")
        remaining_heat_ax.axis("off")
    else:
        remaining_info = []
        for lbl, corr, block in unassigned_clusters:
            mean_trace = np.mean(block, axis=0)
            peak_frame = int(np.argmax(mean_trace))
            remaining_info.append((peak_frame, lbl, corr, block))
        remaining_info.sort(key=lambda x: x[0])
        remaining_sorted = [(lbl, corr, block) for _, lbl, corr, block in remaining_info]

        remaining_traces = [block for (_, _, block) in remaining_sorted]
        stacked_remaining = np.vstack(remaining_traces)
        sns.heatmap(
            stacked_remaining,
            cmap=heatmap_cmap,
            vmin=global_vmin,
            vmax=global_vmax,
            cbar=False,
            ax=remaining_heat_ax,
        )

        remaining_sizes = [block.shape[0] for block in remaining_traces]
        for boundary in np.cumsum(remaining_sizes)[:-1]:
            remaining_heat_ax.hlines(boundary, xmin=0, xmax=stacked_remaining.shape[1], colors="white", linestyles="--", linewidth=1.2)

        remaining_centers = np.cumsum(remaining_sizes) - np.array(remaining_sizes) / 2.0
        remaining_labels = [lbl for (lbl, _, _) in remaining_sorted]
        remaining_corrs = [corr for (_, corr, _) in remaining_sorted]
        remaining_heat_ax.set_yticks(remaining_centers)
        remaining_heat_ax.set_yticklabels(
            [f"Cluster {lbl} (n={size}, max|r|={abs(corr):.2f})" for lbl, size, corr in zip(remaining_labels, remaining_sizes, remaining_corrs)],
            rotation=0,
        )
        remaining_heat_ax.tick_params(axis="x", labelbottom=False)
        remaining_heat_ax.collections[0].set_rasterized(True)

    for name, color in behav_colors.items():
        behav_idx = regressor_idx_map[name]
        rem_trace = behav_data_plot[behav_idx, start_analyze_frame : start_analyze_frame + window_len]
        remaining_behav_ax.plot(x_coords, rem_trace, color=color, linewidth=1.5, alpha=0.85, label=name)

    remaining_behav_ax.set_xlim(behav_xlim)
    remaining_behav_ax.set_ylabel("Signal", fontsize=10)
    remaining_behav_ax.spines["top"].set_visible(False)
    remaining_behav_ax.spines["right"].set_visible(False)
    remaining_behav_ax.spines["left"].set_visible(False)
    remaining_behav_ax.set_yticks([])
    remaining_behav_ax.legend(loc="upper right", fontsize=8, ncol=2, frameon=False)
    remaining_behav_ax.set_xticks(tick_positions + 0.5)
    remaining_behav_ax.set_xticklabels(xticklabels_min)
    remaining_behav_ax.set_xlabel("Time (min)")
    title_str = fish_name + "\n Fish Type = " + fish_type + " Clusters by regressors"
    remaining_behav_ax.set_title(title_str, fontsize=23)

    plt.savefig(os.path.join(out_dir_orderedheatmaps, Ca2ImagingFns.safe_filename(title_str) + '.png'), dpi=300)
    plt.savefig(
        os.path.join(out_dir_orderedheatmaps, Ca2ImagingFns.safe_filename(title_str) + ".svg"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    cluster_categories = {}
    for r_name, data in cluster_hits.items():
        cluster_sizes = [block.shape[0] for block in data["cluster_traces"]]
        cluster_categories[r_name] = {
            "cluster_labels": [int(lbl) for lbl in data["labels"]],
            "cluster_sizes": [int(sz) for sz in cluster_sizes],
            "cluster_correlations": [float(c) for c in data["corrs"]],
        }

    remaining_cluster_manifest = []
    for lbl, corr, block in remaining_sorted:
        mean_trace = block.mean(axis=0)
        peak_frame = int(np.argmax(mean_trace))
        remaining_cluster_manifest.append({
            "cluster_label": int(lbl),
            "cluster_size": int(block.shape[0]),
            "peak_frame": peak_frame,
            "peak_time_minutes": float((start_analyze_frame + peak_frame) / (frame_rate * 60)),
            "max_abs_corr": float(abs(corr)),
        })

    tick_manifest = {
        "frame_positions": [int(pos) for pos in tick_positions.tolist()],
        "minute_labels": [float(m) for m in xticklabels_min.tolist()],
    }

    enriched_entry = dict(cluster_results[fish_ind])
    enriched_entry.update({
        "cluster_categories": cluster_categories,
        "remaining_clusters": remaining_cluster_manifest,
        "remaining_cluster_order": [item["cluster_label"] for item in remaining_cluster_manifest],
        "tick_manifest": tick_manifest,
        "active_neuron_ids": active_neurons_in_fish.copy(),
        "final_roi_indices": final_roi_indices.copy(),
    })
    enriched_results.append(enriched_entry)

cluster_results_enriched_path = os.path.join(out_dir, "cluster_results_enriched.npy")
np.save(cluster_results_enriched_path, np.array(enriched_results, dtype=object))

print(f"Saved enriched clustering metadata to: {cluster_results_enriched_path}")


#%%
