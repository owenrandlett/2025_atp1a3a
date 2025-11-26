#%%

import numpy as np
import os
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import tifffile
import tqdm
os.chdir(r'/home/zeneb/github/2025_atp1a3a/Ca2+Imaging/')
import Ca2ImagingFns
from tifffile import imwrite
import nrrd
from matplotlib.colors import ListedColormap
from tifffile import imwrite
from PIL import Image
import seaborn as sns
import pandas as pd
import itertools
from scipy.stats import kruskal, mannwhitneyu

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
metadata_zbrain = all_fish_data['metadata_zbrain'].item()

print(f"Reloaded correlation results from: {corr_load_path}")

cluster_results = np.load(os.path.join(out_dir, "cluster_results.npy"),allow_pickle=True)
#%

# out_dir_orderedheatmaps = os.path.join(out_dir, 'clustered_ordered_heatmaps')
# os.makedirs(out_dir_orderedheatmaps, exist_ok=True)

#%%


selected_regressor_indices = [0, 1, 2]
selected_regressor_names = [regressor_names[i] for i in selected_regressor_indices]

out_dir_regressor_association = os.path.join(out_dir, 'regressor_cluster_association')
os.makedirs(out_dir_regressor_association, exist_ok=True)

start_analyze_frame = 200
re_analyze_signals = True

if re_analyze_signals:
    regressor_associated_signals = []

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
        print('counts per category:', counts_per_category)

        regressor_associated_signals.append(
            {
            "signal_associations": signal_associations,
            "fish_name": fish_name,
            "fish_type": fish_type, 
            "colormap_unassigned": colormap_unassigned,
            "counts_per_category": counts_per_category,
            }
        )

    np.save(
        os.path.join(out_dir, 'regressor_associated_signals.npy'),
        regressor_associated_signals)
#%
regressor_associated_signals = np.load(    
    os.path.join(out_dir, 'regressor_associated_signals.npy'),
    allow_pickle=True
)
#%%


for fish_ind in range(len(regressor_associated_signals)):
    counts_per_category = regressor_associated_signals[fish_ind]['counts_per_category']
    fish_type = regressor_associated_signals[fish_ind]['fish_type']
    print('counts per category for fish ' + str(fish_ind) + ': fish_type : ' + str(fish_type) + ', counts: ' + str(counts_per_category))

category_labels = selected_regressor_names + ["Unassigned"]
records = []
insufficient_data_threshold = 500
insufficient_data = []
for fish_ind, entry in enumerate(regressor_associated_signals):
    total_counts = int(np.sum(entry["counts_per_category"]))
    if total_counts < insufficient_data_threshold or entry["fish_type"] == "UNKNOWN":
        insufficient_data.append(fish_ind)
    else:
        row = {"fish_type": entry["fish_type"]}
        row.update(
            {label: int(count) for label, count in zip(category_labels, entry["counts_per_category"])}
        )
        records.append(row)

counts_df = pd.DataFrame(records)
counts_by_genotype = counts_df.groupby("fish_type")[category_labels].median()

print(counts_df)
print(counts_by_genotype)

plot_df = counts_df.melt(id_vars="fish_type", value_vars=category_labels,
                         var_name="category", value_name="count")
plot_df["count_plot"] = plot_df["count"].clip(lower=1)

genotype_order = ["+/+", "+/-", "-/-"]
plot_df["fish_type"] = pd.Categorical(plot_df["fish_type"], categories=genotype_order, ordered=True)
counts_df["fish_type"] = pd.Categorical(counts_df["fish_type"], categories=genotype_order, ordered=True)

kruskal_rows = []
pairwise_rows = []
valid_genos = [g for g in genotype_order if (counts_df["fish_type"] == g).sum() > 1]

for cat in category_labels:
    data_per_geno = [counts_df[counts_df["fish_type"] == geno][cat].values for geno in valid_genos]
    if len([arr for arr in data_per_geno if arr.size > 0]) < 2:
        continue
    stat, pval = kruskal(*data_per_geno)
    kruskal_rows.append({"category": cat, "stat": stat, "pval": pval})
    for g1, g2 in itertools.combinations(valid_genos, 2):
        vals1 = counts_df[counts_df["fish_type"] == g1][cat].values
        vals2 = counts_df[counts_df["fish_type"] == g2][cat].values
        if len(vals1) == 0 or len(vals2) == 0:
            continue
        stat, p = mannwhitneyu(vals1, vals2, alternative="two-sided")
        pairwise_rows.append({"category": cat, "geno_a": g1, "geno_b": g2, "stat": stat, "pval": p})

kruskal_df = pd.DataFrame(kruskal_rows)
pairwise_df = pd.DataFrame(pairwise_rows)
pairwise_lookup = {}
for _, row in pairwise_df.iterrows():
    pairwise_lookup[(row["category"], row["geno_a"], row["geno_b"])] = float(row["pval"])

def p_to_stars(p):
    if p <= 1e-4:
        return "****"
    if p <= 1e-3:
        return "***"
    if p <= 1e-2:
        return "**"
    if p <= 0.05:
        return "*"
    return "ns"

def annotate_sig(ax, x1, x2, y, text):
    y_top = y * 1.05
    ax.plot([x1, x1, x2, x2], [y, y_top, y_top, y], color="black", linewidth=1)
    ax.text((x1 + x2) / 2, y_top * 1.05, text, ha="center", va="bottom", fontsize=10)

palette = {"+/+": "#1f77b4", "+/-": "#ff7f0e", "-/-": "#2ca02c"}
fig, axes = plt.subplots(len(category_labels), 1, figsize=(5, 4 * len(category_labels)))

if len(category_labels) == 1:
    axes = [axes]

order_index = {geno: idx for idx, geno in enumerate(genotype_order)}

for ax, category in zip(axes, category_labels):
    cat_df = plot_df[plot_df["category"] == category]
    if cat_df.empty:
        ax.set_visible(False)
        continue
    sns.stripplot(
        data=cat_df,
        x="fish_type",
        y="count_plot",
        order=genotype_order,
        palette=palette,
        ax=ax,
        jitter=0.15,
        linewidth=0.5,
        alpha=0.6
    )
    for idx, geno in enumerate(genotype_order):
        vals = cat_df[cat_df["fish_type"] == geno]["count_plot"].dropna().to_numpy()
        if vals.size == 0:
            continue
        median_val = np.median(vals)
        ax.plot([idx - 0.2, idx + 0.2], [median_val, median_val],
                color="black", linewidth=2, solid_capstyle="butt")

    max_val = cat_df["count_plot"].max()
    if np.isfinite(max_val) and max_val > 0:
        current_height = max_val * 1.05
        step = 1.2
        for g1, g2 in itertools.combinations(genotype_order, 2):
            pval = (pairwise_lookup.get((category, g1, g2)) or
                    pairwise_lookup.get((category, g2, g1)))
            if pval is None or pval > 0.05:
                continue
            annotate_sig(ax, order_index[g1], order_index[g2], current_height, p_to_stars(pval))
            current_height *= step

    tick_labels = []
    for geno in genotype_order:
        n_samples = cat_df[cat_df["fish_type"] == geno].shape[0]
        tick_labels.append(f"{geno} (n={n_samples})")
    ax.set_xticks(range(len(genotype_order)))
    ax.set_xticklabels(tick_labels)

    ax.set_title(category)
    ax.set_xlabel("Genotype")
   
    ax.set_ylabel("Neuron count" if ax is axes[0] else "")


plt.tight_layout()
plt.show()

print("Kruskal-Wallis comparisons:")
print(kruskal_df)
print("Pairwise Mann-Whitney comparisons:")
print(pairwise_df)


#%% make a spatial map for each cluster based on the reference brain coordinates
ref_brain_path = r'/media/FastDrive/atp1a3a_data/registration/telen_template_allfish_HuC-H2BGCaMP.nrrd' 
ref_brain, ref_meta = nrrd.read(ref_brain_path)
width, height, Zs = ref_brain.shape
ref_brain = np.moveaxis(ref_brain, [0,1,2], [2,1,0])
xy_rez = ref_meta['space directions'][0][0]
z_rez = ref_meta['space directions'][-1][-1]

ref_meta = {
    'xy_rez': xy_rez,
    'z_rez': z_rez,
    'width': width,
    'height': height,
    'Zs': Zs,
}
#%%



for fish_ind in range(len(regressor_associated_signals)):

    regressor_associated_signals_fish = regressor_associated_signals[fish_ind]
    signal_associations = regressor_associated_signals_fish["signal_associations"]
    fish_name = regressor_associated_signals_fish["fish_name"]
    fish_type = regressor_associated_signals_fish["fish_type"]
    colormap_unassigned = regressor_associated_signals_fish["colormap_unassigned"]

    print(f'fish type : {fish_type}, fish name : {fish_name}')

    subfolder_out = os.path.join(out_dir_regressor_association, Ca2ImagingFns.safe_filename(f'GENOTYPE_{fish_type}_FISH_{fish_name}'))
    os.makedirs(subfolder_out, exist_ok=True)


                    # signal_associations.append({
                    #     "signal_name": signal_name,
                    #     "associated_neurons": associated_neurons,
                    #     "associated_neurons_idx": associated_neurons_idx,
                    #     "roi_stats_in_signal": roi_stats_in_signal,
                    #     "corr_associated_neurons": corr_associated_neurons,
                    #     "F_norm_in_signal": F_norm_in_signal,
                    # })

    k = 0
    
    crop_extents = metadata_zbrain['crop_extents']
    x_size_crop = crop_extents[1] - crop_extents[0]
    y_size_crop = crop_extents[3] - crop_extents[2]
    z_size_crop = crop_extents[5] - crop_extents[4]
    clusters_stacks = np.zeros((z_size_crop, y_size_crop, x_size_crop), dtype='uint8')
    clusters_projections = []
    centroids_projections = []
    unnasigned_mask = np.zeros(len(signal_associations))
    centroids_of_signals = np.zeros((len(signal_associations), 3))
    for signal_ind, signal in enumerate(signal_associations):
        signal_name = signal["signal_name"]
        associated_neurons_idx = signal["associated_neurons_idx"]
        roi_stats_in_signal = signal["roi_stats_in_signal"]
        associated_regressor = signal['associated_regressor']
        if associated_regressor == 'Unassigned':
            cmap = ListedColormap(np.vstack((np.zeros(3), colormap_unassigned[k])))
            unnasigned_mask[k] = 1
            k += 1
        else:
            cmap = 'gray'


        IM_roi, im_proj = Ca2ImagingFns.draw_hit_volume_provideROIstats(
            np.arange(len(roi_stats_in_signal)),
            roi_stats_in_signal,
            metadata_zbrain,
            outline=metadata_zbrain['outline_crop'])
        im_proj = Ca2ImagingFns.to_rgb(im_proj, cmap_name=cmap, vmin=0, vmax=1)
        
        clusters_stacks[IM_roi > 0] = signal_ind + 1
        clusters_projections.append(im_proj)
        



        centroids_in_signal = np.zeros((len(roi_stats_in_signal), 3))
        for j, roi in enumerate(roi_stats_in_signal):
            centroid = roi['centroid_refbrain']
            centroids_in_signal[j, :] = centroid
        centroid_signal = np.mean(centroids_in_signal, axis=0)
        centroids_of_signals[signal_ind, :] = centroid_signal



    unnasigned_coords = []
    unnasigned_colors = []
    color_cursor = 0
    for signal_ind, signal in enumerate(signal_associations):
        if signal["associated_regressor"] == "Unassigned" and color_cursor < len(colormap_unassigned):
            unnasigned_coords.append(centroids_of_signals[signal_ind])
            unnasigned_colors.append(colormap_unassigned[color_cursor])
            color_cursor += 1
    unnasigned_coords = np.asarray(unnasigned_coords)

    view_angles = [
        (90, 90, "Axial (top)"),
        (45, 90, "Axial tilt"),
        (0, 90, "Coronal (front)"),
        (0, 45, "Coronal tilt"),
        (0, 0, "Sagittal (side)"),
        (30, 0, "Sagittal tilt"),
    ]

    fig = plt.figure(figsize=(18, 4))
    for subplot_idx, (elev, azim, title) in enumerate(view_angles, start=1):
        ax = fig.add_subplot(1, len(view_angles), subplot_idx, projection="3d")
        if unnasigned_coords.size:
            ax.scatter(
                unnasigned_coords[:, 0],
                unnasigned_coords[:, 1],
                unnasigned_coords[:, 2],
                color=unnasigned_colors,
                s=60,
                depthshade=False,
            )
            ax.plot(
                unnasigned_coords[:, 0],
                unnasigned_coords[:, 1],
                unnasigned_coords[:, 2],
                color="black",
                linewidth=1.5,
                alpha=0.8,
            )
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_xlabel("X coord")
        ax.set_ylabel("Y coord")
        ax.set_zlabel("Z coord")

    plt.tight_layout()
    plt.savefig(os.path.join(subfolder_out, Ca2ImagingFns.safe_filename(f'GENO_{fish_type}_FISH_{fish_name}_UnassignedNeurons_3Dscatter.svg')), dpi=300)
    plt.show()

    #%

    fig, ax = plt.subplots(len(clusters_projections), 1, figsize=(5, len(clusters_projections)* 5))
    for i in range(len(clusters_projections)):
        ax[i].imshow(clusters_projections[i], rasterized=True)
        ax[i].set_title(signal_associations[i]["signal_name"] + f" (n={signal_associations[i]['associated_neurons'].size})")
        ax[i].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(subfolder_out, Ca2ImagingFns.safe_filename(f'GENO_{fish_type}_FISH_{fish_name}_Clusters_Projections.svg')), dpi=300)
    plt.show()

    stack_tif = os.path.join(
        subfolder_out,
        Ca2ImagingFns.safe_filename(f"GENO_{fish_type}_FISH_{fish_name}_ClustersStack.tif")
    )
    imwrite(stack_tif, clusters_stacks.astype(np.uint8), photometric="minisblack")

    proj_arr = (np.stack(clusters_projections) * 255).clip(0, 255).astype(np.uint8)
    proj_tif = os.path.join(
        subfolder_out,
        Ca2ImagingFns.safe_filename(f"GENO_{fish_type}_FISH_{fish_name}_ClustersProjections.tif")
    )
    imwrite(proj_tif, proj_arr, photometric="rgb")



#%% quantify for each category the spatial location of the associated neurons in teh z-brain regions

z_brain_2 = tifffile.imread(os.path.realpath(r'/media/BigBoy/ciqle/ref_brains/ZBrain2_0.tif')) 

IDs = {

        'olfactory bulb' : z_brain_2 == 29, # olfactory bulb
        'pallium': z_brain_2 == 30, # pallium
        'subpallium': z_brain_2 == 31, # subpallium
        'habenula': z_brain_2 == 2, # habenula
        'pretectum': z_brain_2==23, # pretectum
        'thalamus': (z_brain_2 >=27) & (z_brain_2 <= 28), # thalamus
        'posterior tuberculum': z_brain_2 == 23, # posterior tuberculum
        'tectum': (z_brain_2 >=111) & (z_brain_2 <= 112), # tectum
        'tegmentum': (z_brain_2 >=113) & (z_brain_2 <= 115), # tegmentum
}

for key in IDs.keys():
    print(f"{key}: {np.sum(IDs[key])} voxels")
    fig, ax = plt.subplots(1,2, figsize=(6,6))
    ax[0].imshow(np.max(IDs[key], axis=0))
    ax[0].set_title(key)
    ax[1].imshow(np.max(IDs[key], axis=2).T)
    plt.show()
#%%
n_fish = len(regressor_associated_signals)
valid_fish = []
categories_labels = np.zeros((Zs_zbrain, height_zbrain, width_zbrain, len(valid_fish)),dtype='uint8')
fish_types_valid = []
fish_names_valid = []
for fish_ind in range(n_fish):
    if fish_ind not in insufficient_data:
        valid_fish.append(fish_ind)

for fish_ind in valid_fish:
    #%%
fish_names_valid.append(regressor_associated_signals[fish_ind]['fish_name'])
fish_types_valid.append(regressor_associated_signals[fish_ind]['fish_type'])

signal_associations = regressor_associated_signals[fish_ind]['signal_associations']
roi_stats_in_category = {}


for reg_ind, regressor_name in enumerate(all_categories):
    roi_stats_in_category[regressor_name] = []
    for signals in signal_associations:
        if signals['associated_regressor'] == regressor_name:
            for roi in signals['roi_stats_in_signal']:
                roi_stats_in_category[regressor_name].append(roi)