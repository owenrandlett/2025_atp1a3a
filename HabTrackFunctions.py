import pickle
import matplotlib as mpl

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from scipy import stats
import scikit_posthocs as sp
import seaborn as sns
import os
from pylatexenc.latex2text import LatexNodes2Text


def make_graph_folder(folder_name, root_dir):
    graph_folder = os.path.join(
        os.path.split(root_dir)[0], "BigRigData_Graphs", folder_name
    )
    if not os.path.exists(graph_folder):
        os.makedirs(graph_folder)
    return graph_folder


def get_all_matching_ROIS(plot_IDs, track_data_combined):
    rois_matching = []
    for i in plot_IDs:
        rois_matching = np.hstack((rois_matching, track_data_combined["rois"][i]))
    return rois_matching.astype(int)


def ffill_cols(a, startfillval=0):
    mask = np.isnan(a)
    tmp = a[0].copy()
    a[0][mask[0]] = startfillval
    mask[0] = False
    idx = np.where(~mask, np.arange(mask.shape[0])[:, None], 0)
    out = np.take_along_axis(a, np.maximum.accumulate(idx, axis=0), axis=0)
    a[0] = tmp
    return out


def remove_brackets_invalid(string, invalid=r'<>:"/\|?*() '):
    while "(" in string and ")" in string:
        start = string.index("(")
        end = string.index(")", start) + 1
        string = string[:start] + string[end:]

    for char in invalid:
        string = string.replace(char, "")
    
    # remove any latex-notation stuff
    string = LatexNodes2Text().latex_to_text(string)
    return string


def load_burst_pkl(burst_file):
    with open(burst_file, "rb") as f:
        burst_data = pickle.load(f)

    tail_coords = burst_data["tail_coords"]
    orientations = burst_data["orientations"]

    heading_dir = burst_data["heading_dir"]
    bend_amps = burst_data["bend_amps"]

    return tail_coords, orientations, heading_dir, bend_amps


def subtract_angles(lhs, rhs):
    import math

    """Return the signed difference between angles lhs and rhs

    Return ``(lhs - rhs)``, the value will be within ``[-math.pi, math.pi)``.
    Both ``lhs`` and ``rhs`` may either be zero-based (within
    ``[0, 2*math.pi]``), or ``-pi``-based (within ``[-math.pi, math.pi]``).
    """

    return math.fmod((lhs - rhs) + math.pi * 3, 2 * math.pi) - math.pi


def simpleaxis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def convert_roi_str(roi_str):
    ### convert from a string that has a matlab stype indexing, using a mix of commans and colons, into a vector of indexes
    ### note that ROIs in the spreadsheet should be 1 indexed to match with old maltab formatting
    roi_str = roi_str.strip("[").strip("]").split(",")
    # print(roi_str)
    for k, roi_part in enumerate(roi_str):
        roi_part = roi_part.strip(" ")
        if roi_part.find(":") == -1:  # if there is no colon, just a single number
            vec = int(roi_part) - 1
        else:
            roi_subparts = roi_part.split(":")
            if len(roi_subparts) == 2:  # only 1 semicolon, count by ones
                vec = np.arange(
                    int(roi_subparts[0]) - 1, int(roi_subparts[1])
                )  # subtract 1 from first index to make 0 indexed
            elif (
                len(roi_subparts) == 3
            ):  # 2 semicolons in matlab index style, jump by middle number
                vec = np.arange(
                    int(roi_subparts[0]) - 1, int(roi_subparts[2]), int(roi_subparts[1])
                )
            else:
                raise ValueError(
                    "problem with ROI parsing for roi string" + str(roi_str)
                )
        if k == 0:
            roi_vec = vec
        else:
            roi_vec = np.hstack((roi_vec, vec))

    return roi_vec


def plot_burst_responses(
    track_data,
    fish_names,
    fish_ids,
    col_vec,
    save_str,
    nStimInBlocks=60,
    smooth_window=15,
    components_to_plot=range(10),
    plot_taps=True,
    plot_retest=True,
    stim_times=None,
    first_block=False,
):
    import warnings
    from scipy.signal import savgol_filter
    mpl.rcParams['svg.fonttype'] = 'none'

    # Hardcoded font sizes
    legend_fontsize = 18
    y_axis_fontsize = 25
    x_axis_fontsize = 25
    ticks_fontsize = 14

    n_gr = len(fish_names)


    # If no stimulus times are given, calculate them from the TiffTimeInds
    if np.sum(stim_times == None) > 0:
        stim_times = []
        for i in range(len(track_data["TiffTimeInds"])):
            stim_times.append(
                (
                    track_data["TiffTimeInds"][i] - track_data["TiffTimeInds"][0]
                ).total_seconds()
                / 60
                / 60
            )
        stim_times = np.array(stim_times)

    time_inds = stim_times
    stim_given = track_data["stim_given"]

    dtype_to_plot = np.array(list(track_data.keys()))[components_to_plot]
    y_text = dtype_to_plot

    for d, dtype in enumerate(dtype_to_plot):

        data = abs(track_data[dtype])
        plt.figure(figsize=(10, 7))
        plt.xlabel("time (hr)", fontsize=x_axis_fontsize)
        plt.ylabel(y_text[d].replace("_", " "), fontsize=y_axis_fontsize)

        for i in range(n_gr):  # plot the raw dark flash stimuli
            inds_stim = np.ix_((stim_given == 1) | (stim_given == 3))[0]
            inds_fish = fish_ids[i]
            inds_both = np.ix_(inds_stim, inds_fish)

            plt.plot(
                time_inds[inds_stim],
                np.nanmean(data[inds_both], axis=1),
                ".",
                markersize=3,
                color=col_vec[i],
                label=fish_names[i] + " , n=" + str(len(inds_fish)),
            )

        lgnd = plt.legend(fontsize=legend_fontsize, markerscale=3, loc="lower right")

        for i in range(
            n_gr
        ):  # plot the smoothed data off of the frist 4 blocks, and retest block
            inds_fish = fish_ids[i]
            # inds_stim =
            for k in range(5):
                inds_block = np.ix_((stim_given == 1) | (stim_given == 3))[0][
                    k * nStimInBlocks : k * nStimInBlocks + nStimInBlocks
                ]
                inds_both_block = np.ix_(inds_block, inds_fish)

                y = np.nanmean(data[inds_both_block], axis=1)
                x = time_inds[inds_block]
                # remove NaNs
                x = x[~np.isnan(y)]
                y = y[~np.isnan(y)]
                try:
                    y = savgol_filter(y, smooth_window, 2)
                    plt.plot(x, y, "-", color=col_vec[i], linewidth=5, alpha=0.8)
                except:
                    warnings.warn("savgol did not converge")

        # plot taps
        if plot_taps:
            for i in range(n_gr):
                inds_fish = fish_ids[i]
                inds_block = np.where(stim_given == 2)[0]
                n_blocks_taps = np.ceil(len(inds_block) / nStimInBlocks).astype(int)

                for tp_blk in range(n_blocks_taps):
                    inds_tp_block = np.zeros(stim_given.shape).astype(bool)
                    inds_tp_block[:] = False

                    st_tap = tp_blk * nStimInBlocks
                    end_tap = min(
                        nStimInBlocks + tp_blk * nStimInBlocks, len(inds_block)
                    )
                    inds_tp_block[inds_block[st_tap:end_tap]] = True
                    inds_both_block = np.ix_(inds_tp_block, inds_fish)
                    y1 = np.nanmean(data[inds_both_block], axis=1)
                    plt.plot(
                        time_inds[inds_tp_block],
                        y1,
                        "x",
                        markersize=3,
                        color=col_vec[i],
                    )

                    try:
                        y2 = savgol_filter(y1, smooth_window, 2)
                        plt.plot(
                            time_inds[inds_tp_block],
                            y2,
                            "-",
                            color=col_vec[i],
                            linewidth=3,
                            alpha=0.8,
                        )

                    except:
                        warnings.warn("savgol did not converge")

        if not plot_retest and not plot_taps:
            plt.xlim((-0.1, 8.1))

        if plot_taps and not plot_retest:
            plt.xlim((-0.1, 9.55))

        if first_block:  # plot only first block
            plt.xlim((-0.1, 1.1))

        simpleaxis(plt.gca())
        plt.xticks(fontsize=ticks_fontsize)
        plt.yticks(fontsize=ticks_fontsize)
        plt.savefig(
            remove_brackets_invalid(save_str + "_" + dtype + ".svg"),
            bbox_inches="tight",
            transparent=True,
        )
        plt.savefig(
            remove_brackets_invalid(save_str + "_" + dtype + ".png"),
            bbox_inches="tight",
            transparent=True,
            dpi=100,
        )

        plt.show()


def plot_cum_diff(
    data,
    fish_names,
    fish_ids,
    save_name,
    components_to_plot=np.arange(8),
    n_norm=3,
    n_boots=2000,
    ylim=0.25,
):
    ### calculate cumulative difference relative to controls, as in Randlett et al., Current Biology, 2019. Controls are at 0 index, treatemnt are 1st index. 
    # n_norm will give the number of inital responses to normalize to

    # Hardcoded font sizes
    legend_fontsize = 12
    axes_fontsize = 18
    ticks_fontsize = 14
    
    mpl.rcParams['svg.fonttype'] = 'none'

    plt.fill_between(
        np.arange(240),
        np.ones(240) * -0.05,
        np.ones(240) * 0.05,
        color=[0.5, 0.5, 0.5],
        alpha=0.4,
    )
    plt.vlines(60, -1, 1, colors="k", linestyles="dashed")
    plt.vlines(120, -1, 1, colors="k", linestyles="dashed")
    plt.vlines(180, -1, 1, colors="k", linestyles="dashed")
    plt.hlines(0, 0, 240, colors="k", linestyles="dashed")
    stim_given = data["stim_given"]

    # color palatte to match Randlett 2019 if plotted in current order in track_data files
    col_vec = [
        [0, 0, 1], # probability = blue
        [0, 1, 0], # latency = red
        [1, 0, 0], # double responses = red    
        [0.20588235, 0.41764705, 0.06470587], # Reorientation = dark green
        [0.9, 0.9, 0], # displacement = yellow
        [1, 0, 1], # duration = magenta
        [0.51764706, 0.41960784, 1], # bend amplitude = lilac 
        [0, 0, 0], # compound obend = black
    ]

    dtype_to_plot = np.array(list(data.keys()))[components_to_plot]
    col_vec = list(np.array(col_vec)[components_to_plot])
    for col_id, data_type in enumerate(dtype_to_plot):
        if (
            data_type == "Latency_(msec)"
        ):  # invert so that habituation changes match direction
            data_to_plot = 1000 - abs(data[data_type][stim_given == 1, :])
        else:
            data_to_plot = abs(
                data[data_type][stim_given == 1, :]
            )  # use absolute values to ignore tail bending/directional signs

        cont_ids = fish_ids[0]  # 0'th index is the control
        cont_data = data_to_plot[:, cont_ids]
        n_cont = len(cont_ids)

        treat_ids = fish_ids[1]  # 1st index is the treatment
        treat_data = data_to_plot[:, treat_ids]
        n_treat = len(treat_ids)

        cum_diff_dist = np.zeros((240, n_boots))

        for i in range(n_boots):
            mean_treat = np.nanmean(
                treat_data[:, np.random.randint(0, n_treat, n_treat)], axis=1
            )
            mean_cont = np.nanmean(
                cont_data[:, np.random.randint(0, n_cont, n_cont)], axis=1
            )
            nan_IDs = np.isnan(mean_treat) | np.isnan(mean_cont)
            norm_vec = np.arange(1, 241)
            for el, val in enumerate(norm_vec):
                if nan_IDs[el] == True:
                    norm_vec[el:] = norm_vec[el:] - 1

            cum_diff_dist[:, i] = (
                np.nancumsum(
                    mean_cont / np.nanmean(mean_cont[:n_norm])
                    - mean_treat / np.nanmean(mean_treat[0:n_norm])
                )
                / norm_vec
            )

        cum_diff_dist[~np.isfinite(cum_diff_dist)] = 0
        mu = np.nanmean(cum_diff_dist, axis=1)
        sigma = np.nanstd(cum_diff_dist, axis=1)
        CI = stats.norm.interval(0.95, loc=mu, scale=sigma / np.sqrt(n_treat))
        CI[0][np.isnan(CI[0])] = mu[np.isnan(CI[0])]
        CI[1][np.isnan(CI[1])] = mu[np.isnan(CI[1])]
        plt.plot(
            np.arange(240),
            mu,
            color=col_vec[col_id],
            label=data_type.replace("_", " "),
            linewidth=2,
        )
        plt.fill_between(
            np.arange(240),
            CI[0],
            CI[1],
            alpha=0.3,
            color=col_vec[col_id],
            label="_nolegend_",
            interpolate=True,
        )

    plt.title(
        fish_names[1],
        fontsize=axes_fontsize,
    )

    plt.ylabel(
        "Cumulative Mean Difference \n(norm.) vs. " 
        + fish_names[0], 
        fontsize=axes_fontsize
    )
    plt.xlabel("Stimuli", fontsize=axes_fontsize)
    legend = plt.legend(
        #bbox_to_anchor=(1.05, 1.0, 0.3, 0.2),
        #loc="upper left",
        fontsize=legend_fontsize,
        markerscale=2,
    )

    # Increase the linewidth in the legend
    for line in legend.get_lines():
        line.set_linewidth(4.0)

    plt.xticks((0, 60, 120, 180, 240), fontsize=ticks_fontsize)
    plt.ylim((-ylim, ylim))
    plt.xlim((0, 240))

    plt.savefig(
        remove_brackets_invalid(
            save_name + "_CumulDiff_" + fish_names[0] + "_minus_" + fish_names[1] + ".png"
        ),
        dpi=100,
        bbox_inches="tight",
    )
    plt.savefig(
        remove_brackets_invalid(
            save_name + "_CumulDiff_" +fish_names[0] + "_minus_" + fish_names[1] + ".svg"
        ),
        dpi=100,
        bbox_inches="tight",
    )
    plt.show()


def plot_means_epoch(
    track_data,
    fish_names,
    fish_ids,
    stim_epochs,
    epoch_names,
    save_str,
    col_vec,
    components_to_plot=np.arange(8),
    ylim_all_same=True,
):

    n_gr = len(fish_ids)
    mpl.rcParams['svg.fonttype'] = 'none'

    all_rois = []
    group_labels = []
    for i, rois in enumerate(fish_ids):
        for roi in rois:
            all_rois.append(roi)
            group_labels.append(fish_names[i])
    all_rois = np.array(all_rois).astype(int)
    n_rois = len(all_rois)

    dtype_to_plot = np.array(list(track_data.keys()))[components_to_plot]
    for dtype in dtype_to_plot:
        dataset = abs(
            track_data[dtype]
        )  # take the absolute value of the data to ignore sign of turn

        epoch_response_per_fish = np.zeros((n_rois, len(stim_epochs)))

        for i, epoch in enumerate(stim_epochs):
            epoch_data = dataset[epoch, :]
            epoch_response_per_fish[:, i] = np.nanmean(epoch_data[:, all_rois], axis=0)

        # Create a DataFrame for plotting

        df_epoch_responses = pd.DataFrame(epoch_response_per_fish, columns=epoch_names)
        df_epoch_responses["Group"] = group_labels
        # want 5 x 4 for each subplot
        
        n_high = np.ceil(len(epoch_names) / 2)
        n_wide = np.floor(len(epoch_names) / 2)
        plt.figure(figsize=(n_wide*5, n_high*5))


        # Function to add significance annotations
        def add_significance(ax, x1, x2, y, p_val, h=0.02):
            if p_val < 0.001:
                text = "***"
            elif p_val < 0.01:
                text = "**"
            elif p_val < 0.05:
                text = "*"
            else:
                text = "n.s."
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1, color="k")
            ax.text(
                (x1 + x2) * 0.5,
                y + h - 0.05,
                text,
                ha="center",
                va="bottom",
                color="k",
                fontsize=14,
            )

        # Loop through each epoch and create a strip plot with half violin plot
        for i, epoch in enumerate(epoch_names):
            ax = plt.subplot(int(np.ceil(len(epoch_names)/ 2)), int(np.floor(len(epoch_names) / 2)), i + 1)  # Adjust the grid size as needed

            # Drop NaN values for the current epoch
            df_epoch_nonan = df_epoch_responses[["Group", epoch]].dropna()

            # Check the number of unique groups
            unique_groups = df_epoch_nonan["Group"].unique()
            if len(unique_groups) == 2:
                # Perform Mann-Whitney U test if there are only two groups
                group1_data = df_epoch_nonan[
                    df_epoch_nonan["Group"] == unique_groups[0]
                ][epoch]
                group2_data = df_epoch_nonan[
                    df_epoch_nonan["Group"] == unique_groups[1]
                ][epoch]
                U, p = stats.mannwhitneyu(group1_data, group2_data)
                significant_pairs = (
                    [(unique_groups[0], unique_groups[1])] if p < 0.05 else []
                )

            else:
                # Perform Kruskal-Wallis test
                group_data = [
                    df_epoch_nonan[df_epoch_nonan["Group"] == group][epoch]
                    for group in df_epoch_nonan["Group"].unique()
                ]
                H, p = stats.kruskal(*group_data)

                # Perform Dunn's test for post-hoc pairwise comparisons if Kruskal-Wallis is significant
                if p < 0.05:
                    dunn_results = sp.posthoc_dunn(
                        df_epoch_nonan,
                        val_col=epoch,
                        group_col="Group",
                        p_adjust="bonferroni",
                    )
                    significant_pairs = (
                        dunn_results[dunn_results < 0.05].stack().index.tolist()
                    )
                else:
                    significant_pairs = []

            sns.stripplot(
                x="Group",
                y=epoch,
                data=df_epoch_nonan,
                hue="Group",  # Assign the x variable to hue
                size=4,  # Marker size
                palette=col_vec,  # Use the defined color palette
                alpha=0.3,  # Marker transparency
                # edgecolor='black', # Marker edge color
                linewidth=1,  # Marker edge width
                jitter=0.2,  # Add jitter to the points
                zorder = 5,
            )

            sns.violinplot(
                x="Group",
                y=epoch,
                data=df_epoch_nonan,
                hue="Group",  # Assign the x variable to hue
                split=False,  # Do not split the violin plot
                inner=None,  # Remove the fill color
                palette=col_vec,  # Use the defined color palette
                linewidth=1,  # Set the line width
                alpha=0.5,
                cut=0,  # Do not extend the violin plot beyond the data range
                zorder = 3,
            )

            # Plot the means using sns.pointplot
            sns.pointplot(
                x="Group",
                y=epoch,
                data=df_epoch_nonan,
                estimator=np.mean,
                color=(0.5, 0.5, 0.5),
                markeredgewidth=1,
                markeredgecolor="black",
                markerfacecolor="white",
                alpha=0.9,
                errorbar=None,
                markers="o",  # Marker style
                linestyles="",  # No line connecting the points        # Scale the size of the markers
                linewidth=1,
                markersize=15,  # Increase the size of the markers
                zorder=10,  # Increase zorder to make it more prominent
            )
            

            # Add significance annotations
            ax = plt.gca()
            if ylim_all_same:
                y_max = df_epoch_responses[epoch_names].quantile(0.95).quantile(0.99)
                y_min = df_epoch_responses[epoch_names].min().min()
            else:
                y_max = df_epoch_responses[epoch].quantile(0.95)
                y_min = df_epoch_responses[epoch].min()

            y_lim = y_max * 1.3  # Add some space at the top
            plt.ylim(y_min, y_lim)
            k = 1
            perc_ymax = (y_max - y_min) * 0.1

            unique_significant_pairs = set(
                tuple(sorted(pair)) for pair in significant_pairs
            )
            for group1, group2 in unique_significant_pairs:
                x1, x2 = df_epoch_nonan["Group"].unique().tolist().index(
                    group1
                ), df_epoch_nonan["Group"].unique().tolist().index(group2)
                if len(unique_groups) == 2:
                    add_significance(ax, x1, x2, y_max + ((k - 3) * perc_ymax), p)
                else:
                    add_significance(
                        ax,
                        x1,
                        x2,
                        y_max + ((k - 3) * perc_ymax),
                        dunn_results.loc[group1, group2],
                    )
                k += 1
            plt.title(epoch, fontsize=19, weight='bold', pad=-20)
            plt.ylabel(dtype.replace("_", " "), fontsize=16)
            plt.xlabel("")  # Remove the x-axis label
            plt.xticks(rotation=9, fontsize=12, weight='bold')
            # Remove top and right spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            
            
            # # Calculate mean values for each group using numpy
            # mean_values = df_epoch_nonan.groupby("Group")[epoch].apply(np.mean).values


            # # Add horizontal lines from each point to the y-axis
            # x_min, x_max = ax.get_xlim()
            # for mean_ind, value in enumerate(mean_values):
            #     ax.hlines(y=value, xmin=x_min, xmax=mean_ind, color=col_vec[mean_ind], linestyle='--',linewidth=0.85, alpha=1, zorder=0)
            
            # ax.set_xlim(x_min, x_max)
            
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.4, hspace=0.3)
        plt.savefig(
            remove_brackets_invalid(save_str + "_" + dtype + "_Epochs.png"),
            dpi=100,
            bbox_inches="tight",
        )
        plt.savefig(
            remove_brackets_invalid(save_str + "_" + dtype + "_Epochs.svg"),
            dpi=100,
            bbox_inches="tight",
        )
        plt.show()
