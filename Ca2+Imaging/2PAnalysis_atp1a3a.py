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
import tqdm
os.chdir(os.path.dirname(__file__))
import Ca2ImagingFns


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


re_analyze = True # set to True to re-process all data from raw
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
        # "20250910_atp1a3a_Fish8_func-000", # data corrupted in cluster plot... not sure what is wrong but need to ignore
        "20250910_atp1a3a_Fish9_func-000",
        # "20250911_atp1a3a_Fish4_func-000", # z position unstable
        "20250911_atp1a3a_Fish6_func-000",
        "20250911_atp1a3a_Fish8_func-000",
        # "20250911_atp1a3a_Fish9_func-000", # data corrupted in cluster plot... not sure what is wrong but need to ignore
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

    #% transform ROIs to z-brain coordiantes

    z_brain_2 = tifffile.imread(os.path.realpath(r'/media/BigBoy/ciqle/ref_brains/ZBrain2_0.tif')) 
    bigwarp_transform_folder = os.path.realpath(r'/media/BigBoy/ciqle/ref_brains/Atp1a3_bigwarp')
        
    z_brain_stack_path = r'/media/BigBoy/ciqle/ref_brains/HuC-H2BRFP_ZBrain.nrrd'
    z_brain_stack, z_brain_meta = nrrd.read(z_brain_stack_path)
    width_zbrain, height_zbrain, Zs_zbrain = z_brain_stack.shape
    z_brain_stack = np.moveaxis(z_brain_stack, [0,1,2], [2,1,0])

    xy_rez_zbrain = z_brain_meta['space directions'][0][0]
    z_rez_zbrain = z_brain_meta['space directions'][-1][-1]

    ref_brain_path = r'/media/FastDrive/atp1a3a_data/registration/telen_template_allfish_HuC-H2BGCaMP.nrrd' 
    ref_brain, ref_meta = nrrd.read(ref_brain_path)
    width, height, Zs = ref_brain.shape
    ref_brain = np.moveaxis(ref_brain, [0,1,2], [2,1,0])


    xy_rez = ref_meta['space directions'][0][0]
    z_rez = ref_meta['space directions'][-1][-1]

    

    z_brain_centroids = []

    centroid_records = []
    for idx, roi in enumerate(tqdm.tqdm(roi_stats)):
        centroid_refbrain = np.asarray(roi["centroid_refbrain"], dtype=float)
        centroid_records.append({
            "x_refbrain": centroid_refbrain[0] * xy_rez,
            "y_refbrain": centroid_refbrain[1] * xy_rez,
            "z_refbrain": centroid_refbrain[2] * z_rez,
        })

    centroids_df = pd.DataFrame(centroid_records)
    centroids_csv = os.path.join(bigwarp_transform_folder, "roi_centroids_refbrain.csv")
    centroids_df.to_csv(centroids_csv, index=False, header=False)
    print(f"Saved {len(centroid_records)} centroids → {centroids_csv}")

    # now I am using this script to transform the centroids in ImageJ using bigwarp into z-brain coordinates:
    # https://raw.githubusercontent.com/saalfeldlab/bigwarp/master/scripts/Apply_Bigwarp_Xfm_csvPts.groovy



    centroids_zbrain_file = os.path.join(bigwarp_transform_folder, "roi_centroids_zbrain.csv")
    centroids_zbrain = np.loadtxt(centroids_zbrain_file, delimiter=',')




    # put the centroids and shifted ROI definitions back into the roi_stats structure

    roi_stats_with_zbrain = roi_stats.copy()
    for idx, roi in enumerate(roi_stats_with_zbrain):
        centroid_zbrain = centroids_zbrain[idx]  
        centroid_zbrain = centroid_zbrain/np.array([xy_rez_zbrain, xy_rez_zbrain, z_rez_zbrain])  # convert to pixel coordinates
        centroid_zbrain = centroid_zbrain.astype(int).flatten()
        centroid_refbrain = roi["centroid_refbrain"]
        roi["centroid_zbrain"] = centroid_zbrain
        

        x_offsets, y_offsets, z_offsets = centroid_zbrain - centroid_refbrain

        roi['xpix_zbrain'] = roi['xpix_refbrain'] + x_offsets
        roi['ypix_zbrain'] = roi['ypix_refbrain'] + y_offsets

    crop_extents = [175, 450, 85, 550,  50, 110] # minX , maxX, minY, maxY, minZ, maxZ
    full_extents = [0, width_zbrain, 0, height_zbrain, 0, Zs_zbrain]

    # set up outlines image
    def make_cropped_outline(crop_extents):

        [xmin, xmax, ymin, ymax, zmin, zmax] = crop_extents

        crop_height = ymax - ymin
        crop_width = xmax - xmin
        crop_Zs = zmax - zmin   

        zbrain_outline_z = np.zeros((crop_height, crop_width))
        zbrain_outline_x = zoom(np.zeros((crop_Zs, crop_height)).T, [1, z_rez_zbrain/xy_rez_zbrain])

        z_brain_2_cropped = z_brain_2[zmin:zmax, ymin:ymax, xmin:xmax]
        IDs = [
            np.where(z_brain_2_cropped == 43), # olfactory epithelium
            np.where(z_brain_2_cropped == 29), # olfactory bulb
            np.where(z_brain_2_cropped == 30), # pallium
            np.where(z_brain_2_cropped == 31), # subpallium
            np.where(z_brain_2_cropped == 2), # habenula
            # np.where(z_brain_2_cropped == 23), # pretectum
            # np.where(z_brain_2_cropped == 118), # retina
            np.where(z_brain_2_cropped == 119), # spinal cord
            # np.where((z_brain_2_cropped >=27) & (z_brain_2_cropped <= 28)), # thalamus
            np.where((z_brain_2_cropped >=1) & (z_brain_2_cropped <= 28)), #  entire diencephalon
            # np.where((z_brain_2_cropped >=17) & (z_brain_2_cropped <= 18)), # posterior tuberculum
            np.where((z_brain_2_cropped >=111) & (z_brain_2_cropped <= 112)), # tectum
            # np.where((z_brain_2_cropped >=113) & (z_brain_2_cropped <= 115)), # tegmentum
            np.where((z_brain_2_cropped >=48) & (z_brain_2_cropped <= 110)), # hindbrain

            
        ]
        mask_3d = np.zeros((crop_Zs, crop_height, crop_width))

        for ids in IDs:
            mask_3d[:] = 0
            mask_3d[ids] = 1
            
            mask = np.max(mask_3d, axis=0)
            outline = morphology.distance_transform_edt(1-mask) == 1
            #outline = morphology.binary_dilation(outline, iterations=1)
            zbrain_outline_z[outline==1] =1

            mask = zoom(np.max(mask_3d, axis=2).T, [1, z_rez_zbrain/xy_rez_zbrain], order=0)
            outline = morphology.distance_transform_edt(1-mask) == 1
            #outline = morphology.binary_dilation(outline, iterations=1)
            zbrain_outline_x[outline==1] =1
        zbrain_outline_z[:, -1] = 1
        proj = np.hstack((zbrain_outline_z, zbrain_outline_x))
        proj = proj * 2
        proj = proj.astype(np.uint8)
        proj[proj>0] = 255
        proj[proj < 255] = 0
        outline = proj
        return outline

    outline_crop = make_cropped_outline(crop_extents)
    # plt.imshow(outline_crop, cmap='gray')
    # plt.title('Z-brain forebrain outlines _ crop')
    # plt.show()

    outline_full = make_cropped_outline(full_extents)
    # plt.imshow(outline_full, cmap='gray')
    # plt.title('Z-brain forebrain outlines _ full brain')
    # plt.show()

    metadata_zbrain = {'height': height_zbrain,
        'width': width_zbrain,
        'Zs': Zs_zbrain,
        'xy_rez': xy_rez_zbrain,
        'z_rez': z_rez_zbrain,
        'crop_extents': crop_extents,
        'full_extents': full_extents,
        'outline_crop': outline_crop,
        'outline_full': outline_full
    }   

    #

    IM_roi, im_proj = Ca2ImagingFns.draw_hit_volume_provideROIstats(np.arange(len(roi_stats_with_zbrain)), roi_stats_with_zbrain, metadata_zbrain, normalize=True, outline=outline_crop)

    #%

    plt.imshow(im_proj, cmap='inferno')
    plt.title('ROIs identified, Cropped to forebrain')
    plt.axis('off')
    plt.show()



    IM_roi, im_proj = Ca2ImagingFns.draw_hit_volume_provideROIstats(np.arange(len(roi_stats_with_zbrain)), roi_stats_with_zbrain, metadata_zbrain, crop_str='full_extents', normalize=True, outline=outline_full)

    #
    plt.imshow(im_proj, cmap='inferno')
    plt.title('ROIs identified, Full brain')
    plt.axis('off')
    plt.show()

    np.savez(os.path.join(processed_data_flds_path, 'ImagingData_allFish.npz'),
                        roi_stats=roi_stats_with_zbrain,
                        F=F, 
                        F_norm=F_norm, 
                        fish_data=fish_data, 
                        ops=ops,
                        metadata_zbrain = metadata_zbrain,
                    )

all_fish_data = np.load(os.path.join(processed_data_flds_path, 'ImagingData_allFish.npz'), allow_pickle=True)

roi_stats = all_fish_data['roi_stats']
metadata_zbrain = all_fish_data['metadata_zbrain'].item()
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



#%
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



#%

out_dir_heatmaps = os.path.join(out_dir, 'clustered_heatmaps')
os.makedirs(out_dir_heatmaps, exist_ok=True)
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
# Select first fish
if re_analyze:
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

            # ax_heatmap.set_aspect(traces_to_cluster_sorted.shape[1] / traces_to_cluster_sorted.shape[0])
            ax_heatmap.set_title(f"{fish_names[fish_ind]}\nFish Type = {matched_pairs[fish_ind][-1]}")
            for start, end in zip(label_starts, label_ends):
                ax_heatmap.hlines(start, xmin=0, xmax=traces_to_cluster_sorted.shape[1], colors="black", linestyles="--", linewidth=2.5)

            tick_step = 1000
            y_tick_positions = np.arange(0, traces_to_cluster_sorted.shape[0], tick_step)-1
            ax_heatmap.set_yticks(y_tick_positions + 0.5)
            ax_heatmap.set_yticklabels((y_tick_positions + 1).astype(int))
            fish_type = matched_pairs[fish_ind][-1]
            title_str = fish_names[fish_ind] + "\nFish Type = " + fish_type
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
                "fish_type": fish_type,
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

import numpy as np
cluster_results = np.load(os.path.join(out_dir, "cluster_results.npy"),allow_pickle=True)
#%%
out_dir_orderedheatmaps = os.path.join(out_dir, 'clustered_ordered_heatmaps')
os.makedirs(out_dir_orderedheatmaps, exist_ok=True)

selected_regressor_indices = [0, 1, 2]
selected_regressor_names = [regressor_names[i] for i in selected_regressor_indices]

enriched_results = []
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
    fish_type = matched_pairs[fish_ind][-1]
    print(fish_type)

    corr_threshold = 0.35  # adjust as needed

    reg_signals = np.asarray(cluster_results[fish_ind]["regressors_window"])[selected_regressor_indices, :]
    regressor_idx_map = {name: idx for idx, name in enumerate(selected_regressor_names)}
    ordered_labels = cluster_results[fish_ind]["cluster_order"]
    cluster_centroids = cluster_results[fish_ind]["cluster_centroids"]

    label_indices = {
        lbl: np.where(labels_sorted == lbl)[0]
        for lbl in ordered_labels
        if np.any(labels_sorted == lbl)
    }

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

    plt.savefig(os.path.join(out_dir_orderedheatmaps, safe_filename(title_str) + '.png'), dpi=300)
    plt.savefig(
        os.path.join(out_dir_orderedheatmaps, safe_filename(title_str) + ".svg"),
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





#%% Plot per fish category: std and mean fluorescence. This is a bit janky, so I will leave this for now and concentrate on clustering. 
reload(Ca2ImagingFns)
common_normalize = False

n_fish_in_category = [len(fish_dict[ft]) for ft in fish_dict.keys()]
fish_type_labels = ['WT', 'HET', 'MUT']


z_score_thresh = 2.5
# Define all metrics to plot
metrics = [
    # (np.nansum(abs(F_norm), axis=1), "Sum z_score", "viridis"),
    # (np.nanmean(F, axis=1), "Mean_F", "viridis"),
    # (np.nanstd(F_dff, axis=1), "Std_dF/F", "viridis"),
    (np.sum(F_norm > z_score_thresh, axis=1), "Active_frames_zscore_gt_1", "inferno"),
]

# --- Find global maximum for each metric ---

global_max_per_metric = []
for measure, _, _ in metrics:
    # For each metric, draw hit volume for all neurons and get max value in projection
    IM_rois, im_rois_proj = Ca2ImagingFns.draw_hit_volume_provideROIstats(np.arange(len(measure)), roi_stats, metadata_zbrain, values=measure, normalize=False)
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
        IM_rois, im_rois_proj = Ca2ImagingFns.draw_hit_volume_provideROIstats(inds_type, roi_stats, metadata_zbrain, values=values, outline=metadata_zbrain['outline_crop'], normalize=False)

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
        # ref_rgb = Ca2ImagingFns.to_rgb(ref_proj, cmap_name="gray", vmin=0, vmax=np.percentile(ref_proj, 95))
        v_max = norm_value if (norm_value is not None and not np.isnan(norm_value)) else np.nanmax(im_rois_proj_norm)
        rois_rgb = Ca2ImagingFns.to_rgb(im_rois_proj_norm, cmap_name=cmap, vmin=0, vmax=v_max * 0.5)

        # # Weighted additive blending
        # w_ref = 0.4
        # w_rois = 1.0
        # blended = np.clip(w_ref * ref_rgb + w_rois * rois_rgb, 0, 1)

        plt.figure(figsize=(20, 20))
        plt.imshow(rois_rgb)
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
blurr_size = 10 # in microns
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
#%
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
    valid_keys = []
    for a,b in pairs:
        if a not in groups["blurred"] or b not in groups["blurred"]:
            continue
        diff = (groups["blurred"][a] - groups["blurred"][b]).astype(np.float32)
        key = f"{a}_minus_{b}"
        groups["diffs"][key] = diff
        diff_projs[key] = project_stack(diff, proj_mean=True)
        valid_keys.append(key)
        # save diffs
        np.save(os.path.join(comparisons_dir, f"{safe_filename(measure_name)}_{key}.npy"), diff)
        try:
            tifffile.imwrite(os.path.join(comparisons_dir, f"{safe_filename(measure_name)}_{key}.tif"), diff)
        except Exception:
            pass

    # # visualize each diff projection using the magenta-black-green map
    # for key, proj in diff_projs.items():
    #     vmax = np.nanmax(np.abs(proj))
    #     plt.figure(figsize=(10, 5))
    #     plt.imshow(proj, cmap=cmap_mag_black_green, vmin=-vmax, vmax=vmax)
    #     plt.title(f"{measure_name}: {key} (blurred diff projection)")
    #     plt.axis("off")
    #     plt.colorbar(label="Δ intensity")
    #     plt.show()

    # visualize genotype comparisons side‑by‑side with shared LUT
    if diff_projs:
        vmax = max(np.nanmax(np.abs(proj)) for proj in diff_projs.values())
        fig, axes = plt.subplots(1, len(valid_keys), figsize=(6 * len(valid_keys), 5), constrained_layout=True)
        if len(valid_keys) == 1:
            axes = [axes]
        last_im = None
        for ax, key in zip(axes, valid_keys):
            proj = diff_projs[key]
            last_im = ax.imshow(proj, cmap=cmap_mag_black_green, vmin=-vmax, vmax=vmax)
            ax.set_title(f"{measure_name}: {key}")
            ax.axis("off")
        fig.colorbar(last_im, ax=axes, fraction=0.025, pad=0.04, label="Δ intensity")
        plt.show()
#%%


#%%buildTransform

roi_image = np.zeros(z_brain_2.shape, dtype=float)
for roi in roi_stats_with_zbrain:
    xpixs = roi['xpix_zbrain'].astype(int)
    ypixs = roi['ypix_zbrain'].astype(int)
    z = roi['centroid_zbrain'][2].astype('int')
    for i in range(len(xpixs)):

        roi_image[z, ypixs[i], xpixs[i]] +=1

plt.imshow(np.max(roi_image, axis=0), vmin=0, vmax=55,cmap='inferno')

#%%


