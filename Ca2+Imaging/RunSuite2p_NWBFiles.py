
#%%
import suite2p, glob, os, pynwb, inspect, shutil
from pynwb import NWBHDF5IO
import numpy as np
from natsort import natsorted
from pathlib import Path



rerun_suite2p = False # force running of suite2p again? if not, will not re-run analysis if we have already got a sub-folder called "suite2p"


if os.uname().nodename == 'MeLiS-7920':
    local_dir = os.path.realpath(r'/media/FastDrive')
    imaging_python_dir = os.path.realpath(r'/home/zeneb/github/imaging')
else:
    local_dir = os.path.realpath(r'/mnt/md0')
    imaging_python_dir = os.path.realpath(r'/home/zeneb/github/imaging')
    imaging_python_dir = os.path.realpath(r'/home/lab/imaging')

dataRoots = [
    local_dir + r'/processed_2p_data/',
]

nwb_files = natsorted(glob.glob(dataRoots[0] + r'/*/2p_Data_RAW.nwb'))
# nwb_files = nwb_files[-1::-1]
print('asdfsdf')
print('found nwb files:')
for file in nwb_files:  
    print(file)

#%%
for nwb_filename in nwb_files:
    nwb_folder = os.path.dirname(nwb_filename)
    nwb_filename = glob.glob(nwb_folder + r'/*.nwb')[0]

    if os.path.isdir(os.path.join(nwb_folder, 'suite2p')):
        print(f"\033[91mWARNING: SKIPPING ANALYSIS \n'suite2p' folder exists in {nwb_folder}\033[0m")
        
    else:

        twophoton_series_names = []
        frame_rates = []
        with NWBHDF5IO(nwb_filename, 'r') as io:
            nwbfile = io.read()
            
                # Get all TwoPhotonSeries objects
            for name, obj in nwbfile.acquisition.items():
                if isinstance(obj, pynwb.ophys.TwoPhotonSeries):
            
                    twophoton_series_names.append(name)
                    # Access timestamps
                    timestamps = obj.timestamps
                    if timestamps is not None:
                        frame_periods = np.diff(timestamps[:])
                        frame_rates.append(1000 / np.mean(frame_periods))
            # Get all TwoPhotonSeries names
        twophoton_series_names = natsorted(twophoton_series_names)
        frame_rate = np.mean(frame_rates)
        n_planes = len(twophoton_series_names)

        print('for experiment :', os.path.split(nwb_folder)[-1])
        print('found n_planes:', n_planes)
        print('found frame rate:', frame_rate)


        save_paths = []
        for series in twophoton_series_names:
            save_path = os.path.join(nwb_folder, series)
            save_paths.append(save_path)
            ops = suite2p.default_ops()
            ops['nwb_file'] =  nwb_filename
            ops['save_NWB'] = False
            ops['tau'] = 1.5
            ops['fs'] = frame_rate
            ops['diameter'] = 8
            ops['sparse_mode'] = True
            ops['spatial_scale'] = 1
            ops['nplanes'] = n_planes
            ops['functional_chan'] = 2
            ops['high_pass'] = 300
            ops['nbinned'] = 7000
            ops['data_path'] = [nwb_folder + r'/']
            ops['batch_size'] = 500
            ops['nimg_init'] = 1000
            ops['look_one_level_down'] = 1
            ops['max_iterations'] = 50
            ops['denoise'] = True
            ops['do_registration'] = 1
            ops['nonrigid'] = True
            ops['keep_movie_raw'] = False
            ops['neuropil_extract'] = True
            ops['delete_bin'] = True
            ops['classifier_path'] = os.path.join(imaging_python_dir, 'classifiers', 'HuC-H2BGCaMP7.npy')

            ops['save_path0'] = save_path
            ops['nwb_series'] = series
            # suite2p.io.nwb_to_binary(ops)
            output_ops = suite2p.run_s2p(ops=ops)


        s2p_dir = os.path.join(nwb_folder, 'suite2p')
        Path(s2p_dir).mkdir(parents=True, exist_ok=True)

        for k, outpath in enumerate(save_paths):
            plane_data_folder = os.path.join(os.path.realpath(outpath), 'suite2p/plane0/')
            new_folder = os.path.join(s2p_dir, os.path.split(outpath)[-1])
            shutil.copytree(plane_data_folder, new_folder)

            # nwb_output_file = os.path.join(os.path.realpath(outpath), 'suite2p/ophys.nwb')
            # shutil.copy(nwb_output_file, new_folder)


        (stat, ops, F, Fneu, spks, iscell_0, iscell_1, redcell_0, redcell_1, hasred) = suite2p.io.combined(s2p_dir, save=True)

        combined_out_folder = ops["save_path"]
        shutil.move(combined_out_folder, os.path.join(s2p_dir, 'combined/'))


        # clean up folders
        for folder in save_paths:
            shutil.rmtree(folder)


#%% load in planes to view from the NWB file
import napari
import zarr
nwb_file = '/media/FastDrive/processed_2p_data/20220802_HuCGCaMP7f_5dpf_CGP7930_fish3_func-000/2p_Data_RAW.nwb'

plane_ind = 5
with NWBHDF5IO(nwb_filename, 'r') as io:
    nwbfile = io.read()
    twophoton_series = nwbfile.acquisition[twophoton_series_names[plane_ind]]
    
    # Load the data 
    data = twophoton_series.data[:] 

# Launch Napari viewer
viewer = napari.Viewer()
viewer.add_image(data, name=plane_ind)

napari.run()

