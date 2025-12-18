#%%

# script to process anatomical stack taken before the Ca2+ imaging run, and then to run suite2p on the functional stack.



from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np

import suite2p
import xml.etree.ElementTree as ET

import glob
import os
import pandas as pd
import nrrd
from tifffile import imread, imsave
from PIL import Image
import tqdm
from natsort import natsorted
import h5py
import warnings
from scipy.ndimage import zoom

from uuid import uuid4
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.image import ImageSeries
from pynwb.ophys import (
    ImageSegmentation,
    MotionCorrection,
    RoiResponseSeries,
    TwoPhotonSeries,
    OpticalChannel,
)
from datetime import datetime
from dateutil.tz import tzlocal
from hdmf.backends.hdf5.h5_utils import H5DataIO
import gc
import traceback 
#%


x_rez = 0.5799 # in microns
y_rez = 0.5799

z_rez_anat = 1
z_rez_func = 10
imaging_dir = Path.cwd()

func_anat_separate = True # set to true if there are separate folders for the functional and the anaotomy stack (newer protocols), or False if the first cycles of the functional stack are the anatomy stack. This was what we did prior to using the stabilization routine on the 2p

if os.uname().nodename == 'MeLiS-7920':
    local_dir = os.path.realpath(r'/media/FastDrive')
else:
    local_dir = os.path.realpath(r'/mnt/md0')

#%
dataRoots = [
    os.path.realpath(r'/media/BigBoy/ciqle/2p/20250904_atp1a3aExperiments_Day3'), 
    os.path.realpath(r'/media/BigBoy/ciqle/2p/20250903_atp1a3aExperiments_Day2'),
    ]

if func_anat_separate: 
    file_search_string = '*ish*func*'
else: 
    file_search_string = '*fish*'


dataDirs = natsorted(glob.glob(dataRoots[0]+f'/{file_search_string}/'))

#

if len(dataRoots) > 1:
    for root in range(len(dataRoots)-1):
        dataDirs = dataDirs + natsorted(glob.glob(dataRoots[root+1]+f'/{file_search_string}/'))
#%
print('found the following data directories:')
for dir in dataDirs:
    print(dir)



out_data_root = os.path.join(local_dir, 'processed_2p_data')


# make dirs if they dont exist
Path(out_data_root).mkdir(parents=True, exist_ok=True)


#%%
def print_big_error(e):
    error_msg = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
    banner = f"""
        \033[1;41m{' ERROR! ':^60}\033[0m
        \033[1;31m{error_msg}\033[0m
        {'='*60}
    """
    print(banner)

for dataDir in tqdm.tqdm(dataDirs, desc = 'Processing data folders'):
    try:
        dataDir = os.path.realpath(dataDir)
        folder_name = os.path.basename(os.path.normpath(dataDir))
        nwb_folder = os.path.join(out_data_root, folder_name)
        Path(nwb_folder).mkdir(parents=True, exist_ok=True)



        #%
        os.chdir(dataDir)
        xml = glob.glob(dataDir+'/*.xml')[0] # first one seems to be the relevant one

        
        


        if not func_anat_separate:
            stim_file_name = glob.glob(dataDir+'/*VoltageRecording*.csv')[0]
            n_cyc_anat = 128
            tiff_files = natsorted(glob.glob('*_Ch2_*'))

            sl = imread(tiff_files[-1])
            y_size, x_size = sl.shape


            first_stack = [s for s in tiff_files if "_Cycle00001_Ch2_" in s]
            n_planes_anat = len(first_stack)



            anat_stack = os.path.join(nwb_folder,'AnatStack.nrrd')

            if Path(anat_stack).is_file():
                print('skipping anat stack writing__already exists')
                IM_anat, meta = nrrd.read(anat_stack)
                IM_anat = IM_anat.T
            else:

                IM = np.zeros((n_planes_anat, y_size, x_size), dtype='float')
                k = 0
                for i in tqdm.tqdm(range(n_cyc_anat)):
                    for j in range(n_planes_anat):
                        sl_name = tiff_files[k]
                        IM[j,:,:] = IM[j,:,:] + Image.open(sl_name)
                        k+=1
                    # plt.imshow(IM[50,:,:])
                    # plt.show()


                IM_anat = IM/np.max(IM)*65535
                IM_anat = IM_anat.astype('uint16')
                #%
                # write anatomy stack as NRRD for CMTK registration, with spacing information
                header = {'kinds': ['domain', 'domain', 'domain'], 'units': ['micron', 'micron', 'micron'], 'spacings': [x_rez, y_rez, z_rez_anat]} 
                nrrd.write(anat_stack, np.moveaxis(IM_anat, [0,1,2], [2,1,0]), header)
                imsave(anat_stack.replace('.nrrd', '.tif'), data=IM_anat)


            func_stat_frame = int(stim_file_name[stim_file_name.find('Cycle')+5:stim_file_name.find('Cycle')+10])
            cycle_start_func = 'Cycle'+(stim_file_name[stim_file_name.find('Cycle')+5:stim_file_name.find('Cycle')+10])
            first_func_ind = [i for i, e in enumerate(tiff_files) if cycle_start_func in e][0]

            fnames = tiff_files[first_func_ind:]
        
            print(fnames[0])
            single_cycle = glob.glob('*Cycle00500*')
            n_planes = len(single_cycle)
            n_frames = np.floor(len(fnames)/n_planes).astype(int)


            tree = ET.parse(xml)
            root = tree.getroot()
            #%
            t_stamps = np.zeros((n_planes, n_frames), dtype=float)
            for frame in range(func_stat_frame, n_frames+func_stat_frame):
                if frame == 0: # the first frame has two extra entries for the voltage recordings
                    adder = 3 
                else:
                    adder = 1

                for sl in range(n_planes):
                    t_stamps[sl, frame-func_stat_frame] = root[frame+1][sl+adder].attrib['relativeTime']



        else: # if we have separate folders for anatomy stack and functional stack

            anat_dir = dataDir.replace('func', 'anatomy')
            if not os.path.isdir(anat_dir):
                warnings.warn('\nno anatomy stack directory found for %s, there will be no anatomy stack for this experiment' % (anat_dir))
            else:
                os.chdir(anat_dir)
                tiff_files = natsorted(glob.glob('*_Ch2_*'))
                first_stack = [s for s in tiff_files if "_Cycle00001_Ch2_" in s]
                n_planes_anat = len(first_stack)
                n_cyc_anat = int(len(tiff_files)/n_planes_anat)


                sl = imread(first_stack[-1])
                y_size, x_size = sl.shape

                anat_stack = os.path.join(nwb_folder,'AnatStack.nrrd')

                if Path(anat_stack).is_file():
                    print('skipping anat stack writing__already exists')
                    IM_anat, meta = nrrd.read(anat_stack)
                    IM_anat = IM_anat.T
                else:
                    IM = np.zeros((n_planes_anat, y_size, x_size), dtype='float')
                    k = 0
                    for i in tqdm.tqdm(range(n_cyc_anat)):
                        for j in range(n_planes_anat):
                            sl_name = tiff_files[k]
                            IM[j,:,:] = IM[j,:,:] + Image.open(sl_name)
                            k+=1


                    IM_anat = IM/np.max(IM)*65535
                    IM_anat = IM_anat.astype('uint16')
                    #%
                    # write anatomy stack as NRRD for CMTK registration, with spacing information
                    header = {'kinds': ['domain', 'domain', 'domain'], 'units': ['micron', 'micron', 'micron'], 'spacings': [x_rez, y_rez, z_rez_anat]} 
                    nrrd.write(anat_stack, np.moveaxis(IM_anat, [0,1,2], [2,1,0]), header)
                    imsave(anat_stack.replace('.nrrd', '.tif'), data=IM_anat)



            #% now deal with the funcitonal stack 

            os.chdir(dataDir)
            fnames = natsorted(glob.glob('*_Ch2_*'))
            #%
            print(fnames[0])
            #

            single_cycle = glob.glob('*Cycle00500*')
            n_planes = len(single_cycle)
            n_frames = np.floor(len(fnames)/n_planes).astype(int)


            tree = ET.parse(xml)
            root = tree.getroot()


            t_stamps = np.zeros((n_planes, n_frames), dtype=float)
            for frame in range(n_frames):
                for sl in range(n_planes):
                    t_stamps[sl, frame] = root[frame+2][sl+1].attrib['relativeTime']



        # convert t_stamps to msec and get frame rate

        t_stamps = np.array(t_stamps * 1000, dtype=int)
        frame_period = np.mean(np.diff(t_stamps[0,:])) 
        frame_rate = 1000/frame_period
        print("frame rate = " + str(np.round(frame_rate, decimals=2)) + ' fps')






        #% create the NWB object



        nwb_filename = os.path.join(nwb_folder, '2p_Data_RAW.nwb')
        nwbfile = NWBFile(
            session_description=folder_name,
            identifier=str(uuid4()),
            session_start_time=datetime.now(tzlocal()),
            experimenter=[
                "Laurie Anne Lamire",
                "Owen Randlett"
            ],
            lab="Owen Randlett",
            institution="MeLiS, Lyon",
            experiment_description="Single block dark flash habituiation experiment",
            related_publications="doi:10.7554/eLife.84926.3",
        )


        device = nwbfile.create_device(
            name="Bruker 2p, CIQLE",
            description="two-photon microscope",
            manufacturer="Bruker",
        )

        optical_channel = OpticalChannel(
            name="Green Channel",
            description = "generic green PMT channel",
            emission_lambda=510.0,
        )


        #% now load in the voltage_recording, if we have a stimulus file

        if 'stim_file_name' in locals():


            stim_file = pd.read_csv(stim_file_name)

            stim_volts = stim_file.loc[:,' Input 3'].values
            stim_tstamps = stim_file.loc[:, 'Time(ms)'].values

            # create a 2d matrix to save the relevant voltage for each slice and each frame
            volt_stamps = np.zeros((n_planes, n_frames), dtype=float)

            #assume 5V if we didnt get a recording there
            volt_stamps[:] = 5

            print('sorting out stimulus trace')
            for i in tqdm.trange(n_frames):
                for j in range(n_planes):
                    t_stamp = t_stamps[j,i]
                    ind = np.where(stim_tstamps <= t_stamp)[0][-1]
                    volt_stamps[j,i] = stim_volts[ind]


            stim_vec = (5-np.mean(volt_stamps, axis=0).flatten())/5

            plt.plot(stim_vec)
            plt.show()
            print(np.shape(stim_vec))
            np.save(os.path.join(nwb_folder,'stimvec_mean.npy'), stim_vec)
            np.save(os.path.join(nwb_folder,'volt_stamps.npy'), volt_stamps)
            np.save(os.path.join(nwb_folder,'time_stamps.npy'), t_stamps)



            LED_ts = TimeSeries(
                name='Red LED stimulus',
                data=stim_vec,
                unit='a.u.',  # or 'presentation' / 'intensity' etc.
                timestamps = np.mean(t_stamps, axis=0),
                description='value of the red LED, 0-1 for full brightness range',
            )

            # Add to the stimulus group
            nwbfile.add_stimulus(LED_ts)


        for plane in range(n_planes):
            plane_name = f"plane {plane}"
            imaging_plane = nwbfile.create_imaging_plane(
                name=plane_name,
                description = "single plane of multiplane experiment, resonant scanner",
                optical_channel=optical_channel,
                imaging_rate=frame_rate,
                device=device,
                excitation_lambda=920.0,
                indicator="GCaMP",
                location="Pan-neuronal",
                grid_spacing=[y_rez, x_rez],
                grid_spacing_unit="microns",
                origin_coords=[0,0,plane*z_rez_func],
                origin_coords_unit="microns",
            )

        with NWBHDF5IO(nwb_filename, "w") as io:
            io.write(nwbfile)

        #%
        nwb_series = []
        for plane in tqdm.trange(n_planes, desc = folder_name):
            plane_name = f"plane {plane}"
            nwb_series.append(plane_name)
            with NWBHDF5IO(nwb_filename, "r+") as io:
                nwbfile = io.read()
                im_inds = np.arange(plane, len(fnames), n_planes).astype(int)
                im_inds = im_inds[:n_frames] # make sure we dont have an extra frame or two on the end
                IM = np.zeros((n_frames, y_size, x_size), dtype='uint16')

                for k, im_ind in enumerate(im_inds):
                    IM[k, :, :] = Image.open(fnames[im_ind])
                plt.imshow(np.mean(IM, axis=0))
                plt.title(f"{plane_name} from {folder_name}")
                plt.show()

                wrapped_data = H5DataIO(
                        data=IM,
                        compression='gzip',
                        chunks=(10, y_size, x_size)
                    )
                #%
                two_p_series = TwoPhotonSeries(
                    name=plane_name,
                    description="Raw 2p data",
                    data=wrapped_data,
                    imaging_plane=nwbfile.imaging_planes[plane_name],
                    unit="photons",
                    timestamps = t_stamps[plane,:]
                )
                nwbfile.add_acquisition(two_p_series)
                io.write(nwbfile)
            # Clean up explicitly
            del IM
            del two_p_series
            del nwbfile
            gc.collect()
    except Exception as e:
        print('error in %s' % (dataDir))
        print_big_error(e)
        if os.path.isfile(nwb_filename):
            os.remove(nwb_filename)



