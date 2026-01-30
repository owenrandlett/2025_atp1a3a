#%%
import nrrd, napari, os, glob, natsort, datetime
import numpy as np
import matplotlib.pyplot as plt
from imageio import imwrite
from skimage.io import imsave, imread
from scipy.ndimage import zoom
from napari_animation import Animation
import datetime
#%
MAPMaps_to_load = {

    'names': [
        'Free Swimming_-,- vs +,+',
        'Dark Flashes_-,- vs +,+',
        'OMR_-,- vs +,+',
        'Intersection_-,- vs +,+',
    ],
    'paths': [
        r'/media/BigBoy/ciqle/tmp_analysis/atp1a3a/MAPMaps_atp1a3a/FreeSwim_atp1a3a -.-_over_atp1a3a +.+_SignificantDeltaMedians.tif',
        r'/media/BigBoy/ciqle/tmp_analysis/atp1a3a/MAPMaps_atp1a3a/Darkflash_atp1a3a -.-_over_atp1a3a +.+_SignificantDeltaMedians.tif',
        r'/media/BigBoy/ciqle/tmp_analysis/atp1a3a/MAPMaps_atp1a3a/OMR_atp1a3a -.-_over_atp1a3a +.+_SignificantDeltaMedians.tif',
        r'/media/BigBoy/ciqle/tmp_analysis/atp1a3a/MAPMaps_atp1a3a/intersection/Darkflash_atp1a3a -.-_over_atp1a3a +.+_AND__OMR_atp1a3a -.-_over_atp1a3a +.+_AND__FreeSwim_atp1a3a -.-_over_atp1a3a +.+_AND_.tif',
    ],
}


zbrain_dir = r'/media/BigBoy/ciqle/ref_brains/'
zbrain_im = imread(os.path.join(zbrain_dir, 'ZBrain2_0.tif'))

zbrain_shape = zbrain_im.shape
z_rez = 2
xy_rez = 0.798

def center_viewer(viewer):
    viewer.reset_view()

    layer = viewer.layers[0]  # or any layer of interest
    center = np.array(layer.data.shape)[:3] / 2
    viewer.camera.center = center

    # Get layer shape (Z, Y, X)
    layer_shape = np.array(layer.data.shape)[:3]

    # Get canvas size (height, width)
    canvas_size = np.array(viewer.window.qt_viewer.canvas.size) + 50
    canvas_shape = np.array([canvas_size[1], canvas_size[0]])

    # Calculate zoom factor for best fit (use min(canvas/layer))
    zoom_factor = np.min(canvas_shape / layer_shape[1:])  # layer_shape[1:] is (Y, X)
    viewer.camera.zoom = zoom_factor * 0.8

#%%
color_list = ['gray',  'red', 'cyan', 'blue',  'yellow', 'green', 'magenta',]
labels_added = 0

viewer = napari.Viewer()

# add zbrain regions

# viewer.add_image(zbrain_im,
#     # np.swapaxes(zbrain_im, 1, 2),
#     blending='additive',
#     name = 'ZBrain MECE regions',
#     scale=([z_rez/xy_rez, 1, 1]),
#     )

for i, file in enumerate(MAPMaps_to_load['paths']):
    stack = imread(file)

    zoom_factors = [zbrain_shape[0]/stack.shape[0], zbrain_shape[1]/stack.shape[1], zbrain_shape[2]/stack.shape[2], 1]
    stack_resized = zoom(stack, zoom_factors, order=0)
    
    # xy_rez = header['space directions'][0][0]
    # z_rez = header['space directions'][2][2]

    layer_pos = viewer.add_image(
        stack_resized[:,:,:,1],
        scale=([z_rez/xy_rez, 1, 1]),
        name = 'Pos_Signals' + MAPMaps_to_load['names'][i],
        colormap = 'green',
        blending='additive'
        )
    layer_neg = viewer.add_image(
        stack_resized[:,:,:,0],
        scale=([z_rez/xy_rez, 1, 1]),
        name = 'Neg_Signals' + MAPMaps_to_load['names'][i],
        colormap = 'magenta',
        blending='additive'
        )
    
    if not i == 0: # show onnly the first MAPMap
        layer_pos.visible = False
        layer_neg.visible = False
        
viewer.dims.order = [0, 2, 1]


#%% add some pre-defined stacks

stacks_list = {
    'names': [
        'H2BGCaMP_zBrain',
        'tac1_HCR',
        'penka_HCR_thiele',
        'penkb_HCR',
        'npy_HCR',

        ]
        ,
    'paths': [
        r'/media/BigBoy/Common/atp1a3a_Data/pERKData/ZBrainLabels/HuC-H2BRFP_ZBrain.nrrd',
        r'/media/BigBoy/Common/atp1a3a_Data/pERKData/ZBrainLabels/tac1_HCR_meanOf_9fish.nrrd',
        r'/media/BigBoy/Common/atp1a3a_Data/pERKData/ZBrainLabels/Elavl3-GCaMP5G_penka_02_warp_m0g80c8e1e-1x52r3.nrrd',
        r'/media/BigBoy/Common/atp1a3a_Data/pERKData/ZBrainLabels/penkb_HCR_meanOf_9fish.nrrd',
        r'/media/BigBoy/Common/atp1a3a_Data/pERKData/ZBrainLabels/npy_HCR_meanOf_10fish.nrrd',
        ]

}



for i in range(len(stacks_list['names'])):
    viewer.add_image(imread(stacks_list['paths'][i]),
        # np.swapaxes(zbrain_im, 1, 2),
        blending='additive',
        name = stacks_list['names'][i],
        scale=([z_rez/xy_rez, 1, 1]),
        colormap = color_list[labels_added%len(color_list)],
        )
    labels_added+=1
#%
# for layer in viewer.layers:
#     layer._keep_auto_contrast = True

#%% run dialog to add stacks

from qtpy.QtWidgets import QFileDialog


default_dir = os.path.realpath('/media/BigBoy/ciqle/ref_brains/AnatomyDatabases/')  # no r'' prefix
file_paths, _ = QFileDialog.getOpenFileNames(
    None,
    "Select stack files",
    default_dir,
    "Image Files (*.tif *.tiff *.nrrd *.png *.jpg);;All Files (*)"
)
for file_path in file_paths:
    stack = imread(file_path)
    viewer.add_image(
        stack,
        blending='additive',
        name=os.path.basename(file_path),
        scale=([z_rez/xy_rez, 1, 1]),
        colormap = color_list[labels_added%len(color_list)],
    )
    labels_added+=1


#%% crop to subvolume


# Telencephalon
z_min, z_max = 25, 300
y_min, y_max = 95, 360
x_min, x_max = 185,440

# Save original data before cropping
original_data = {layer.name: layer.data.copy() for layer in viewer.layers}

# Crop
for layer in viewer.layers:
    arr = layer.data
    layer.data = arr[z_min:z_max, y_min:y_max, x_min:x_max]
center_viewer(viewer)
#%%
# Uncrop (restore original data)
for layer in viewer.layers:
    layer.data = original_data[layer.name]

#%%

for layer in viewer.layers:
    layer.visible = False
    
tstamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')
for layer in viewer.layers:
    layer.visible = True
    name = layer.name
    col_orig = layer.colormap.name
    layer.colormap = 'gray'
    screen_shot = viewer.screenshot()
    plt.imshow(screen_shot)
    plt.title(name)
    plt.show()
    layer.colormap = col_orig
    layer.visible = False
    save_name = 'ScreenShot_' + tstamp + name + '.png'
    imwrite(save_name, screen_shot)


#%% record a video of the viewer


center_viewer(viewer)

out_dir = r'/media/BigBoy/Common/atp1a3a_Data/pERKData/napari_videos'
anim = Animation(viewer)


# Go to first slice

viewer.dims.set_point(0, 0)
anim.capture_keyframe()

# Go to last slice
viewer.dims.set_point(0, z_rez/xy_rez * zbrain_shape[0])
anim.capture_keyframe(steps = zbrain_shape[0]-1)

tstamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')
video_name = f'scroll_{tstamp}.mp4'

# Export video
anim.animate(os.path.join(out_dir, video_name), fps=15)

#%%


out_dir = r'/media/BigBoy/Common/atp1a3a_Data/pERKData/napari_videos'
anim = Animation(viewer)
center_viewer(viewer)
anim.capture_keyframe()

ang1 , ang2 , roll = viewer.camera.angles
for i in range (4):
    viewer.camera.angles = (viewer.camera.angles[0], viewer.camera.angles[1], roll + (i+1) * 90)
    anim.capture_keyframe(steps = 45)

# Add timestamp to filename
tstamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')
video_name = f'rotate_barbecue_{tstamp}.mp4'

# Export video
anim.animate(os.path.join(out_dir, video_name), fps=23, quality=7)


#%% concatenate videos

from moviepy.editor import VideoFileClip, concatenate_videoclips

# List your mp4 files
dir = out_dir

# mp4_files = [
#     'scroll_InstersectionPlusHuCHB_20260122_103953_.mp4',
#     'scroll_Itersection_AllLabels_andHuCH2B_20260122_104615_.mp4',
#     'scroll_Itersection_AllLabels_20260122_104916_.mp4',
# ]

mp4_files = [
    'rotate_barbecue_Intersection_HuCH2B_20260122_105222_.mp4',
    'rotate_barbecue_Intersection_AllLables_HuCH2B20260122_111026_.mp4',
    'rotate_barbecue_Intersection_AllLables_20260122_105547_.mp4',
    'rotate_barbecue_IntersectionPositive_NPY_20260122_110058_.mp4',
    'rotate_barbecue_IntersectionNegative_penk_tac1_20260122_110623_.mp4',

]

# Load clips
clips = [VideoFileClip(os.path.join(dir, f)) for f in mp4_files]

# Concatenate
final_clip = concatenate_videoclips(clips)


tstamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')
video_name = f'Concatenated_video_{tstamp}.mp4'
# Save result
final_clip.write_videofile(os.path.join(out_dir, video_name), codec='libx264')