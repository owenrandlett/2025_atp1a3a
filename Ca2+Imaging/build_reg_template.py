#%% use antspy envorinment on MeLiS analysis server

import ants, os, glob, shutil, subprocess
from natsort import natsorted

images_fld = os.path.realpath(r'/media/FastDrive/atp1a3a_data/registration/images')
images_list = natsorted(glob.glob(images_fld + '/*.nrrd'))
bridge_template = r'/media/FastDrive/atp1a3a_data/registration/telen_template_allfish_HuC-H2BGCaMP.nrrd'
zbrain_ref = r'/media/BigBoy/ciqle/ref_brains/HuC-H2BRFP_ZBrain_onlyTelenLeft.nrrd'
#%% build the common template bridge reference bran 
template_list = []
for image_name in images_list:
    print(os.path.split(image_name)[-1])
    image = ants.image_read(image_name)
    template_list.append(image)


args = {
    "type_of_transform": "SyNAggro"
}

timage = ants.build_template( image_list = template_list, kwargs = args)

ants.image_write(timage, bridge_template)

#%% load in the bridge template and align to zbrain ref brain

IM_bridge_template = ants.image_read(bridge_template)
IM_zbrain_ref = ants.image_read(zbrain_ref)

moving_dir = os.path.dirname(bridge_template)
out_dir = os.path.join(moving_dir, "registration_ANTs")

reg = ants.registration(
    fixed=IM_zbrain_ref, moving=IM_bridge_template,
    type_of_transform='SyNAggro',
    # affine multiresolution schedule (coarse->fine)
    aff_iterations=(2100, 1200, 1200, 10),
    aff_shrink_factors=(8,4,2,1),
    aff_smoothing_sigmas=(3,2,1,0),
    aff_metric='mattes',       # affine metric; ok for inter-sample intensity differences
    # SyN parameters
    reg_iterations=(200,100,50,20),   # nonlinear iterations (coarse->fine)
    syn_metric='CC',                  # cross-correlation (same-modality confocal)
    syn_sampling=2,                   # metric sampling
    grad_step=0.1,                    # integration step
    flow_sigma=3, total_sigma=0,      # smoothing parameters for updates
    outprefix=out_dir,
    verbose=True
)


#%%

# --- Inspect the results visually (optional) ---
ants.plot(IM_zbrain_ref, overlay=reg['warpedmovout'], overlay_alpha=0.7, title='Registered Result', axis=2)

#%%
# --- Prepare output folder ---

os.makedirs(out_dir, exist_ok=True)

# --- Define output paths ---
aligned_path = os.path.join(out_dir, "bridge_template_registered_wParams_CC.nii.gz")

# --- Save registered image ---
ants.image_write(reg['warpedmovout'], aligned_path)

params_dir = os.path.join(out_dir, "ANTs_Registration_Parameters")
os.makedirs(params_dir, exist_ok=True)

# The forward and inverse transforms returned by ANTsPy are file paths (to .mat and .nii.gz)
fwd_transform_paths = reg['fwdtransforms']
inv_transform_paths = reg['invtransforms']

# Copy transform files to the parameters subfolder
for tf_path in fwd_transform_paths + inv_transform_paths:
    if os.path.exists(tf_path):
        shutil.copy2(tf_path, params_dir)



print(f"\n✅ Registration complete!")
print(f"Aligned image saved to: {aligned_path}")
print(f"Transform files saved in: {params_dir}")




#%% register template to reference brain
ref_brain = r'/media/BigBoy/ciqle/ref_brains/HuC-H2BRFP_ZBrain_onlyTelenLeft.nrrd'

# move bridge_template into images subfolder and make copy to plat nice wiht munger
template_name = os.path.split(bridge_template)[-1]
dst_path = os.path.join(os.path.split(bridge_template)[0], 'images')
os.makedirs(dst_path, exist_ok=True)
shutil.copy2(bridge_template, dst_path)

src_file = os.path.join(dst_path, template_name)
dst_file = os.path.join(dst_path, template_name.replace('.nrrd', '_01.nrrd'))

shutil.copy2(src_file, dst_file)   # or use shutil.copy2(src_file, dst_file)

#%

cmd = 'cd ' + os.path.split(dst_path)[0] + ' && /home/melis/cmtk/build/bin/munger -v -awr 01 -X 52 -C 8 -G 80 -R 3 -A "--accuracy 0.4" -W "--accuracy 1.6" -s '+ ref_brain + ' "images"'
print(cmd)

pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
text =pipe.communicate()[0]
print(text)

# %%


import ants, os

def register_zebrafish_brain(moving_path, fixed_path, out_dir="registration_ANTs", live=True, verbose=True):
    """
    Register a zebrafish brain image to a reference (ZBrain) using Harold Burgess lab parameters.

    Parameters
    ----------
    moving_path : str
        Path to the moving image (your brain / bridge template).
    fixed_path : str
        Path to the fixed image (reference / ZBrain).
    out_dir : str, optional
        Output directory for registration results.
    live : bool, optional
        True = live sample (SyN[0.05,6,0.5]); False = fixed tissue (SyN[0.1,6,0]).
    verbose : bool, optional
        Print progress messages.

    Returns
    -------
    dict
        ANTsPy registration dictionary from the SyN stage (includes warped output and transforms).
    """

    os.makedirs(out_dir, exist_ok=True)

    # === Load images ===
    fixed = ants.image_read(fixed_path)
    moving = ants.image_read(moving_path)

    if verbose:
        print(f"\n--- Registering {moving_path} → {fixed_path} ---")
        print(f"Mode: {'LIVE' if live else 'FIXED'} tissue")

    # === Stage 1: Rigid ===
    rigid = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform='Rigid',
        metric='mattes',
        metric_weight=1,
        radius=32,
        sampling_strategy='Regular',
        sampling_percentage=0.25,
        reg_iterations=(200, 200, 200, 0),
        convergence_threshold=1e-8,
        convergence_window_size=10,
        shrink_factors=(12, 8, 4, 2),
        smoothing_sigmas=(4, 3, 2, 1),
        use_histogram_matching=False,
        verbose=verbose,
        outprefix=os.path.join(out_dir, "rigid_")
    )

    # === Stage 2: Affine ===
    affine = ants.registration(
        fixed=fixed,
        moving=rigid['warpedmovout'],
        type_of_transform='Affine',
        metric='mattes',
        metric_weight=1,
        radius=32,
        sampling_strategy='Regular',
        sampling_percentage=0.25,
        reg_iterations=(200, 200, 200, 0),
        convergence_threshold=1e-8,
        convergence_window_size=10,
        shrink_factors=(12, 8, 4, 2),
        smoothing_sigmas=(4, 3, 2, 1),
        use_histogram_matching=False,
        verbose=verbose,
        outprefix=os.path.join(out_dir, "affine_")
    )

    # === Stage 3: SyN nonlinear ===
    if live:
        grad_step, flow_sigma, total_sigma = 0.05, 6, 0.5   # SyN[0.05,6,0.5]
    else:
        grad_step, flow_sigma, total_sigma = 0.1, 6, 0      # SyN[0.1,6,0]

    syn = ants.registration(
        fixed=fixed,
        moving=affine['warpedmovout'],
        type_of_transform='SyN',
        grad_step=grad_step,
        flow_sigma=flow_sigma,
        total_sigma=total_sigma,
        metric='CC',
        metric_weight=1,
        radius=2,
        reg_iterations=(200, 200, 200, 200, 10),
        convergence_threshold=1e-7,
        convergence_window_size=10,
        shrink_factors=(12, 8, 4, 2, 1),
        smoothing_sigmas=(4, 3, 2, 1, 0),
        use_histogram_matching=False,
        verbose=verbose,
        outprefix=os.path.join(out_dir, "syn_")
    )

    # === Save output ===
    out_path = os.path.join(out_dir, "bridge_registered_burgessparams.nii.gz")
    ants.image_write(syn['warpedmovout'], out_path)
    if verbose:
        print(f"\nRegistration complete. Warped image saved to:\n  {out_path}\n")

    return syn

register_zebrafish_brain(bridge_template, zbrain_ref, out_dir=out_dir, live=False, verbose=True)