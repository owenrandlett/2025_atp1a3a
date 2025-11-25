#%%

import os
import glob
import shutil
from pathlib import Path
import fireants as fa
import fireants.io
from fireants.registration import Registration  # Explicitly import Registration
from natsort import natsorted

# --- Configuration ---
images_fld = Path('/media/FastDrive/atp1a3a_data/registration/images')
images_list = natsorted(list(images_fld.glob('*.nrrd')))

# Modified output filename for comparison
bridge_template = Path('/media/FastDrive/atp1a3a_data/registration/telen_template_allfish_HuC-H2BGCaMP_fireants.nrrd')
zbrain_ref = Path('/media/BigBoy/ciqle/ref_brains/HuC-H2BRFP_ZBrain_onlyTelenLeft.nrrd')

# --- 1. Build Template (Optional) ---
rebuild_template = False

if rebuild_template:
    # Note: TemplateBuilder requires 'fireants.scripts' or similar depending on version
    pass 

# --- 2. Register to ZBrain ---
re_align_to_zbrain = True

if re_align_to_zbrain:
    print(f"Loading fixed: {zbrain_ref}")
    print(f"Loading moving: {bridge_template}")

    # Load images using FireANTs Image class methods
    fixed_img = fa.io.Image.load_file(str(zbrain_ref))
    moving_img = fa.io.Image.load_file(str(bridge_template))

    # Setup output directories
    moving_dir = bridge_template.parent
    out_dir = moving_dir / "registration_fireants"
    out_dir.mkdir(exist_ok=True)
    
    params_dir = out_dir / "FireANTs_Registration_Parameters"
    params_dir.mkdir(exist_ok=True)

    # Initialize Registration using imported class
    reg = Registration(fixed_img, moving_img)

    # --- Stage 1: Affine ---
    reg.add_affine(
        scales=[8, 4, 2, 1],
        iterations=[2100, 1200, 1200, 10],
        smoothing=[3, 2, 1, 0],
        metric=fa.CCMetric(), # Note: If this fails, check fireants.losses or fireants.registration for CCMetric
    )

    # --- Stage 2: SyN ---
    reg.add_syn(
        scales=[8, 4, 2, 1],
        iterations=[200, 100, 50, 20],
        smoothing=[3, 2, 1, 0],
        metric=fa.CCMetric(),
        step_size=0.1,
    )

    print("Starting registration (this may take a moment)...")
    
    # Run optimization
    warped_img, fwd_transforms, inv_transforms = reg.optimize()

    # --- Save Results ---
    aligned_path = out_dir / "bridge_template_registered_fireants.nii.gz"
    print(f"Saving warped image to: {aligned_path}")
    # Save using the image object's save method
    warped_img.save(str(aligned_path))

    # --- Save Transforms ---
    print(f"Saving transforms to: {params_dir}")
    
    for i, tf in enumerate(fwd_transforms):
        name = f"fwd_transform_{i}_{tf.__class__.__name__}"
        if hasattr(tf, 'save'):
            save_path = params_dir / (name + ".nii.gz")
            tf.save(str(save_path))
        else:
            save_path = params_dir / (name + ".txt")
            try:
                if hasattr(tf, 'save'):
                    tf.save(str(save_path))
                else:
                    print(f"   Info: Transform {name} does not have a save method.")
            except Exception as e:
                print(f"   Warning: Could not save transform {name}: {e}")

    print("\n✅ FireANTs Registration complete!")
    #%%