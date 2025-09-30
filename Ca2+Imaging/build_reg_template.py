#%% use antspy envorinment on MeLiS analysis server

import ants, os, glob, shutil, subprocess
from natsort import natsorted

images_fld = os.path.realpath(r'/media/FastDrive/atp1a3a_data/registration/images')
images_list = natsorted(glob.glob(images_fld + '/*.nrrd'))

#%%
template_list = []
for image_name in images_list:
    print(os.path.split(image_name)[-1])
    image = ants.image_read(image_name)
    template_list.append(image)
#%%

args = {
    "type_of_transform": "SyNAgro"
}

timage = ants.build_template( image_list = template_list, kwargs = args)



bridge_template = r'/media/FastDrive/atp1a3a_data/registration/telen_template_allfish_HuC-H2BGCaMP.nrrd'
ants.image_write(timage, bridge_template)


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
