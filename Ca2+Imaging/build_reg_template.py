#%% use antspy envorinment on MeLiS analysis server

import ants, os, glob


images_fld = os.path.realpath(r'/media/FastDrive/atp1a3a_data/registration/images')
images_list = glob.glob(images_fld + '/*.nrrd')
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