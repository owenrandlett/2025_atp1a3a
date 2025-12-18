from numba import njit, prange
import numpy as np
from matplotlib.colors import hsv_to_rgb  # HSV → RGB conversion for cluster coloring
from scipy.ndimage import zoom
import matplotlib.cm as cm

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


def cluster_hsv_palette(n_clusters, hue_start=0.07, hue_end=1.0, saturation=1.0):
    """
    Return a stable list of RGB colors for cluster-level plotting.

    The hue channel is swept uniformly across the specified hue range so the
    first cluster always starts at `hue_start` and subsequent clusters march
    forward around the HSV wheel.  Saturation is kept high (default 1.0) so
    colors pop against grayscale baselines; value (brightness) will later be
    modulated by the normalized activity traces.

    Parameters
    ----------
    n_clusters : int
        Number of distinct colors required (e.g., number of clusters).
    hue_start, hue_end : float
        Hue range (0–1).  By default we avoid the red notch at 0 to keep colors
        more distinct from the black/white background.
    saturation : float
        Constant saturation level per cluster.

    Returns
    -------
    palette_rgb : np.ndarray, shape (n_clusters, 3)
        RGB triplets (float32, 0–1) ordered by cluster index.
    """
    if n_clusters <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    hues = np.linspace(hue_start, hue_end, n_clusters, endpoint=False, dtype=np.float32)
    hsv = np.stack(
        [
            hues,                      # varying hue per cluster
            np.full_like(hues, saturation),  # fixed saturation
            np.ones_like(hues),        # full value (brightness) before modulation
        ],
        axis=1,
    )
    return hsv_to_rgb(hsv).astype(np.float32)


def draw_hit_volume_provideROIstats(hits_inds, roi_stats, ref_meta, crop_str = 'crop_extents', outline = None, values = [1], draw_centroid=False, add_write=True, proj_mean=True, save_name = None, normalize=True):
    height = ref_meta['height']
    width = ref_meta['width']
    Zs = ref_meta['Zs']
    xy_rez = ref_meta['xy_rez']
    z_rez = ref_meta['z_rez']
    [xmin, xmax, ymin, ymax, zmin, zmax] = ref_meta[crop_str]  # 

    IM_roi = np.zeros((Zs, height, width))
    for j in range(len(hits_inds)):
        roi_coords_y = roi_stats[hits_inds[j]]['ypix_zbrain'].astype('int')
        roi_coords_x = roi_stats[hits_inds[j]]['xpix_zbrain'].astype('int')
        roi_coords_z = roi_stats[hits_inds[j]]['centroid_zbrain'][2].astype('int')
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

    IM_roi = IM_roi[zmin:zmax, ymin:ymax, xmin:xmax]
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

    if outline is not None:
        im_proj[outline > 0.01] = np.max(im_proj)

    # if not save_name==None:
    #     imsave(os.path.join(analysis_out, save_name+'_proj_image.tif'), im_proj)
    return IM_roi, im_proj


def to_rgb(image, cmap_name="gray", vmin=None, vmax=None):
    
    """Convert scalar image to RGB using a colormap."""
    cmap = cm.get_cmap(cmap_name)
    normed = np.clip((image - (vmin if vmin is not None else image.min())) /
                     ((vmax if vmax is not None else image.max()) -
                      (vmin if vmin is not None else image.min()) + 1e-8), 0, 1)
    return cmap(normed)[..., :3]  # drop alpha channel