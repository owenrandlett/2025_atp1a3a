from numba import njit, prange
import numpy as np
from matplotlib.colors import hsv_to_rgb  # HSV → RGB conversion for cluster coloring


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

