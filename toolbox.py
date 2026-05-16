'''
Toolbox of various functions

# List of Functions

## Loading and Preprocessing
- load_dataset()
- preprocess_dataset()
- interpolate_dataset()
'''

import xarray as xr
import numpy as np

import gsw_xarray as gsw

import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from functools import wraps

### --------------------
### MISC EXTRA FUNCTIONS
### --------------------

# to be reclassified once i have time

def plot_stats(arr, ax=None, **plot_kw):
    '''
    plots stats; assumes we have shape (n_values, n_runs)
    '''

    if ax == None:
        fig, ax = plt.subplots(figsize=(8, 5))

    mean = arr.mean(axis=1)
    std = arr.std(axis=1)

    plt_range = range(2, len(mean)+2)

    im = ax.errorbar(
        plt_range, mean, yerr=std, 
        marker='o', markersize=2,
        linewidth=0.7,
        ecolor='black', elinewidth=0.7, capsize=0.5,
        **plot_kw
    )

    ax.set_xticks(plt_range)

    return im

def random_split_arr(arr, qty):
    '''
    Splits array (n_samples, n_features) into qty number of random arrays
    '''

def random_subset(arr, qty):
    '''
    Given an array in the shape (n_samples, n_features), subsample this to get a random subarray (qty, n_features)
    '''

    idx_size = arr.shape[0]

    idxs = np.random.choice(range(idx_size), size=qty, replace=False)

    return np.sort(arr[idxs])

def rmsd(vec_1, vec_2):
    '''
    Calculates Root Mean Square Deviation (RMSD)
    '''

    return np.sqrt(
        (vec_1 - vec_2) ** 2
    ).sum(axis=1).mean() ## sum along pressure axis so we get a total rtsqr difference, then take mean across all profiles
    

def true_runs(arr):
    '''
    Finds the lengths of true runs in an array (1 or 2d)
    '''
    arr = np.asarray(arr, dtype=bool)
    # Find boundaries where True starts and ends

    if len(arr.shape) == 2:
        ## Add a row of False to break up rollovers
        ## ravel to make it 1D
        cap = np.zeros((arr.shape[0], 1), dtype=bool)

        arr = np.concatenate((arr, cap), axis=1).ravel()
    
    diff = np.diff(arr.astype(int))
    #print(f'diff: {diff}')
    starts = np.where(diff == 1)[0] + 1
    #print(f'starts: {starts}')
    ends   = np.where(diff == -1)[0] + 1
    #print(f'ends: {ends}')

    if arr[0]:
        starts = np.r_[0, starts]
    if arr[-1]:
        # ends = np.r_[ends, len(arr)]
        ## if it ends in true, we want to remove the final entry from starts
        starts = np.delete(starts, -1)

    return ends - starts

### --------------
### NOTEBOOK SETUP
### --------------

# nothing here just yet

### ----------------------
### PROJECT SPECIFIC FUNCS
### ----------------------

DATA_DEFAULT_PATH = './data/ACOD_CTD_v1.0.nc'
ALL_REGIONS = ['Chukchi Sea', 'Gulf of Alaska', 'Bering Sea']

def load_dataset(
    ds_path = DATA_DEFAULT_PATH, 
    preprocess = True
):
    '''
    Loads the ACOD main CTD dataset; I think the path needs to be set to the relative path from the file this is being run from to the dataset (by default it assumes data is stored in the `./data` directory)

    Args:
        ds_path (str or Path-like): Path to data; default is `./data/ACOD_CTD_v1.0.nc`
        preprocess (bool): Whether to preprocess data. See docs for `preprocess_dataset()` for more info

    Returns:
        ds (Dataset): Dataset opened with Xarray
    '''

    ds = xr.open_dataset(ds_path)

    if preprocess:
        ds = preprocess_dataset(ds)

    return ds

def preprocess_dataset(
    ds,
    merge_salinity = True,
    combine_region_name = True,
    
):
    '''
    Performs various preprocessing tasks, each of which has its own toggle.

    Args:
        ds (Dataset): Base dataset, should be ACOD_CTD_v1.0.nc by default
        merge_salinity (bool): Whether to merge PPT and PSS Salinity values. Default: True
        combine_region_name (bool): Whether to turn the region names into a readable form. Default: True 

    Returns:
        new_ds (Dataset): Preprocessed dataset
    '''
    new_ds = ds.copy()

    if merge_salinity:
        new_ds['Merged_Salinity'] = ds.Salinity_PPT_with_QC_applied.combine_first(ds.Salinity_PSS_with_QC_applied)


    if combine_region_name:
        new_ds['Region_Name'] = ds.REGION.sum(axis=0)

    return new_ds
    
def interpolate_dataset(
    ds,
    max_gap = 5,
    keep_vars = ['BOTTOM_DEPTH', 'Region_Name'],
    temperature_var = 'Temperature',
    salinity_var = 'Merged_Salinity'
):
    '''
    Interpolates Temperature and Salinity values along the PRESSURE axis. By default, will get rid of all other variables.
    Extra variables (eg BOTTOM_DEPTH) can be kept via `keep_vars`

    Args:
        ds (Dataset): Base dataset
        max_gap (int): Maximum number of `nan`s that will be interpolated - gaps larger than this will be skipped
        additional_vars (List): List of additional variables to be kept. These will not be interpolated
        temperature_var (str): Name of temperature variable
        salinity_var (str): Name of salinity variable

    Returns:
        new_ds (Dataset): Interpolated dataset
    '''

    new_ds = ds[[temperature_var, salinity_var]]

    new_ds = new_ds.interpolate_na(dim='PRESSURE', max_gap = max_gap)

    for var in keep_vars:
        new_ds[var] = ds[var]

    return new_ds

def drop_region(
    ds,
    region_choice = 'Bering Sea'
):
    '''
    Drop all profiles in the region, `region_choice`

    Args:
        ds (Dataset): Base dataset
        region_choice (str): Must be a part of `ALL_REGIONS`

    Returns
        reduced_ds (Dataset): Dataset without `region_choice`
    '''

    assert region_choice in ALL_REGIONS, f'Invalid region: {region_choice}' 

    reduced_ds = ds.where(ds.Region_Name != f'{region_choice:<14}', drop=True)

    return reduced_ds

### =====================
### DATA PROCESSING FUNCS
### =====================

def normalise_array(
    arr,
    return_stats = False
):
    '''
    Normalises an array by mean-centering then dividing by standard deviation

    Args:
        arr (array-like): Array in the order (n_features, n_samples)

    Returns:
        normalised_arr (array-like)
    '''

    mean = arr.mean(axis=0)
    std = arr.std(axis=0)

    if return_stats:
        return np.divide(arr - mean, std), (mean, std)
    return np.divide(arr - mean, std)

def clean_dataset(
    ds,
    temperature_var = 'Temperature', salinity_var = 'Merged_Salinity',
    verbose = False, return_mask = False
):
    '''
    Removes all profiles with any nan values. This is the default cleaning step.

    Args:
        ds (Dataset): Base dataset
        temperature_var (str): Temperature Variable name
        salinity_var (str): Salinity variable name
        verbose (bool): Whether to output debugging messages

    Returns:
        clean_ds (Dataset): Clean dataset
    '''

    temp_mask = ~np.any(np.isnan(ds[temperature_var]), axis=0) 
    sal_mask = ~np.any(np.isnan(ds[salinity_var]), axis=0)
    
    nan_mask = temp_mask & sal_mask
    
    clean_ds = ds.where(nan_mask, drop=True)

    if verbose:
        
        print(f'[clean_dataset] Dropped {len(nan_mask)-sum(nan_mask).data}/{len(nan_mask)} values ({len(nan_mask)-sum(nan_mask).data/len(nan_mask):.02f}%)')

    if return_mask:
        return clean_ds, nan_mask
    return clean_ds
    
def prepare_training_data(
    ds,
    min_depth = None, max_depth = None,
    normalise = True, clean = True,
    temperature_var = 'Temperature', salinity_var = 'Merged_Salinity',
    verbose = False
):
    '''
    Transforms dataset to numpy ndarray in the shape (n_samples, n_features) to be used for clustering etc

    Args:
        ds (Dataset): Base dataset
        temperature_var (str): Temperature Variable name
        salinity_var (str): Salinity variable name
        normalise (bool): Whether the data needs to be normalised

    Returns:
        training_data (np.ndarray): Training data in shape (n_samples, n_features)
    '''

    if min_depth or max_depth:
        ds = ds.sel(PRESSURE=slice(min_depth, max_depth))
    
    if clean:
        clean_ds = clean_dataset(ds, verbose=verbose)
    else:
        clean_ds = ds

    # depth = len(clean_ds.PRESSURE)

    temp = clean_ds[temperature_var].data.T
    sal = clean_ds[salinity_var].data.T

    training_data = np.concatenate((temp, sal), axis=1)

    if normalise:
        training_data = normalise_array(training_data)

    if verbose:
        print(f'[prepare_training_data] Training data of shape: {training_data.shape}')

    if clean:
        return training_data, clean_ds
    else:
        return training_data

def select_region(
    ds,
    longitude_range=None, latitude_range=None,
    lon_var = 'LONGITUDE', lat_var = 'LATITUDE',
):
    '''
    Restricts a dataset to only those values within the longitude and latitude ranges given

    Args:
        ds (Dataset): Base dataset
        longitude_range (tuple[float, float]): Range of longitudes. Should lie in the range (0, 360)
        latitude_range (tuple[float, float]): Range of latitudes. Should lie in the range (-90, 90)
        lon_var (str): Name of longitude variable
        lat_var (str): Name of latitude variable

    Returns:
        regional_ds (Dataset): Restricted dataset
    '''
    
    n_profiles = ds.PROFILE.shape[0]

    if longitude_range:
        lon_mask = (ds[lon_var] > longitude_range[0]) & (ds[lon_var] < longitude_range[1])
    else:
        lon_mask = np.ones(n_profiles, dtype=bool)

    if latitude_range:
        lat_mask = (ds[lat_var] > latitude_range[0]) & (ds[lat_var] < latitude_range[1])
    else:
        lat_mask = np.ones(n_profiles, dtype=bool)

    mask = lon_mask & lat_mask
    regional_ds = ds.where(mask, drop=True)

    return regional_ds

def enforce_2d_arr(arr, dim=0):
    '''
    if array 1d, make it 2d
    '''

    if len(arr.shape) == 1:
        if dim == 0:
            arr = arr.reshape(-1, 1)
        elif dim == 1:
            arr = arr.reshape(1, -1)
        else:
            raise ValueError('Invalid dim')

    return arr

### ====================
### GSW DEFAULT SETTINGS
### ====================

# functions to make gsw quicker

def calculate_sigma_t(
    ds,
    temperature_var = 'Temperature',
    salinity_var = 'Merged_Salinity',
    pressure_var = 'PRESSURE', longitude_var = 'LONGITUDE', latitude_var = 'LATITUDE',
    as_var = None, ct_var = None
):
    '''
    Calculates the density anomaly ($\\sigma_t$) using gsw

    Is just a wrapper with some default settings basically

    Args:
        ds
        temperature_var: in-situ temperature
        salinity_var: practical salinity
        pressure_var: pressure

    Returns:
        sigma_t (DataArray) or possibly ndarray idk: the density anomaly as compared to 0dbar 
    '''

    pressure = ds[pressure_var]
    
    if as_var:
        assert True # add check that this is an actual variable
        
        asal = ds[as_var]

    else:

        psal = ds[salinity_var]
        lon = ds[longitude_var]
        lat = ds[latitude_var]

        asal = gsw.conversions.SA_from_SP(psal, pressure, lon, lat)

    if ct_var:
        assert True # check is real var

        ctemp = ds[ct_var]

    else:
        temp = ds[temperature_var]

        ctemp = gsw.conversions.CT_from_t(asal, temp, pressure)
    
    sigma_t = gsw.density.sigma0(asal, ctemp)
    
    return sigma_t

### ==============
### PLOTTING FUNCS
### ==============

# these probably need to be moved to viz

def geoaxes(
    n_rows=1, n_cols=1, 
    projection = None,
    feature_alpha = 1,
    **mpl_kw
):
    '''
    Creates a set of GeoAxes (on which maps etc can be plotted)

    Args:
        n_rows (int): Number of rows. Default = 1
        n_cols (int): Number of columns. Default = 1

    Returns:
        fig, ax(s) 
    '''

    assert n_rows >= 1
    assert n_cols >= 1

    if projection == None:
        projection = ccrs.AlbersEqualArea(central_longitude=180, central_latitude=60)

    fig, axs = plt.subplots(
        n_rows, n_cols, **mpl_kw, 
        subplot_kw={'projection': projection}
    )

    if n_rows + n_cols > 2:
        axs = axs.ravel()

    if n_rows == 1 and n_cols == 1:
        axs.add_feature(cfeature.COASTLINE, alpha=feature_alpha)
        axs.add_feature(cfeature.BORDERS, alpha=feature_alpha)
        axs.add_feature(cfeature.LAND, alpha=feature_alpha)

    else:
        for ax in axs:
            ax.add_feature(cfeature.COASTLINE, alpha=feature_alpha)
            ax.add_feature(cfeature.BORDERS, alpha=feature_alpha)
            ax.add_feature(cfeature.LAND, alpha=feature_alpha)
            
    return fig, axs

# techincally GeoAxes is cartopy.mpl.geoaxes.GeoAxes
def plot_all_vals(
    ds, color=None, 
    LON_VAR='LONGITUDE', LAT_VAR='LATITUDE', 
    subplots = None, ax=None,
    alpha = 0.6, pt_size=1,
    cbar_kwargs = dict(),
    transform = None,
    **plot_kw
):
    '''
    Plots all points (latitude-longitude) from a dataset; gives a general gist of where the data lies. 

    Args:
        ds (xr.Dataset): Dataset of values
        subplots (plt.figure, GeoAxes; optional): Tuple containing figure and GeoAxes
        color (string; optional): Variable to color by

    Returns:
        im (matplotlib scatter plot)
    '''

    if subplots == None:
        if ax:
            fig = None
        else:
            fig, ax = geoaxes()
    else:
        fig, ax = subplots

    if transform == None:
        transform = ccrs.PlateCarree()

    lons = ds[LON_VAR]
    lats = ds[LAT_VAR]

    im = ax.scatter(
        lons, lats, s=pt_size, c=color, alpha=alpha, transform=transform, **plot_kw
    )

    if type(color) == 'str':
        # probably add cluster to legend
        pass

    elif type(color) == xr.core.dataarray.DataArray or type(color) == np.ndarray:
        assert fig, 'Need to give fig and ax: use `subplots=(fig, ax)`'
        fig.colorbar(im, location='bottom', **cbar_kwargs)

    return im

### ===============
### POST CLUSTERING
### ===============

def cluster_wrapper(
    ds,
    clusters
):
    '''
    Wrapper to repeat function for each cluster. Requires the wrapped function to have `cid` (cluster id) and `cds` (cluster dataset) kwargs. 

    Args:
        ds (Dataset): Base dataset
        clusters (Array-like): List of cluster labels

    Returns:
        None
    '''
    
    def cw_outer(func):
        @wraps(func)
        def cw_inner(*args, **kwargs):
            results = []
            cluster_labels = list(set(clusters))
        
            for cluster_id, cluster_lbl in enumerate(cluster_labels):
        
                cluster_mask = clusters == cluster_lbl
                cluster_ds = ds.sel(PROFILE=cluster_mask, drop=True)
                
                cluster_result = func(cid = cluster_id, cds = cluster_ds, *args, **kwargs)
                results.append(cluster_result)

            return results
        return cw_inner
    return cw_outer