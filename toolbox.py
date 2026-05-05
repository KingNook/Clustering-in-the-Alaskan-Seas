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

import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from functools import wraps

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
    keep_vars = ['BOTTOM_DEPTH', 'Region_Name'],
    temperature_var = 'Temperature',
    salinity_var = 'Merged_Salinity'
):
    '''
    Interpolates Temperature and Salinity values along the PRESSURE axis. By default, will get rid of all other variables.
    Extra variables (eg BOTTOM_DEPTH) can be kept via `keep_vars`

    Args:
        ds (Dataset): Base dataset
        additional_vars (List): List of additional variables to be kept. These will not be interpolated
        temperature_var (str): Name of temperature variable
        salinity_var (str): Name of salinity variable

    Returns:
        new_ds (Dataset): Interpolated dataset
    '''

    new_ds = ds[[temperature_var, salinity_var]]

    new_ds = new_ds.interpolate_na(dim='PRESSURE')

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

def normalise_array(arr):
    '''
    Normalises an array by mean-centering then dividing by standard deviation

    Args:
        arr (array-like): Array in the order (n_features, n_samples)

    Returns:
        normalised_arr (array-like)
    '''

    mean = arr.mean(axis=0)
    std = arr.std(axis=0)

    return np.divide(arr - mean, std)

def clean_dataset(
    ds,
    temperature_var = 'Temperature',
    salinity_var = 'Merged_Salinity',
    verbose = False,
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

    nan_mask = np.isnan(ds[temperature_var]) | np.isnan(ds[salinity_var])
    clean_ds = ds.where(~nan_mask, drop=True)

    if verbose:
        print(f'Dropped {sum(nan_mask.data)} values')

    return clean_ds
    
def prepare_training_data(
    ds,
    temperature_var = 'Temperature',
    salinity_var = 'Merged_Salinity',
    normalise = True
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

    temp = ds[temperature_var].data.reshape(-1, 1)
    sal = ds[salinity_var].data.reshape(-1, 1)

    training_data = np.concatenate((temp, sal), axis=1)

    if normalise:
        training_data = normalise_array(training_data)

    return training_data

def select_region(
    ds,
    longitude_range=None, lon_var = 'LONGITUDE',
    latitude_range=None, lat_var = 'LATITUDE',
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

### ==============
### PLOTTING FUNCS
### ==============

def geoaxes(n_rows=1, n_cols=1, **mpl_kw):
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

    fig, axs = plt.subplots(
        n_rows, n_cols, **mpl_kw, 
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}
    )

    if n_rows + n_cols > 2:
        axs = axs.ravel()

    if n_rows == 1 and n_cols == 1:
        axs.add_feature(cfeature.COASTLINE, alpha=0.2)
        axs.add_feature(cfeature.BORDERS, alpha=0.2)
        axs.add_feature(cfeature.LAND, alpha=0.2)
    
    return fig, axs

# techincally GeoAxes is cartopy.mpl.geoaxes.GeoAxes
def plot_all_vals(
    ds, ax=None, color=None, 
    LON_VAR='LONGITUDE', LAT_VAR='LATITUDE', LON_SHIFT = 180, 
    alpha = 0.6,
    **plot_kw
):
    '''
    Plots all points (latitude-longitude) from a dataset; gives a general gist of where the data lies. 

    Args:
        ds (xr.Dataset): Dataset of values
        ax (GeoAxes; optional): Axis to plot data on
        color (string; optional): Variable to color by

    Returns:
        im (matplotlib scatter plot)
    '''

    if ax == None:
        fig, ax = geoaxes()

    lons = ds[LON_VAR] + LON_SHIFT
    lats = ds[LAT_VAR]

    im = ax.scatter(
        lons, lats, c=color, s=1, alpha=alpha, **plot_kw
    )

    if type(color) == 'str':
        # probably add cluster to legend
        pass

    elif type(color) == xr.core.dataarray.DataArray or type(color) == np.ndarray:
        fig.colorbar(im)

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