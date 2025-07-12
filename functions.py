import numpy as np
import regionmask
import xarray as xr
import geopandas as gp
import warnings
import zarr

data_path = 'files/'
NH_seaice_regions = gp.read_file(data_path+'NSIDC-0780_SeaIceRegions_NH_v1.0.shp')
SH_seaice_regions =gp.read_file(data_path+'NSIDC-0780_SeaIceRegions_SH-NASA_v1.0.shp')
ATL20_area_NH = xr.open_dataset(data_path+'NSIDC0771_CellArea_PS_N25km_v1.0.nc').cell_area
ATL20_area_SH = xr.open_dataset(data_path+'NSIDC0771_CellArea_PS_S25km_v1.0.nc').cell_area

def is_valid_zarr_store(s3map):
    """
    Check whether a Zarr store is valid and readable as a consolidated dataset.
    
    Parameters:
        s3map: Zarr store mapping object (e.g., S3Map or FSMap)
        
    Returns:
        bool: True if the store is valid and readable, False otherwise
    """
    try:
        # Try to open with consolidated metadata
        zarr.convenience.open_consolidated(s3map)
        return True
    except Exception as e:
        print(f"Skipping invalid store: {s3map.root} -- {e}")
        return False

def SAE(model,obs,weight):
    """
    Compute Sum Absolute Error (SAE) between model and observations, weighted by cell area.
    
    Parameters:
        model (xr.DataArray): Model data array
        obs (xr.DataArray): Observational data array  
        weight (xr.DataArray): Weight array for spatial weighting (typically cell area)
        
    Returns:
        xr.DataArray: Sum absolute error scaled by 1e-6
    """
    # Calculate overestimate (model > obs) and underestimate (obs > model) components
    O = (model - obs).where((model - obs)>0).fillna(0)
    U = (obs - model).where((obs - model)>0).fillna(0)
    total = O+U
    
    # Reset coordinate indices to ensure proper alignment
    total['x'] = np.arange(0,total.x.size,1)
    total['y'] = np.arange(0,total.y.size,1)
    if isinstance(weight,xr.DataArray):
        weight['x'] = total.x
        weight['y'] = total.y

    # Compute weighted sum and scale to appropriate units
    sum = (total*weight).sum(['x','y'])
    return sum*1e-6

def MAE(model, obs, dim=None, skipna=True, weights=None):
    """
    Compute Mean Absolute Error (MAE) between model and observations.
    
    Parameters:
        model (xr.DataArray): Model data array
        obs (xr.DataArray): Observational data array
        dim (str or list, optional): Dimension(s) over which to compute the mean.
        skipna (bool, optional): Whether to skip NaN values. Default is True.
        weights (xr.DataArray, optional): Weights for weighted average.
        
    Returns:
        xr.DataArray: Mean absolute error
    """
    res = np.abs(model - obs)
    if weights is not None:
        res = res.weighted(weights)
    res = res.mean(dim=dim, skipna=skipna)
    return res

def to_monthly(ds):
    """
    Convert the time dimension to month and year dimensions.
    
    Parameters:
        ds (xr.Dataset or xr.DataArray): Input dataset with time dimension
        
    Returns:
        xr.Dataset or xr.DataArray: Reshaped data with separate year and month dimensions
    """
    year = ds.time.dt.year
    month = ds.time.dt.month
    # assign new coords
    ds = ds.assign_coords(year=("time", year.data), month=("time", month.data))
    # reshape the array to (..., "month", "year")
    return ds.set_index(time=("year", "month")).unstack("time") 

def sel_model(ds,sid='CESM2'):
    """
    Select a specific model from the dataset using a source_id prefix.
    
    Parameters:
        ds (xr.Dataset or xr.DataArray): Input dataset with member_id dimension
        sid (str, optional): Source ID prefix to select. E.g., 'CESM2'.
        
    Returns:
        xr.Dataset or xr.DataArray: Subset containing all members of the specified model
    """
    # Extract model name from member_id by splitting on '_' and taking first part
    subset = ds.sel(member_id=ds.member_id.str.split('split','_').sel(split=0)==sid)
    return subset

def ensemble_count(ds,rename=True):
    """
    Count ensemble members for each model, optionally renaming to 'source_id'.
    
    Parameters:
        ds (xr.Dataset or xr.DataArray): Input dataset with member_id dimension
        rename (bool, optional): Whether to rename 'member_id' to 'source_id'. Default is True.
        
    Returns:
        xr.Dataset: Dataset containing member counts for each model
    """
    members_count = ds.member_id.groupby(ds.member_id.str.split('split','_').sel(split=0)).count()
    members_count['member_id'] = members_count.member_id.astype(dtype='<U25')
    if rename==True:
        members_count=members_count.rename({'member_id':'source_id'})
    return members_count.to_dataset(name='members')

def spatial_average(ds, weight=None, keep_zeros=True, sector_mean='NH', mask=None):
    """
    Compute spatial average for a dataset within defined polar sectors using area or latitude weights.
    Optionally, pass a precomputed mask for efficiency.
    
    Parameters:
        ds (xr.Dataset or xr.DataArray): Input dataset with lat/lon coordinates
        weight (xr.DataArray, optional): Area weights for spatial averaging. If None, uses cosine latitude weighting.
        keep_zeros (bool, optional): Whether to include zero values in averaging. Default is True.
        sector_mean (str or dict, optional): Sector for averaging. Options: 'NH', 'SH', 'Arctic', 'Antarctic', 
                                           'Inner_Arctic', or dict with region specifications. Default is 'NH'.
        mask (xr.DataArray, optional): Precomputed boolean mask for efficiency. Default is None.
        
    Returns:
        xr.Dataset or xr.DataArray: Spatially averaged data
    """
    # Sector selection - create boolean mask based on latitude or region
    if 'lat' in ds.coords and 'y' in ds.dims:
        # Handle alternative naming for Inner Arctic
        if sector_mean in ['Inner Arctic', 'IA']:
            sector_mean = 'Inner_Arctic'
        # Define sector masks
        if sector_mean == 'NH':
            mask = ds.lat > 0
        elif sector_mean == 'SH':
            mask = ds.lat < 0
        elif sector_mean == 'Arctic':
            mask = ds.lat > 60
        elif sector_mean == 'Antarctic':
            mask = ds.lat < -60
        elif sector_mean == 'Inner_Arctic':
            # Use predefined sea ice regions
            mask = regionmask.mask_geopandas(NH_seaice_regions, ds.lon, ds.lat, overlap=False).isin([0,1,2,3,4,5,6])
        elif isinstance(sector_mean, dict):
            # Custom region dictionary: {'Arctic': [region_ids]} or {'Antarctic': [region_ids]}
            if list(sector_mean.keys())[0] == 'Arctic':
                mask = regionmask.mask_geopandas(NH_seaice_regions, ds.lon, ds.lat, overlap=False).isin(list(sector_mean.values())[0])
            elif list(sector_mean.keys())[0] == 'Antarctic':
                mask = regionmask.mask_geopandas(SH_seaice_regions, ds.lon, ds.lat, overlap=False).isin(list(sector_mean.values())[0])
        else:
            # Default: include all points
            mask = xr.full_like(ds.lat, True, dtype=bool)
        ds_subset = ds.where(mask)
        lat = ds.lat.where(mask)
    else:
        # No spatial subsetting if lat/lon not available
        ds_subset = ds
        lat = ds.lat if 'lat' in ds.coords else None

    # Weighting
    if weight is None and lat is not None:
        # Use cosine latitude weighting when no explicit weights provided
        lat = lat.where(ds_subset.notnull())
        weight = np.cos(np.deg2rad(lat)) / np.cos(np.deg2rad(lat)).mean('y')
        ds_mean = (ds_subset * weight).mean(['x', 'y'])
    elif weight is not None:
        # Use provided weights (typically cell area)
        if keep_zeros is False:
            # Exclude zero values from weighting
            ds_mean = ((ds_subset * weight).sum(['x', 'y'], min_count=1) /
                       weight.where(np.logical_and(ds_subset.notnull(), ds_subset > 0)).sum(['x', 'y']))
        else:
            # Include zero values in weighting
            ds_mean = ((ds_subset * weight).sum(['x', 'y'], min_count=1) /
                       weight.where(ds_subset.notnull()).sum(['x', 'y']))
    else:
        # Simple unweighted average
        ds_mean = ds_subset.mean(['x', 'y'])

    return ds_mean

def ensemble_mean(ds,thresh=1):
    """
    Compute the ensemble mean for models with a minimum number of members.
    
    Parameters:
        ds (xr.Dataset or xr.DataArray): Input dataset with member_id dimension
        thresh (int, optional): Minimum number of ensemble members required. Default is 1.
        
    Returns:
        xr.Dataset or xr.DataArray: Ensemble mean for qualifying models
    """
    members_count = ds.member_id.groupby(ds.member_id.str.split('split','_').sel(split=0)).count()
    models = members_count.where(members_count>=thresh).dropna('member_id').member_id
    ens_mean = ds.groupby(ds.member_id.str.split('split','_').sel(split=0)).mean()
    ens_mean = ens_mean.sel(member_id=models)
    ens_mean['member_id'] = ens_mean.member_id.astype(dtype='<U25')
    return ens_mean

def int_variability(ds,thresh=2,dims='member_id'):
    """
    Compute internal variability (variance) for models with at least `thresh` ensemble members.
    
    Parameters:
        ds (xr.Dataset or xr.DataArray): Input dataset with member_id dimension
        thresh (int, optional): Minimum number of ensemble members required. Default is 2.
        dims (str, optional): Dimension over which to compute variance. Default is 'member_id'.
        
    Returns:
        xr.Dataset or xr.DataArray: Internal variability (variance) for qualifying models
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        members_count = ds.member_id.groupby(ds.member_id.str.split('split','_').sel(split=0)).count()
        models = members_count.where(members_count>=thresh).dropna('member_id').member_id
        int_var = ds.groupby(ds.member_id.str.split('split','_').sel(split=0)).std(dims,ddof=1)**2
        int_var = int_var.sel(member_id=models)
        int_var['member_id'] = int_var.member_id.astype(dtype='<U25')
    return int_var

def find_nan_surrounded_by_nan(data, threshold=0.8):
    """
    Identify NaN grid cells in a 3D DataArray that are surrounded by a high fraction of NaN values in their 3x3 neighborhood.
    Example usage: remove zeros from nan streak in the UKESM1-0-LL [after applying ds.fillna(0)] to avoid large values when computing internal variability

    Parameters:
        data (xr.DataArray): Input with dimensions (time, y, x).
        threshold (float): Fraction of NaN neighbors needed to flag a point.

    Returns:
        xr.DataArray: Boolean mask where True indicates surrounded NaNs.
    """
    def process_single_time_slice(slice_2d):
        # Create a padded array to check neighbors
        padded_data = np.pad(slice_2d, pad_width=1, mode='constant', constant_values=np.nan)

        # Initialize a mask for NaN grid cells surrounded by NaN values
        nan_surrounded_by_nan = np.full(slice_2d.shape, False)

        # Loop through the array (excluding padding) and check neighbors
        for i in range(1, padded_data.shape[0] - 1):
            for j in range(1, padded_data.shape[1] - 1):
                # Extract 3x3 neighborhood (including the center point)
                neighbors = padded_data[i-1:i+2, j-1:j+2]
                
                # Only check if the center point is NaN
                if np.isnan(neighbors[1, 1]):
                    # Count NaN values in the neighborhood (excluding the center point)
                    nan_count = np.sum(~np.isnan(neighbors)) - int(~np.isnan(neighbors[1, 1]))
                    
                    # Total possible neighbors (8 for interior points, fewer at edges/corners)
                    total_neighbors = np.prod(neighbors.shape) - 1  # Exclude center point
                    
                    # Check if NaN percentage meets the threshold
                    if nan_count / total_neighbors >= threshold:
                        nan_surrounded_by_nan[i-1, j-1] = True

        return nan_surrounded_by_nan

    # Apply the function across the time dimension
    result = xr.apply_ufunc(
        process_single_time_slice,
        data,
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y", "x"]],
        vectorize=True,  # Apply function independently for each time slice
        dask="parallelized",  # Enable Dask for parallel computation if needed
        output_dtypes=[bool]
    )

    # Return the result
    return result.rename("nan_surrounded_by_nan")

