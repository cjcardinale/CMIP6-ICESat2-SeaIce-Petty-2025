import numpy as np
import regionmask
import xarray as xr
import geopandas as gp
import warnings

data_path = 'files/'
NH_seaice_regions = gp.read_file(data_path+'NSIDC-0780_SeaIceRegions_NH_v1.0.shp')
SH_seaice_regions =gp.read_file(data_path+'NSIDC-0780_SeaIceRegions_SH-NASA_v1.0.shp')
ATL20_area_NH = xr.open_dataset(data_path+'NSIDC0771_CellArea_PS_N25km_v1.0.nc').cell_area
ATL20_area_SH = xr.open_dataset(data_path+'NSIDC0771_CellArea_PS_S25km_v1.0.nc').cell_area

def SAE(model,obs,weight):
    O = (model - obs).where((model - obs)>0).fillna(0)
    U = (obs - model).where((obs - model)>0).fillna(0)
    total = O+U
    total['x'] = np.arange(0,total.x.size,1)
    total['y'] = np.arange(0,total.y.size,1)
    if isinstance(weight,xr.DataArray):
        weight['x'] = total.x
        weight['y'] = total.y
    #integ1 = (total*weight).integrate('x')
    #integ2 = (integ1).integrate('y')
    sum = (total*weight).sum(['x','y'])
    return sum*1e-6

def MAE(model, obs, dim=None, skipna=True, weights=None):
    res = np.abs(model - obs)
    if weights is not None:
        res = res.weighted(weights)
    res = res.mean(dim=dim, skipna=skipna)
    return res

def to_monthly(ds):
    year = ds.time.dt.year
    month = ds.time.dt.month
    # assign new coords
    ds = ds.assign_coords(year=("time", year.data), month=("time", month.data))
    # reshape the array to (..., "month", "year")
    return ds.set_index(time=("year", "month")).unstack("time") 

def sel_model(ds,sid='CESM2'):
    subset = ds.sel(member_id=ds.member_id.str.split('split','_').sel(split=0)==sid)
    return subset

def spatial_average(ds,weight=None,keep_zeros=True,sector_mean='NH'):
    ds = ds.copy()
    if 'lat' in ds.coords and 'y' in ds.dims:
        if sector_mean in ['Inner Arctic','IA']:
            sector_mean = 'Inner_Arctic'
        if sector_mean == 'NH':
            ds_subset = ds.where(ds.lat>0)
            lat = ds.lat.where(ds.lat>0)
        elif sector_mean == 'SH':
            ds_subset = ds.where(ds.lat<0)
            lat = ds.lat.where(ds.lat<0)
        elif sector_mean == 'Arctic':
            ds_subset = ds.where(ds.lat>60)
            lat = ds.lat.where(ds.lat>60)
        elif sector_mean == 'Antarctic':
            ds_subset = ds.where(ds.lat<-60)
            lat = ds.lat.where(ds.lat<-60)
        elif sector_mean == 'Inner_Arctic':
            df = NH_seaice_regions
            mask = regionmask.mask_geopandas(df, ds.lon, ds.lat,overlap=False)
            ds_subset = ds.where(mask.isin([0,1,2,3,4,5,6]))
            lat = ds.lat.where(mask.isin([0,1,2,3,4,5,6]))
        elif isinstance(sector_mean, dict): 
            if list(sector_mean.keys())[0] == 'Arctic':
                df = NH_seaice_regions
                mask = regionmask.mask_geopandas(df, ds.lon, ds.lat,overlap=False)
                ds_subset = ds.where(mask.isin(list(sector_mean.values())[0]))
                lat = ds.lat.where(mask.isin(list(sector_mean.values())[0]))
        elif isinstance(sector_mean, dict): 
            if list(sector_mean.keys())[0] == 'Antarctic':
                df = SH_seaice_regions
                mask = regionmask.mask_geopandas(df, ds.lon, ds.lat,overlap=False)
                ds_subset = ds.where(mask.isin(list(sector_mean.values())[0]))
                lat = ds.lat.where(mask.isin(list(sector_mean.values())[0]))
        else: 
            ds_subset = ds
        if not isinstance(weight,xr.DataArray):
            #lat is already a 2D field, so each point is weighted without having to broadcast!
            #if data is on a recticular grid, then we can just weight is by the cosine of lat
            lat = lat.where(ds_subset.notnull())
            weight=np.cos(np.deg2rad(lat))/np.cos(np.deg2rad(lat)).mean('y')
            ds_mean = (ds_subset*weight).mean(['x','y'])
        else:
            #if data is on a curvilinear grid, we need to weight it by the grid-cell area
            #here, we just compute the average over the area where there is data or where 
            #sea ice variables are not 0
            #ds_subset,weight = _fix_grids(ds_subset,weight)
            if keep_zeros==False: 
                ds_mean = ((ds_subset*weight).sum(['x', 'y'],min_count=1)
                            /weight.where(np.logical_and(ds_subset.notnull(),ds_subset>0)).sum(['x','y']))
            else:
                ds_mean = ((ds_subset*weight).sum(['x', 'y'],min_count=1)
                            /weight.where(ds_subset.notnull()).sum(['x','y']))
    return ds_mean

def ensemble_mean(ds,thresh=1):
    members_count = ds.member_id.groupby(ds.member_id.str.split('split','_').sel(split=0)).count()
    models = members_count.where(members_count>=thresh).dropna('member_id').member_id
    ens_mean = ds.groupby(ds.member_id.str.split('split','_').sel(split=0)).mean()
    ens_mean = ens_mean.sel(member_id=models)
    ens_mean['member_id'] = ens_mean.member_id.astype(dtype='<U25')
    return ens_mean

def int_variability(ds,thresh=2,dims='member_id'):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #ds = ds.dropna('member_id','all')
        members_count = ds.member_id.groupby(ds.member_id.str.split('split','_').sel(split=0)).count()
        models = members_count.where(members_count>=thresh).dropna('member_id').member_id
        int_var = ds.groupby(ds.member_id.str.split('split','_').sel(split=0)).std(dims,ddof=1)**2
        int_var = int_var.sel(member_id=models)
        int_var['member_id'] = int_var.member_id.astype(dtype='<U25')
    return int_var

def find_nan_surrounded_by_nan(data, threshold=0.8):
    """
    Find NaN grid cells in a 3D xarray.DataArray (lat, lon, time) that are surrounded
    by at least `threshold` fraction of NaN values in their neighboring cells for each time step.
    
    Parameters:
        data (xr.DataArray): Input 3D DataArray with dimensions (time, lat, lon).
        threshold (float): Fraction of neighbors that must be NaN for a cell to be marked as True.
        
    Returns:
        xr.DataArray: A boolean DataArray with the same dimensions as `data`,
                      where True indicates NaN grid cells surrounded by NaN values.
    """
    def process_single_time_slice(slice_2d):
        """Process a single 2D slice of the DataArray."""
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