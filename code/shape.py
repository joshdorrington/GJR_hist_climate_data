import xarray as xr
import shapely
import numpy as np
import os.path
import iris
import cartopy.io.shapereader as shpreader
from cartopy.io.shapereader import Reader, natural_earth
from shapely.ops import unary_union

def get_geographic_region(resolution='110m', category='physical', name='land', attribute_filter=None):
    """
    :param str resolution: The resolution of the shape file to use, default is 110m
    :param str category: The Natural Earth category to search
    :param str name: The name of the Natural Earth feature set to use.
    :param dict attribute_filter: A dictionary of attribute filters to apply to the NE records.
    E.g. 'country': 'Albania' would return only those records with the 'country' attribute 'Albalia'
    :return: shapely.geometry.Polygon
    """
    attribute_filter = attribute_filter if attribute_filter is not None else {}
    shpfile = Reader(natural_earth(resolution=resolution, category=category, name=name))
    filtered_records = shpfile.records()

    for key, val in attribute_filter.items():
        filtered_records = filter(lambda x: x.attributes[key] == val, filtered_records)
    region_poly = unary_union([r.geometry for r in filtered_records])
    return region_poly

def roll_da(da):
    rolled = da.roll(longitude=da.dims['longitude'] // 2, roll_coords=False)
    rolled = rolled.assign_coords(longitude=da.longitude - 180.)
    return rolled

def get_indices_for_lat_lon_points(lats, lons, region):
    from shapely.geometry import MultiPoint

    lat_lon_points = np.vstack([lats, lons])
    points = MultiPoint(lat_lon_points.T)

    # Performance in this loop might be an issue, but I think it's essentially how GeoPandas does it. If I want to
    #  improve it I might need to look at using something like rtree.
    return [i for i, p in enumerate(points) if region.contains(p)]

def get_gridded_subset_region_indices(ds, region):
    """
    Get a set of indices representing the lat/lon points within a given region
    """    
    x, y = np.meshgrid(ds.longitude, ds.latitude)
    return get_indices_for_lat_lon_points(x.flat, y.flat, region)

def extract_region(ds, region):
    """
    Return a dataset representing the data which falls within a given region, flattened along a new dimension
    """
    
    # Ensure the longitudes match the shapefiles
    if any(ds.longitude>180):
        temp_ds = roll_da(ds)
    else:
        temp_ds = ds
    
    # Do a fast, coarse subset first
    xmin, ymin, xmax, ymax = region.bounds
    temp_ds = temp_ds.sel(longitude=slice(xmin, xmax), latitude=slice(ymax, ymin))

    matched_indices = get_gridded_subset_region_indices(temp_ds, region)
    
    x, y = np.unravel_index(matched_indices, (temp_ds.latitude.shape[0], temp_ds.longitude.shape[0]))
    return temp_ds.isel(latitude=xr.DataArray(x, dims="obs "), longitude=xr.DataArray(y, dims="obs "))


def get_data_in_country(cube,country_id,data_var):

    arr=xr.DataArray.from_iris(cube).to_dataset()
    region=get_geographic_region(category='cultural',name="admin_0_countries",attribute_filter={"NAME":country_id})
    subsetted=extract_region(arr, region)
    return np.array(subsetted[data_var].data)
