"""
 This file reads the DEM and shapefile of the study site, and creates
 the required static.nc.
"""

import os
import sys
from itertools import product

import numpy as np
import richdem as rd
import xarray as xr


def main():
    static_folder = "../../data/static/"

    tile = True
    aggregate = True

    # input digital elevation model (DEM)
    dem_path = static_folder + "DEM/dem_suldenferner_bolzano_50cm.tif"
    # input shape of glacier or study area, e.g. from the Randolph glacier inventory
    shape_path = static_folder + "Shapefiles/Suldenferner_Paul2016_NicholsonStiperski_2015.shp"
    # path where the static.nc file is saved
    output_path = static_folder + "Sulden_basin_static.nc"

    # to shrink the DEM use the following lat/lon corners
    glacier_coords = {
        "ortler": (10.40, 46.57, 10.80, 46.33),
        "suldenferner": (10.54, 46.51, 10.59, 46.48),
        "sulden_basin": (10.54, 46.51, 10.61, 46.48),
    }
    dem_coords = glacier_coords["sulden_basin"]

    longitude_upper_left = dem_coords[0]
    latitude_upper_left = dem_coords[1]
    longitude_lower_right = dem_coords[2]
    latitude_lower_right = dem_coords[3]

    # to aggregate the DEM to a coarser spatial resolution
    aggregate_degree = "0.0005"

    # intermediate files, will be removed afterwards
    dem_tile_temp = static_folder + "DEM_temp.tif"
    dem_agg_temp = static_folder + "DEM_temp2.tif"
    dem_path_temp = static_folder + "dem.nc"
    aspect_path = static_folder + "aspect.nc"
    mask_path = static_folder + "mask.nc"
    slope_path = static_folder + "slope.nc"

    # If you do not want to shrink the DEM, comment out the following to three lines
    if tile:
        command = (
            "gdal_translate -projwin",
            f"{longitude_upper_left} {latitude_upper_left}",
            f"{longitude_lower_right} {latitude_lower_right}",
            f"{dem_path} {dem_tile_temp}",
        )
        os.system(" ".join(command))
        dem_path = dem_tile_temp

    # If you do not want to aggregate DEM, comment out the following to two lines
    if aggregate:
        command = (
            f"gdalwarp -tr {aggregate_degree} {aggregate_degree}",
            f"-r bilinear {dem_path} {dem_agg_temp}",
        )
        os.system(" ".join(command))
        dem_path = dem_agg_temp

    aspect = get_dem_data(dem_path, dem_path_temp, slope_path)

    # calculate mask as NetCDF with DEM and shapefile
    command = (
        "gdalwarp -of NETCDF  --config GDALWARP_IGNORE_BAD_CUTLINE YES",
        f"-cutline {shape_path} {dem_path} {mask_path}",
    )
    os.system(" ".join(command))

    # open intermediate netcdf files
    dem = xr.open_dataset(dem_path_temp)
    mask = xr.open_dataset(mask_path)
    slope = xr.open_dataset(slope_path)

    mask = replace_nans(array=mask, ndv_mask=-9999, ndv_elevation=1)

    # clean up temp files
    os.system(f"rm {dem_path_temp} {mask_path} {slope_path}")
    if tile:
        os.system(f"rm {dem_tile_temp}")
    if aggregate:
        os.system(f"rm {dem_agg_temp}")

    ds = create_output_dataset(dem, aspect, slope, mask)
    ds.to_netcdf(output_path)

    debrief = (
        f"Study area consists of {np.nansum(mask[mask == 1])}",
        f"glacier points\nDone",
    )
    print(" ".join(debrief))


def get_dem_data(
    dem_path: str, dem_path_temp: str, slope_path: str
) -> np.ndarray:
    # convert DEM from tif to NetCDF
    os.system(f"gdal_translate -of NETCDF {dem_path} {dem_path_temp}")

    # calculate slope as NetCDF from DEM
    os.system(
        f"gdaldem slope -of NETCDF {dem_path_temp} {slope_path} -s 111120"
    )

    # calculate aspect from DEM
    aspect = np.flipud(
        rd.TerrainAttribute(
            rd.LoadGDAL(dem_path, no_data=-9999.0), attrib="aspect"
        )
    )

    return aspect


def replace_nans(
    array: np.ndarray, ndv_mask: float = -9999.0, ndv_elevation: float = 1
) -> np.ndarray:
    """Set NaNs in array to -9999 and elevation within the shape to 1."""
    mask = array.Band1.values
    mask[np.isnan(mask)] = ndv_mask
    mask[mask > 0] = ndv_elevation
    print(mask)
    return mask


def create_output_dataset(dem, aspect, slope, mask):
    """Create output dataset."""
    ds = xr.Dataset()
    ds.coords["lon"] = dem.lon.values
    ds.lon.attrs["standard_name"] = "lon"
    ds.lon.attrs["long_name"] = "longitude"
    ds.lon.attrs["units"] = "degrees_east"

    ds.coords["lat"] = dem.lat.values
    ds.lat.attrs["standard_name"] = "lat"
    ds.lat.attrs["long_name"] = "latitude"
    ds.lat.attrs["units"] = "degrees_north"

    # insert needed static variables
    insert_var(ds, dem.Band1.values, "HGT", "meters", "meter above sea level")
    insert_var(ds, aspect, "ASPECT", "degrees", "Aspect of slope")
    insert_var(ds, slope.Band1.values, "SLOPE", "degrees", "Terrain slope")
    insert_var(ds, mask, "MASK", "boolean", "Glacier mask")

    check_for_nan(ds)

    return ds


# function to insert variables to dataset
def insert_var(ds, var, name, units, long_name):
    ds[name] = (("lat", "lon"), var)
    ds[name].attrs["units"] = units
    ds[name].attrs["long_name"] = long_name
    ds[name].attrs["_FillValue"] = -9999


# save combined static file, delete intermediate files and print number of glacier grid points
def check_for_nan(ds, var=None):
    for y, x in product(range(ds.dims["lat"]), range(ds.dims["lon"])):
        mask = ds.MASK.isel(lat=y, lon=x)
        if mask == 1:
            error_msg = "ERROR!!!!!!!!!!! There are NaNs in the static fields."
            if var is None:
                if np.isnan(ds.isel(lat=y, lon=x).to_array()).any():
                    print(error_msg)
                    sys.exit()
            else:
                if np.isnan(ds[var].isel(lat=y, lon=x)).any():
                    print(error_msg)
                    sys.exit()


if __name__ == "__main__":
    main()
