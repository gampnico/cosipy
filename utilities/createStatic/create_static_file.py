"""
 This file reads the DEM of the study site and the shapefile and creates the needed static.nc

"""
import sys
import os
import xarray as xr
import numpy as np
from itertools import product
import richdem as rd
import fiona
from horayzon.domain import curved_grid

static_folder = '../../data/static/'

sys.path.append("../..")
from utilities.aws2cosipy.crop_file_to_glacier import crop_file_to_glacier

## Define settings ##
distributed_radiation = True
tile = True

## If distributed radation is not desired, run default static file creation according to values below ##
# to aggregate the DEM to a coarser spatial resolution
aggregate = True
aggregate_degree = '0.00277778'
automatic_domain = True #Do km buffer around glacier or set lat lon box by hand in functions below
crop_file = True #crop to minimum extent
# ref exist: If already have high res. static data set to True and skip calculation
ref_exist = False

### input digital elevation model (DEM)
dem_path_tif = static_folder + 'DEM/n30_e090_3arc_v2.tif'  #n30

### input shape of glacier or study area, e.g. from the Randolph glacier inventory
shape_path = static_folder + 'Shapefiles/Zhadang_RGI6.shp'

### path were the static.nc file is saved
output_path = static_folder + 'Zhadang_static_raw.nc'
output_path_agg = static_folder + 'Zhadang_static_agg.nc'

#domain creation assumes WGS84 is valid
def domain_creation(shp_path, dist_search, ellps="WGS84"):
    print("Using automatic domain creation.")
    #Get bound of glacier shapefile
    shp = fiona.open(shp_path)
    domain = {"lon_min": shp.bounds[0], "lon_max": shp.bounds[2],
              "lat_min": shp.bounds[1], "lat_max": shp.bounds[3]}

    #Additionally there is the option of a planar grid which requires importing the function
    domain_outer = curved_grid(domain, dist_search=dist_search, ellps=ellps)
    ### Get lat/lon corners ###
    #Note setup is created based on Northern Hemisphere glaciers
    longitude_upper_left = str(domain_outer['lon_min'])
    latitude_upper_left = str(domain_outer['lat_max'])
    longitude_lower_right = str(domain_outer['lon_max'])
    latitude_lower_right = str(domain_outer['lat_min'])

    return (longitude_upper_left, latitude_upper_left, longitude_lower_right, latitude_lower_right)

def create_static(dem_path_tif=dem_path_tif, shape_path=shape_path, output_path=output_path, tile=tile, aggregate=aggregate, aggregate_degree=aggregate_degree, automatic_domain=automatic_domain, crop_file=crop_file, dist_search=20.0):
    if automatic_domain:
        longitude_upper_left, latitude_upper_left, longitude_lower_right, latitude_lower_right = domain_creation(shape_path, dist_search=dist_search, ellps="WGS84")

    else:
        ### to shrink the DEM use the following lat/lon corners
        print("Please be aware that you switched off the automatic domain creation which requires by-hand adjustment of the borders.")
        longitude_upper_left = '90.62'
        latitude_upper_left = '30.48'
        longitude_lower_right = '90.66'
        latitude_lower_right = '30.46'

    ### to aggregate the DEM to a coarser spatial resolution
    aggregate_degree = aggregate_degree

    ### intermediate files, will be removed afterwards
    dem_path_tif_temp = static_folder + 'DEM_temp.tif'
    dem_path_tif_temp2 = static_folder + 'DEM_temp2.tif'
    dem_path = static_folder + 'dem.nc'
    aspect_path = static_folder + 'aspect.nc'
    mask_path = static_folder + 'mask.nc'
    slope_path = static_folder + 'slope.nc'

    ### If you do not want to shrink the DEM, comment out the following to three lines
    if tile:
        os.system('gdal_translate -projwin ' + longitude_upper_left + ' ' + latitude_upper_left + ' ' +
              longitude_lower_right + ' ' + latitude_lower_right + ' ' + dem_path_tif + ' ' + dem_path_tif_temp)
        dem_path_tif = dem_path_tif_temp

    ### If you do not want to aggregate DEM, comment out the following to two lines
    if aggregate:
        print("Aggregating to ", aggregate_degree)
        os.system('gdalwarp -tr ' + aggregate_degree + ' ' + aggregate_degree + ' -r average ' + dem_path_tif + ' ' + dem_path_tif_temp2)
        dem_path_tif = dem_path_tif_temp2

    ### convert DEM from tif to NetCDF
    os.system('gdal_translate -of NETCDF ' + dem_path_tif  + ' ' + dem_path)

    ### calculate slope as NetCDF from DEM
    os.system('gdaldem slope -of NETCDF ' + dem_path + ' ' + slope_path + ' -s 111120')

    ### calculate aspect from DEM
    #no_data value may differ from file to file, depending on your DEM you may need to change this
    aspect = np.flipud(rd.TerrainAttribute(rd.LoadGDAL(dem_path_tif, no_data=-9999), attrib = 'aspect'))

    ### calculate mask as NetCDF with DEM and shapefile
    os.system('gdalwarp -of NETCDF  --config GDALWARP_IGNORE_BAD_CUTLINE YES -cutline ' + shape_path + ' ' + dem_path_tif  + ' ' + mask_path)

    ### open intermediate netcdf files
    dem = xr.open_dataset(dem_path)
    mask = xr.open_dataset(mask_path)
    slope = xr.open_dataset(slope_path)

    ### set NaNs in mask to -9999 and elevation within the shape to 1
    mask=mask.Band1.values
    #some datasets have 0s instead of -9999
    mask[mask == 0] = -9999
    mask[np.isnan(mask)]=-9999
    mask[mask>0]=1
    print(mask)

    ## create output dataset
    ds = xr.Dataset()
    ds.coords['lon'] = dem.lon.values
    ds.lon.attrs['standard_name'] = 'lon'
    ds.lon.attrs['long_name'] = 'longitude'
    ds.lon.attrs['units'] = 'degrees_east'

    ds.coords['lat'] = dem.lat.values
    ds.lat.attrs['standard_name'] = 'lat'
    ds.lat.attrs['long_name'] = 'latitude'
    ds.lat.attrs['units'] = 'degrees_north'

    ### function to insert variables to dataset
    def insert_var(ds, var, name, units, long_name):
        ds[name] = (('lat','lon'), var)
        ds[name].attrs['units'] = units
        ds[name].attrs['long_name'] = long_name
        ds[name].attrs['_FillValue'] = -9999

    ### insert needed static variables
    insert_var(ds, dem.Band1.values,'HGT','meters','meter above sea level')
    insert_var(ds, aspect,'ASPECT','degrees','Aspect of slope')
    insert_var(ds, slope.Band1.values,'SLOPE','degrees','Terrain slope')
    insert_var(ds, mask,'MASK','boolean','Glacier mask')

    os.system('rm '+ dem_path + ' ' + mask_path + ' ' + slope_path + ' ' + dem_path_tif_temp + ' '+ dem_path_tif_temp2)

    ### save combined static file, delete intermediate files and print number of glacier grid points
    def check_for_nan(ds,var=None):
        for y,x in product(range(ds.dims['lat']),range(ds.dims['lon'])):
            mask = ds.MASK.isel(lat=y, lon=x)
            if mask==1:
                if var is None:
                    if np.isnan(ds.isel(lat=y, lon=x).to_array()).any():
                        print('ERROR!!!!!!!!!!! There are NaNs in the static fields')
                        sys.exit()
                else:
                    if np.isnan(ds[var].isel(lat=y, lon=x)).any():
                        print('ERROR!!!!!!!!!!! There are NaNs in the static fields')
                        sys.exit()
    check_for_nan(ds)
    print(output_path)
    if crop_file == True:
        ds_crop = crop_file_to_glacier(ds)
        ds_crop.to_netcdf(output_path)
        ds.close()
        ds_crop.close()
    else:
        ds.to_netcdf(output_path)
        ds.close()
    print("Study area consists of ", np.nansum(mask[mask==1]), " glacier points")
    print("Done")

#For some reason, runs are switched.

if distributed_radiation:
    #We need to first get a domain at high resolution and then if aggregate is True: create a second domain with lower resolution to use in later stage
    if ref_exist:
        print("Skipping calculation of high resolution static file.")
    else:
        #This needs to have a buffer
        create_static(dem_path_tif=dem_path_tif, shape_path=shape_path, output_path=output_path,
                      tile=tile, aggregate=False, aggregate_degree=aggregate_degree,
                       automatic_domain=True, crop_file=False, dist_search=20.0)
    print("\n----------------------------------------")
    print("Created high resolution domain for LUTs.")
    print("----------------------------------------\n")
    #This doesn't really need a buffer so crop it to minimum extent possible next
    create_static(dem_path_tif=dem_path_tif, shape_path=shape_path, output_path=output_path_agg,
                  tile=tile, aggregate=True, aggregate_degree=aggregate_degree,
                  automatic_domain=True, crop_file=True, dist_search=1.0)
    print("\n----------------------------------------")
    print("Stored aggregated domain for resampling.")
    print("----------------------------------------\n")
else:
    #Generally less grid cells the better, so crop to minimum extent possible
    create_static(dem_path_tif=dem_path_tif, shape_path=shape_path, output_path=output_path,
                  tile=tile, aggregate=aggregate, aggregate_degree=aggregate_degree,
                  automatic_domain=automatic_domain, crop_file=True, dist_search=20.0)
