import sys
import xarray as xr
import numpy as np
from itertools import product

#np.warnings.filterwarnings('ignore')

sys.path.append('../../')
from utilities.aws2cosipy.aws2cosipyConfig import *
import argparse

def crop_file_to_glacier(ds):

    dic_attrs= {'HGT': ('HGT', 'm', 'Elevation'),
                'ASPECT': ('ASPECT', 'degrees', 'Aspect of slope'),
                'SLOPE': ('SLOPE', 'degrees', 'Terrain slope'),
                'MASK': ('MASK', 'boolean', 'Glacier mask'),
                'T2': ('T2', 'K', 'Temperature at 2 m'),
                'RH2': ('RH2', '%', 'Relative humidity at 2 m'),
                'U2': ('U2', 'm s\u207b\xb9', 'Wind velocity at 2 m'),
                'G': ('G', 'W m\u207b\xb2', 'Incoming shortwave radiation'),
                'PRES': ('PRES', 'hPa', 'Atmospheric Pressure'),
                'N_Points': ('N_Points', 'count','Number of Points in each bin'),
                'RRR': ('RRR', 'mm', 'Total precipitation (liquid+solid)'),
                'SNOWFALL': ('SNOWFALL', 'm', 'Snowfall'),
                'LWin': ('LWin', 'W m\u207b\xb2', 'Incoming longwave radiation'),
                'N': ('N', '%', 'Cloud cover fraction'),
                'sw_dir_cor': ('sw_dir_cor', '-', 'correction factor for direct downward shortwave radiation'),
                'slope': ('slope','degrees', 'Horayzon Slope'),
                'aspect': ('aspect','degrees','Horayzon Aspect measured clockwise from the North'),
                'surf_enl_fac': ('surf_enl_fac','-','Surface enlargement factor'),
                'elevation': ('elevation','m','Orthometric Height')}


    dso = ds 

    print('Create cropped file.')
    dso_mod = xr.Dataset()
    for var in list(dso.variables):
        print(var)
        arr = bbox_2d_array(dso.MASK.values, dso[var].values, var)
        if var in ['lat','latitude','lon','longitude','time','Time']:
            if var == 'lat' or var == 'latitude':
                dso_mod.coords['lat'] = arr
                dso_mod.lat.attrs['standard_name'] = 'lat'
                dso_mod.lat.attrs['long_name'] = 'latitude'
                dso_mod.lat.attrs['units'] = 'degrees_north'
            elif var == 'lon' or var == 'longitude':
                dso_mod.coords['lon'] = arr
                dso_mod.lon.attrs['standard_name'] = 'lon'
                dso_mod.lon.attrs['long_name'] = 'longitude'
                dso_mod.lon.attrs['units'] = 'degrees_east'
            else:
                dso_mod.coords['time'] = arr
        elif var in ['HGT','ASPECT','SLOPE','MASK','N_Points','surf_enl_fac','slope','aspect','elevation']:
            add_variable_along_latlon(dso_mod, arr, dic_attrs[var][0], dic_attrs[var][1], dic_attrs[var][2])
        else:
            add_variable_along_timelatlon(dso_mod, arr, dic_attrs[var][0], dic_attrs[var][1], dic_attrs[var][2])
    
    #----------------------
    # Do some checks
    #----------------------
    print("Performing checks.")
    check_for_nan(dso_mod)

    if (T2_var in list(dso_mod.variables)):
        check(dso_mod.T2,316.16,223.16)
    if (RH2_var in list(dso_mod.variables)):
        check(dso_mod.RH2,100.0,0.0)
    if (U2_var in list(dso_mod.variables)):
        check(dso_mod.U2, 50.0, 0.0)
    if (G_var in list(dso_mod.variables)):
        check(dso_mod.G,1600.0,0.0)
    if (PRES_var in list(dso_mod.variables)):
        check(dso_mod.PRES,1080.0,200.0)

    if (RRR_var in list(dso_mod.variables)):
        check(dso_mod.RRR,25.0,0.0)

    if (SNOWFALL_var in list(dso_mod.variables)):
        check(dso_mod.SNOWFALL, 0.05, 0.0)

    if (LWin_var in list(dso_mod.variables)):
        check(dso_mod.LWin, 400, 0.0)

    if (N_var in list(dso_mod.variables)):
        check(dso_mod.N, 1.0, 0.0)

    return dso_mod




### Functions ###
#Note this function has issues if the glacier extent is near a border already:
#In that case the index-1 or +2 may result in index error or using the last value
#This should however never be the case as this script is only applied if we take a larger extent around the glacier for e.g., 
#Mölgs radiation scheme

def bbox_2d_array(mask, arr, varname):
    if arr.ndim == 1:
        if varname in ['time','Time']:
            i_min = 0
            i_max = None
        elif varname in ['lat','latitude']:
            ix = np.where(np.any(mask == 1, axis=1))[0]
            i_min, i_max = ix[[0, -1]]
            i_min = i_min -1
            i_max = i_max +2
        elif varname in ['lon','longitude']:
            ix = np.where(np.any(mask == 1, axis=0))[0]
            i_min, i_max = ix[[0, -1]]
            i_min = i_min -1
            i_max = i_max +2
        bbox = arr[i_min:i_max]
    elif arr.ndim == 2:
        ix_c = np.where(np.any(mask == 1, axis=0))[0]
        ix_r = np.where(np.any(mask == 1, axis=1))[0]
        c_min, c_max = ix_c[[0, -1]]
        r_min, r_max = ix_r[[0, -1]]
    
        #Draw box with one non-value border
        #Now we got bounding box -> just add +1 / +2 at every index and voila
        bbox = arr[r_min-1:r_max+2,c_min-1:c_max+2]
    elif arr.ndim == 3:
        ix_c = np.where(np.any(mask == 1, axis=0))[0]
        ix_r = np.where(np.any(mask == 1, axis=1))[0]
        c_min, c_max = ix_c[[0, -1]]
        r_min, r_max = ix_r[[0, -1]]
        bbox = arr[:, r_min-1:r_max+2,c_min-1:c_max+2]
    return bbox 


def add_variable_along_timelatlon(ds, var, name, units, long_name):
    """ This function adds missing variables to the DATA class """
    if WRF:
         ds[name] = (('time','south_north','west_east'), var)	
    else:
        ds[name] = (('time','lat','lon'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    return ds

def add_variable_along_latlon(ds, var, name, units, long_name):
    """ This function self.adds missing variables to the self.DATA class """
    if WRF: 
        ds[name] = (('south_north','west_east'), var)
    else:
        ds[name] = (('lat','lon'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    ds[name].encoding['_FillValue'] = -9999
    return ds

def check(field, max, min):
    '''Check the validity of the input data '''
    if np.nanmax(field) > max or np.nanmin(field) < min:
        print('\n\nWARNING! Please check the data, its seems they are out of a reasonable range %s MAX: %.2f MIN: %.2f \n' % (str.capitalize(field.name), np.nanmax(field), np.nanmin(field)))
     
def check_for_nan(ds):
    if WRF is True:
        for y,x in product(range(ds.dims['south_north']),range(ds.dims['west_east'])):
            mask = ds.MASK.sel(south_north=y, west_east=x)
            if mask==1:
                if np.isnan(ds.sel(south_north=y, west_east=x).to_array()).any():
                    print('ERROR!!!!!!!!!!! There are NaNs in the dataset')
                    sys.exit()
    else:
        for y,x in product(range(ds.dims['lat']),range(ds.dims['lon'])):
            mask = ds.MASK.isel(lat=y, lon=x)
            if mask==1:
                if np.isnan(ds.isel(lat=y, lon=x).to_array()).any():
                    print('ERROR!!!!!!!!!!! There are NaNs in the dataset')
                    sys.exit()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Crop input file to glacier bounding box.")
    parser.add_argument('-i', '-input_file', dest='input_file', help='Input file to crop to glacier')
    parser.add_argument('-o', '-cosipy_file', dest='cosipy_file', help='Name of resulting COSIPY file')
    
    args = parser.parse_args()
    print('Read input file %s \n' % (args.input_file))
    ds = xr.open_dataset(args.input_file)
    dso_mod = crop_file_to_glacier(ds)
    ## write out to file ##
    print("Writing cropped cosipy file.")
    dso_mod.to_netcdf(args.cosipy_file)
