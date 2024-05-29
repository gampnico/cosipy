"""
 This is the configuration (init) file for the utility aws2cosipy.
 Please make your changes here.
"""

# TODO: Verify lapse rates!

#------------------------
# Declare variable names 
#------------------------

# Pressure
PRES_var = 'P'

# Temperature
T2_var = 'T'
in_K = False

# Relative humidity
RH2_var = 'RH'

# Incoming shortwave radiation
G_var = 'SWIN'

# Precipitation
RRR_var = 'PP'

# Wind velocity
U2_var = 'FF'

# Incoming longwave radiation
LWin_var = 'LWIN'

# Snowfall
SNOWFALL_var = 'SNOWFALL'

# Cloud cover fraction
N_var = 'N'

#------------------------
# Aggregation to hourly data
#------------------------
aggregate = True
aggregation_step = 'H'

# Delimiter in csv file
delimiter = ','

# WRF non uniform grid
WRF = False

#------------------------
# Radiation module 
#------------------------
radiationModule = 'Wohlfahrt2016' # 'Moelg2009', 'Wohlfahrt2016', 'none'
LUT = False                   # If there is already a Look-up-table for topographic shading and sky-view-factor built for this area, set to True

dtstep = 3600               # time step (s)
stationLat = 46.496            # Latitude of station
tcart = 26                    # Station time correction in hour angle units (1 is 4 min)
timezone_lon = 10.569	      # Longitude of station

# Zenit threshold (>threshold == zenit): maximum potential solar zenith angle during the whole year, specific for each location
# overwritten in aws2cosipy
zeni_thld = 85.0              # If you do not know the exact value for your location, set value to 89.0

#------------------------
# Point model 
#------------------------
point_model = True
plon = 10.569
plat = 46.496
hgt = 2625.0

# ECCI
# plon = 10.560
# plat = 46.498
# hgt = 2780.0

# ECCD
# plon = 10.572
# plat = 46.495
# hgt = 2600.0

#------------------------
# Interpolation arguments 
#------------------------
stationName = 'Suldenferner AWS 2016'
stationAlt = 2625.0

lapse_T         = -0.006    # Temp K per  m
lapse_RH        =  0.000    # RH % per  m (0 to 1)
lapse_RRR       =  0.0000   # mm per m
lapse_SNOWFALL  =  0.0000   # Snowfall % per m (0 to 1)
