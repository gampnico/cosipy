import matplotlib

matplotlib.use("TkAgg")
import argparse
import math
import os
import re

import matplotlib.cm as cm
import plotting_utilities as pu
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.colors import BoundaryNorm
from scipy import interpolate
from scipy.interpolate import griddata
import numpy as np
import statsmodels
import datetime


def validate_model(
    filename: str,
    start: str = None,
    end: str = None,
    lat=None,
    lon=None,
    output_path: str = None,
):
    # Get dataset
    prediction_data = pu.read_data(filename=filename)
    snowheights = pu.get_plot_data(
        data=prediction_data,
        name="SNOWHEIGHT",
        start=start,
        end=end,
        latitude=lat,
        longitude=lon,
        convert_units=False,
    )

    var_name = "surfM"

    # verify surface melt
    x_data = pu.get_plot_data(
        data=prediction_data,
        name=var_name,
        start=start,
        end=end,
        latitude=lat,
        longitude=lon,
    )
    snowfall = pu.get_plot_data(
        data=prediction_data,
        name="SNOWFALL",
        start=start,
        end=end,
        latitude=lat,
        longitude=lon,
    )
    x_data = [
        x_data[snowheights["SNOWHEIGHT"] > 0],
        x_data[snowheights["SNOWHEIGHT"] <= 0],
    ]
    assert not (x_data[1][var_name].isna().any())

    x_mask = x_data[1] > 0.0
    if x_mask[var_name].any(skipna=True):
        counts = x_data[1][x_mask][var_name].count()
        print(f"Counts:{counts}")
        mask_index = x_data[1][x_mask][var_name].dropna().index

        # surface melt can occur if snow from previous timestep melts in current timestep.
        test_matrix = {}
        test_matrix["snowheight_previous"] = (
            snowheights.loc[mask_index - datetime.timedelta(hours=1)] > 0.0
        )
        test_matrix["snowheight_current"] = snowheights.loc[mask_index] > 0.0
        test_matrix["snowfall_current"] = snowfall.loc[mask_index] > 0.0

        for k, v in test_matrix.items():
            keyname = k.split("_")

            print(
                f"{keyname[0].capitalize()} in {keyname[1]} timestep: {v.value_counts(dropna=True)}"
            )

    # verify internal temperatures
    raw_data = pu.get_spatiotemporal_domain(
        data=prediction_data, start=start, end=end, latitude=lat, longitude=lon
    )
    layer_temperature = pu.get_squeezed_data(
        data=raw_data, name="LAYER_T", latitude=lat, longitude=lon
    )
    layer_ntype = pu.get_squeezed_data(
        data=raw_data, name="LAYER_NTYPE", latitude=lat, longitude=lon
    )
    layer_density = pu.get_squeezed_data(
        data=raw_data, name="LAYER_RHO", latitude=lat, longitude=lon
    )

    print(layer_ntype)
    temperature_mask = layer_temperature["LAYER_T"].where(layer_ntype["LAYER_NTYPE"]==0)
    print(temperature_mask.shape)
    # temperature_mask = layer_temperature.values[layer_ntype.values==0]
    assert np.isfinite(temperature_mask).all()
    print(temperature_mask[temperature_mask>273.15].shape)
    print(layer_temperature.values.shape)
    
    assert np.isfinite(layer_density.values)
    print(layer_density[temperature_mask<0].shape)

    pu.set_output_path(
        output_path=output_path,
        var_name=f"model_validation",
        start_time=start,
        end_time=end,
    )


if __name__ == "__main__":

    args = pu.get_user_arguments()

    validate_model(
        filename=args.file,
        start=args.start,
        end=args.end,
        lat=args.lat,
        lon=args.lon,
        output_path=args.output_path,
    )
