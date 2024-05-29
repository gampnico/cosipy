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

# def get_obs_heights(
#     filename: str, var: str, period: list = None, offset: float = False
# ) -> pd.DataFrame:
#     df = pd.read_csv(filename, index_col="TIMESTAMP", parse_dates=True)
#     if period:
#         df = df[period[0] : period[1]]
#     if offset:
#         start_point = df[var].iloc[0]
#         data_values = start_point - df[var]
#         height_drop = data_values.iloc[-1] - data_values.iloc[0]
#     else:
#         data_values = df[var]
#         height_drop = data_values.iloc[-1]
#     plt.figure()
#     plt.plot(
#         df.index,
#         data_values,
#         color="black",
#         label=f"{var} [Ref]",
#     )
#     ax = plt.gca()
#     ax.xaxis.set_major_formatter(
#         mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
#     )

#     plt.text(
#         df.index[0],
#         data_values.iloc[0],
#         f"Obs: {height_drop:.6f}",
#     )
#     output_name = get_output_path(
#         var_name=var, start_time=period[0], end_time=period[1]
#     )
#     plt.savefig(f"obs_{output_name}")

#     return df


def plot_1D_scatter(
    prediction: str,
    prediction_var: str,
    reference: str,
    reference_var: str,
    dividing_var: str = None,
    start: str = None,
    end: str = None,
    lat=None,
    lon=None,
    output_path: str = None,
    convert_units: bool = False,
):
    # Get dataset
    prediction_data = pu.read_data(filename=prediction)
    reference_data = pu.read_data(filename=reference)

    x_data = pu.get_plot_data(
        data=reference_data,
        name=reference_var,
        start=start,
        end=end,
        latitude=lat,
        longitude=lon,
        convert_units=convert_units,
    )
    y_data = pu.get_plot_data(
        data=prediction_data,
        name=prediction_var,
        start=start,
        end=end,
        latitude=lat,
        longitude=lon,
        convert_units=convert_units,
    )
    print(y_data.shape)
    y_data, mask = pu.remove_outliers(
        data=y_data, threshold=3, method="zscore"
    )
    x_data = x_data[mask]

    if dividing_var:
        z_data = pu.get_plot_data(
            data=prediction_data,
            name=dividing_var,
            start=start,
            end=end,
            latitude=lat,
            longitude=lon,
            convert_units=False,
        )
        z_data = z_data[mask]
        print(z_data)
        x_data = [
            x_data[z_data[dividing_var] > 0],
            x_data[z_data[dividing_var] <= 0],
        ]
        y_data = [
            y_data[z_data[dividing_var] > 0],
            y_data[z_data[dividing_var] <= 0],
        ]
        print(y_data[0].shape)
        print(y_data[1].shape)
    pu.initialise_formatting(scale_pts=12)
    fig = plt.figure(figsize=(15, 15))
    axes = plt.gca()

    if not dividing_var:
        plt.scatter(
            x_data,
            y_data,
            # marker=".",
            s=20.0,
            color="black",
        )
    else:
        colours = pu.get_colours()
        colours = [("Snow", "firebrick", "x"), ("No Snow", "black", "o")]
        for i in range(len(x_data)):
            axes.scatter(
                x_data[i],
                y_data[i],
                s=50.0,
                marker=colours[i][2],
                label=colours[i][0],
                color=colours[i][1],
            )
            _, results = pu.get_linear_regression(
                x_data=x_data[i], y_data=y_data[i]
            )
            pu.get_best_fit_line(
                axis=axes, results=results, colour=colours[i][1]
            )
            # plt.tight_layout()
        _, ymax = axes.get_ylim()
        _, xmax = axes.get_xlim()
        axes.set_ylim(0, 6)
        axes.set_xlim(0, xmax)
        plt.legend()
    # max_bin = math.ceil(
    #     max(np.nanmax(x_data.values), np.nanmax(y_data.values))
    # )
    # bin_intervals = np.arange(0, max_bin, 0.1)
    # fig, axes = plt.subplots(tight_layout=True)
    # hist = axes.hist2d(
    #     x_data,
    #     y_data,
    #     bins=bin_intervals,
    #     norm=matplotlib.colors.LogNorm(),
    # )
    # hist = axes.hist(y_data, bins=bin_intervals)
    label_text = pu.label_selector(dependent=reference_var)
    x_label = pu.merge_label_with_unit(label=label_text)
    label_text = pu.label_selector(dependent=prediction_var)
    y_label = pu.merge_label_with_unit(label=label_text)

    axes.set_xlabel(f"{x_label}")
    axes.set_ylabel(f"{y_label}")

    pu.set_output_path(
        output_path=output_path,
        var_name=f"regression-{reference_var}_{prediction_var}",
        start_time=start,
        end_time=end,
    )
    pu.save_figure(figure=fig, timestamp=output_path, img_format="svg")


def get_user_arguments():
    parser = pu.get_arg_parser()
    parser.add_argument(
        "--split_var",
        dest="split_var",
        default=None,
        help="Split threshold",
        type=str,
    )

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":

    args = get_user_arguments()

    plot_1D_scatter(
        prediction=args.file,
        prediction_var=args.var,
        reference=args.reference,
        reference_var=args.ref_var,
        dividing_var=args.split_var,
        start=args.start,
        end=args.end,
        lat=args.lat,
        lon=args.lon,
        output_path=args.output_path,
        convert_units=args.convert_units,
    )
