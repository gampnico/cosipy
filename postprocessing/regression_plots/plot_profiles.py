import matplotlib
import matplotlib.colors
import matplotlib.ticker
import scipy.interpolate

matplotlib.use("TkAgg")
import argparse
import os

import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotting_utilities as pu
import scipy
import scipy.interpolate as scinterp
import seaborn as sns
import xarray as xr
from matplotlib.colors import BoundaryNorm


def plot_profile(filename, pdate, lat, lon):
    """This creates a simple plot showing the 2D fields"""
    DATA = xr.open_dataset(filename)

    (c_y, c_x) = pu.naive_fast(DATA.lat.values, DATA.lon.values, lat, lon)
    DATA = DATA.sel(time=pdate, west_east=c_x, south_north=c_y)

    plt.figure(figsize=(20, 12))

    depth = np.append(0, np.cumsum(DATA.LAYER_HEIGHT.values))
    rho = np.append(DATA.LAYER_RHO[0], DATA.LAYER_RHO.values)
    plt.step(rho, depth)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.show()


def plot_profile_1D_timeseries(
    filename,
    var,
    domainy=None,
    start=None,
    end=None,
    lat=None,
    lon=None,
    stake_file=None,
    pit_name=None,
    plot_layer: bool = True,
    output_path: str = None,
):

    ds = xr.open_dataset(filename)
    ds = pu.get_spatiotemporal_domain(
        data=ds, start=start, end=end, latitude=lat, longitude=lon
    )

    # Get first layer height
    fl = ds.attrs["First_layer_height_log_profile"]

    var, barLabel = get_bar_label(data=ds, name=var)
    V = pu.get_squeezed_data(ds, var, lat, lon) - 273.15
    if var == "LAYER_NTYPE":
        density = pu.get_squeezed_data(ds, "LAYER_RHO", lat, lon)
        V = xr.where((density < ds.ice_density) & (V != 1), -1, V)
        V = xr.where((V == 1), V - 0.001, V)  # until the cbar gets fixed
    D = pu.get_squeezed_data(ds, "LAYER_HEIGHT", lat, lon).cumsum(
        axis=1, skipna=True
    )
    V["LAYER_HEIGHT"] = D
    depth = pu.get_squeezed_data(ds, "TOTALHEIGHT", lat, lon).values

    c_levels = 32
    colourmap = get_colourmap(bar_label=var, levels=c_levels)
    colourmap = set_colourmap(cmap=colourmap, name=var, levels=c_levels)
    pu.initialise_formatting(scale_pts=8)
    fig = plt.figure(figsize=(20, 10))
    ax = plt.gca()
    V = V.transpose()
    if not plot_layer:
        CS = ax.pcolormesh(
            V.time,
            depth - V.LAYER_HEIGHT,
            V,
            cmap=colourmap["palette"],
            norm=colourmap["norm"],
            vmin=colourmap["vmin"],
            vmax=colourmap["vmax"],
        )
        cbar = plt.colorbar(CS)
        y_label = "Glacier Height [m]"
        # ax.set_yscale("log", base=2)
    else:
        max_layers = ds.LAYER_HEIGHT.count(dim="layer").max()
        # divnorm = matplotlib.colors.TwoSlopeNorm(
        #     vmin=colourmap["vmin"],
        #     vcenter=0,
        #     vmax=colourmap["vmax"],
        # )
        CS = V.plot.pcolormesh(
            "time",
            "layer",
            ax=ax,
            yincrease=False,
            cbar_kwargs=colourmap["cbar_args"],
            cmap=colourmap["palette"],
            # cmap=sns.diverging_palette(220, 20, as_cmap=True),
            # cmap=sns.color_palette("seismic", as_cmap=True),
            # center=colourmap["c_map_center"],
            levels=c_levels,
            yticks=V.layer,
            ylim=max_layers,
            vmin=colourmap["vmin"],
            vmax=colourmap["vmax"],
            # norm=divnorm,
        )
        cbar = CS.colorbar
        y_label = "Model Layer [-]"
    if domainy:
        ax.set_ylim(domainy[0], domainy[1])
    cbar.ax.set_ylabel(barLabel)
    cbar.ax.set_yscale("linear")

    # cbar.ax.set_yscale("log")
    # cbar.ax.set_ylim(colourmap["vmin"], colourmap["vmax"])
    # cbar.formatter = matplotlib.ticker.LogFormatter(10, labelOnlyBase=False)
    # cbar.locator = matplotlib.ticker.LogLocator(subs="all")
    # cbar.minorlocator = matplotlib.ticker.LogLocator(subs="all")
    # cbar.formatter = matplotlib.ticker.Formatt

    cbar.ax.set_ylim(-50, 0)
    cbar.locator = matplotlib.ticker.AutoLocator()
    # cbar.locator = matplotlib.ticker.MultipleLocator(base=90,offset=50)
    # cbar.locator = matplotlib.ticker.LinearLocator()
    cbar.minorlocator = matplotlib.ticker.AutoMinorLocator()

    # ticks = np.linspace(colourmap["vmin"],colourmap["vmax"],5)
    # ticks = [-45,-40,-35,-30,-25,-20,-15,-10,-5,0,5,10,15,20,25]
    # cbar.set_ticks(ticks)
    cbar.update_ticks()

    pu.format_y_ticks(axis=ax, label=y_label)
    pu.format_date_ticks(axes=[ax], fmt="%b\n%Y", minor_fmt="%b")
    ax.set_xlabel("Date")
    if not output_path:
        output_path = pu.get_output_path(
            var_name=f"{var}_test.png", start_time=start, end_time=end
        )

    pu.save_figure(figure=fig, timestamp=output_path, img_format="png")


def get_colourmap(bar_label: str, levels: int = None) -> dict:
    palette = plt.get_cmap("YlGnBu_r", levels).with_extremes(
        under="#e983f4", bad="#a5a5a5"
    )
    # cmap = set_cmap_errors(cmap=cmap)
    colourmap = {}
    colourmap["palette"] = palette
    # colourmap["c_map"] = plt.get_cmap("YlGnBu_r", levels)
    # colourmap["c_map"]=colourmap["c_map"].set_bad("grey")
    # colourmap["c_map"]=colourmap["c_map"].set_over("#ff3030")
    # colourmap["c_map"]=colourmap["c_map"].set_under("#e983f4")
    colourmap["c_map_center"] = None
    colourmap["cbar_args"] = {"label": bar_label,"extend":"neither"}
    colourmap["vmin"] = None
    colourmap["vmax"] = None
    colourmap["norm"] = None

    return colourmap


def set_colourmap(cmap: dict, name: str, levels: int = None):
    palette = None
    if name == "LAYER_T":
        palette = plt.get_cmap("Blues_r", levels)
        # palette = plt.get_cmap("RdBu_r", levels)
        # palette = sns.color_palette("seismic", as_cmap=True, n_colors=levels),
        # cmap["c_map_center"] = 273.16
        # cmap["c_map_center"] = 0.0
        cmap["vmin"] = -50
        cmap["vmax"] = 0
    elif name == "LAYER_G_PENETRATING":
        palette = plt.get_cmap("bwr", levels)
        cmap["c_map_center"] = 0.0
    elif name == "LAYER_RHO":
        # palette = plt.get_cmap("ocean_r", levels)
        colors = [
            "#ffffff",
            "#aad4e3",
            "#55aac6",
            "#0080aa",
            "#00558e",
            "#002a71",
            "#000055",
            # "#002a39",
            # "#00551c",
            # "#008000",
            "#390f00",
            "#714700",
            "#8e3900",
        ]
        palette = matplotlib.colors.LinearSegmentedColormap.from_list(
            "ocean_test", colors=colors, N=levels
        )
        cmap["vmin"] = 0
        cmap["vmax"] = 3000
        cmap["c_map_center"] = 900
        densities = [100, 200, 300, 400, 830, 900]
        densities = {
            "new": 100,
            "damp": 200,
            "settled": 300,
            "packed": 400,
            "firn": 830,
            "ice": 2000,
            "rock": 3000,
        }
    elif name == "LAYER_ICE_FRACTION":
        palette = plt.get_cmap("Blues", levels)
        cmap["vmax"] = 1.0 + 1e-6
    elif name == "LAYER_NTYPE":
        palette = matplotlib.colors.ListedColormap(
            ["cyan", "blue", "black"], "indexed"
        )
        # bounds = [-1, 0, 1]
        # cmap["norm"] = matplotlib.colors.BoundaryNorm(bounds, cmap["c_map"].N)
        # cmap["c_map"] = plt.get_cmap("tab20c_r", 3)
        cmap["c_labels"] = ["snow", "ice", "debris"]
        # cmap["cbar_args"]["format"] = plt.FuncFormatter(
        #     lambda val, loc: cmap["c_labels"][int(val)]
        # )
        # cmap["cbar_args"]["ticks"] = [-1, 0, 1]
        cmap["vmin"] = -1
        cmap["vmax"] = 1
    elif name not in ["LAYER_CC", "LAYER_REFREEZE"]:
        cmap["vmin"] = 0

    if cmap["c_map_center"]:
        cmap["norm"] = matplotlib.colors.TwoSlopeNorm(
            vmin=cmap["vmin"],
            vcenter=cmap["c_map_center"],
            vmax=cmap["vmax"],
        )
        # cmap["norm"] = matplotlib.colors.CenteredNorm(
        #     vcenter=cmap["c_map_center"]
        # )
    # else:
    #     cmap["norm"] = None
    if palette:
        cmap["palette"] = set_cmap_errors(cmap=palette)

    return cmap


def set_cmap_errors(cmap):
    # cmap.set_bad("grey")
    cmap.set_bad("#a5a5a5")
    # cmap.set_over("#ff3030")
    cmap.set_over("#ffffff")
    cmap.set_under("#e983f4")

    return cmap


def get_variable_name(name: str) -> str:
    var_dict = {
        "T": "T",
        "RHO": "RHO",
        "IF": "ICE_FRACTION",
        "REF": "REFREEZE",
        "LWC": "LWC",
        "POR": "POROSITY",
        "DEPTH": "HEIGHT",
        "HEIGHT": "HEIGHT",
        "CC": "CC",
        "IRR": "IRREDUCIBLE_WATER",
        "THERM": "THERMAL_CONDUCTIVITY",
        "NTYPE": "NTYPE",
    }

    if name in var_dict.keys():
        name = var_dict[name]
    return name


def get_bar_label(data: xr.Dataset, name: str) -> tuple:
    name = get_variable_name(name)
    if f"LAYER_{name}" in data.variables.keys():
        name = f"LAYER_{name}"
        units = f"{data[name].units}"
        barLabel = f"{data[name].long_name} [${units}$]"
    else:
        barLabel = name
    return name, barLabel


def get_minimum_layer_height(data: xr.Dataset):
    height = pu.get_squeezed_data(data=data, name="LAYER_HEIGHT").min(
        skipna=True
    )
    return height


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quick plot of the results file."
    )
    parser.add_argument(
        "-f", "--file", dest="file", help="Path to the result file"
    )
    parser.add_argument(
        "-d", "--date", dest="pdate", help="Date of the profile plot"
    )
    parser.add_argument(
        "-v",
        "--var",
        dest="var",
        default="RHO",
        help="Which variable to plot (e.g. T, RHO, etc.)",
    )
    parser.add_argument(
        "-n",
        "--lat",
        dest="lat",
        default=None,
        help="Latitude value in case of 2D simulation",
        type=float,
    )
    parser.add_argument(
        "-m",
        "--lon",
        dest="lon",
        default=None,
        help="Longitude value in case of 2D simulation",
        type=float,
    )
    parser.add_argument(
        "-s",
        "--start",
        dest="start",
        default=None,
        help="Start date for the time plot",
    )
    parser.add_argument(
        "-e",
        "--end",
        dest="end",
        default=None,
        help="End date for the time plot",
    )
    parser.add_argument(
        "--stake-file",
        dest="stake_file",
        default=None,
        help="Path to the stake data file",
    )
    parser.add_argument(
        "--pit",
        dest="pit_name",
        default=None,
        help="Name of the pit in the stake data file",
    )
    parser.add_argument(
        "-y",
        "--depth",
        dest="d",
        nargs="+",
        default=None,
        help="An array with depth values for which the corresponding values are to be displayed",
        type=float,
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        default=None,
        help="Path to output image",
    )
    parser.add_argument(
        "-l",
        "--plot_layer",
        dest="plot_layer",
        action="store_true",
        default=False,
        help="Plot by layer.",
    )
    args = parser.parse_args()

    if args.pdate is None:
        plot_profile_1D_timeseries(
            filename=args.file,
            var=args.var,
            domainy=args.d,
            start=args.start,
            end=args.end,
            stake_file=args.stake_file,
            pit_name=args.pit_name,
            output_path=args.output_path,
            plot_layer=args.plot_layer,
        )
    else:
        plot_profile(args.file, args.pdate, args.d, args.lat, args.lon)
