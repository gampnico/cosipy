import matplotlib
import matplotlib.axes
import matplotlib.dates
import matplotlib.ticker

matplotlib.use("TkAgg")
import argparse
import math
import os
import re

import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import xarray as xr
from matplotlib.colors import BoundaryNorm
from scipy import interpolate
from scipy.interpolate import griddata


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description="Quick plot of the results file."
    )
    parser.add_argument(
        "-f",
        "--file",
        required=False,
        dest="file",
        help="Path to the result file",
    )
    parser.add_argument(
        "-r",
        "--reference",
        required=False,
        dest="reference",
        help="Path to observation data",
    )
    parser.add_argument(
        "-x",
        "--ref-var",
        dest="ref_var",
        default="T2",
        help="Reference variable name",
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
        required=False,
        dest="output_path",
        default=None,
        type=str,
        help="Path to output file",
    )
    parser.add_argument(
        "-c",
        "--convert_units",
        dest="convert_units",
        action="store_true",
        default=False,
        help="Convert to prettier units.",
    )

    return parser


def get_user_arguments():
    parser = get_arg_parser()
    arguments = parser.parse_args()

    return arguments


def get_colours() -> dict:
    colours = {
        "debris": "#A36900",
        "debris_dark": "#977230",
        "debris_normal": "#C49644",
        "pale_debris": "#F4B183",
        "clean_ice": "#003BA3",
        "clean_ice_dark": "#305597",
        "clean_ice_normal": "#4472C4",
        "silver": "#E0E0E0",
        "dark_silver": "#AFAFAF",
        "light_blue_grey": "#DAE3F3",
        "pink_purple": "#C444B2",
        "pink": "#E983F4",
        "medium_green": "#44C456",
        "tree_green": "#228B22",
    }
    return colours


def get_colourmap(name: str = None):
    colourmap = {
        "energy_fluxes": (
            "#228B22",
            "#1C86EE",
            "#8B1A1A",
            "#7F7F7F",
            "#FF3030",
        ),
        "blues": ("#3DC1F9", "#1897FF", "#4169E1", "#360CF9"),
        "reds": ("#5E1F1F", "#AB3939", "#FF6161", "#D27878"),
        "greys": ("#3F3F3F", "#727272", "#A5A5A5", "#D8D8D8"),
    }

    if name:
        colourmap = colourmap[name]

    return colourmap


def initialise_formatting(
    small=14, medium=16, large=18, extra=20, scale_pts: int = None
):
    """Initialises font sizes for matplotlib figures.

    Called separately from other plotting functions in this module
    to avoid overwriting on-the-fly formatting changes.

    Args:
        small (int): Size of text, ticks, legend. Default 14.
        medium (int): Size of axis labels. Default 16.
        large (int): Size of axis title. Default 18.
        extra (int): Size of figure title (suptitle). Default 20.
    """
    if scale_pts:
        small += scale_pts
        medium += scale_pts
        large += scale_pts
        extra += scale_pts

    rc_settings = {
        "font": {
            "size": small,  # default text size
            "family": "sans-serif",  # default family
            "sans-serif": ["Inter"],  # default font
        },
        "axes": {
            "titlesize": large,  # size of axes title font
            "labelsize": medium,  # size of x and y label fonts
            "autolimit_mode": "round_numbers",  # round ticks to nearest integer
        },
        "xtick": {"labelsize": small},  # size of tick label font
        "ytick": {"labelsize": small},  # size of tick label font
        "legend": {"fontsize": small},  # size of legend font
        "figure": {"titlesize": extra},  # size of figure title's font
        # "text": {"usetex": True},  # format with TeX
    }

    for k, v in rc_settings.items():
        plt.rc(k, **v)

    # plt.rc("axes", spines[["right", "top"]].set_visible(False))
    # plt.rc("spines", set_visible=False)


def parse_formatting_kwargs(axis, **kwargs):
    """Parses kwargs for generic formatting.

    Applies horizontal or vertical lines, updates axis title and
    labels.

    Args:
        axis (plt.Axes): Target axes.

    Keyword Args:
        hlines (dict): Name and y-axis value for which to plot a
            horizontal line.
        title (str): Axis title.
        vlines (dict): Name and x-axis value for which to plot a
            vertical line.
        x_label (str): X-axis label.
        y_label (str): Y-axis label.
    """

    hlines = kwargs.get("hlines")
    title = kwargs.get("title")
    vlines = kwargs.get("vlines")
    x_label = kwargs.get("x_label")
    y_label = kwargs.get("y_label")

    plot_constant_lines(axis=axis, hlines=hlines, vlines=vlines)
    if title:
        plt.title(title, fontweight="bold")
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)


def format_y_ticks(
    axis: matplotlib.axes.Axes, label: str = "", base: int = 10
):
    # axis.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
    axis.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=base))
    axis.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    # plt.ylim(top=plt.yticks()[-1][0])
    # axis.major
    # majorticks.set_va("top")
    axis.set_ylabel(label)
    return axis


def format_date_ticks(
    axes, fmt: str = "%b %Y", minor_fmt="%b %Y", bar: bool = False
):
    major_fmt = mdates.DateFormatter(fmt=fmt)
    minor_fmt = mdates.DateFormatter(fmt=minor_fmt)
    for ax in axes:
        if not bar:
            ax.xaxis.set_major_locator(
                mdates.MonthLocator(bymonth=range(1, 13, 2))
            )
            ax.xaxis.set_minor_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(major_fmt)
            ax.xaxis.set_minor_formatter(minor_fmt)
            # ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

        else:
            # ax.xaxis.set_major_locator(mdates.YearLocator(day=16))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=16))
            # ax.xaxis.set_major_formatter(major_fmt)
            ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
            ax.xaxis.set_minor_formatter(minor_fmt)
            ax.tick_params(
                axis="x", which="minor", tick1On=False, tick2On=False
            )
            # Align the minor tick label
            for label in ax.get_xticklabels(minor=True):
                label.set_horizontalalignment("center")

        # ax.xaxis.set_minor_locator(matplotlib.dates.AutoDateLocator(interval_multiples=True))

    return axes


def label_selector(dependent):
    """Constructs label parameters from dependent variable name.

    If the variable name is not implemented, the unconverted
    variable is passed directly, with the symbol duplicated from its
    name and with no unit provided.

    Args:
        dependent (str): Name of dependent variable.

    Returns:
        tuple[str, str, str]: Name of dependent variable, symbol in
        LaTeX format, unit in LaTeX format -> (name, symbol, unit).
    """

    units = {
        "empty": "-",
        "energy_flux": r"W$\cdot$m$^{-2}$",
        "mass_flux": r"kg$\cdot$m$^{-2}$",
        "mwe_mm": r"mm m.w.e.",
        "speed": r"ms$^{-2}$",
    }
    implemented_labels = {
        "air_pressure": ("Water Vapour Pressure", r"$e$", "Pa"),
        "albedo": ("Albedo", r"$\alpha$", "-"),
        "b": ("Ground Heat Flux", r"$Q_{G}$", units["energy_flux"]),
        "boundary_layer_height": (
            "Boundary Layer Height",
            r"$z_{BL}$",
            "m",
        ),
        "condensation": ("Condensation", r"$C$", units["mass_flux"]),
        "cn2": (
            "Structure Parameter of Refractive Index",
            r"$C_{n}^{2}$",
            "",
        ),
        "ct2": ("Structure Parameter of Temperature", r"$C_{T}^{2}$", ""),
        "deltaheight": ("Relative Surface Height", r"$h_{rel}$", "m"),
        "deposition": ("Deposition", r"$D$", units["mass_flux"]),
        "depth": ("Depth", r"$z$", "m"),
        "environmental_lapse_rate": (
            "Environmental Lapse Rate",
            r"\Gamma_{e}",
            "Km$^{-1}$",
        ),
        "dry_adiabatic_lapse_rate": (
            "Dry Adiabatic Lapse Rate",
            r"\Gamma_{d}",
            "Km$^{-1}$",
        ),
        "energy_flux": ("Energy Flux", r"$Q$", units["energy_flux"]),
        "evaporation": ("Evaporation", r"$E$", units["mass_flux"]),
        "g": (
            "Incoming Shortwave",
            r"$Q_{S}\downarrow$",
            units["energy_flux"],
        ),
        "grad_potential_temperature": (
            "Gradient of Potential Temperature",
            r"$\Delta \theta$",
            r"K$\cdot$m$^{-1}$",
        ),
        "h": ("Sensible Heat Flux", r"$Q_{H}$", units["energy_flux"]),
        "h_free": (
            "Sensible Heat Flux",
            r"$Q_{H free}$",
            units["energy_flux"],
        ),
        "height": ("Total Height", r"$h$", "m"),
        "hour": ("Time of Day", r"Time of Day", "h"),
        "humidity": ("Humidity", r"$\rho_{v}$", r"kg$\cdot$m$^{-3}$"),
        "layer": ("Layer", r"Layer", units["empty"]),
        "le": ("Latent Heat Flux", r"$Q_{L}$", units["energy_flux"]),
        "intmb": ("Interior Mass Balance", r"$MB$", units["mwe_mm"]),
        "mass_flux": ("Mass Flux", r"$MF$", units["mass_flux"]),
        "mb": ("Mass Balance", r"$MB$", units["mwe_mm"]),
        "me": ("Melt Energy", r"$Q_{M}$", units["energy_flux"]),
        "mixing_ratio": ("Mixing Ratio", r"$r$", units["energy_flux"]),
        "moist_adiabatic_lapse_rate": (
            "Moist Adiabatic Lapse Rate",
            r"\Gamma_{m}",
            "Km$^{-1}$",
        ),
        "mol": ("Monin-Obukhov Length", r"$L_{Ob}$", "m"),
        "msl_pressure": ("Mean Sea-Level Pressure", r"$P_{MSL}$", "Pa"),
        "obukhov": ("Obukhov Length", r"$L_{Ob}$", "m"),
        "pcpn": ("Total Precipitation", r"$PCPN$", "mm"),
        "potential_temperature": (
            "Potential Temperature",
            r"$\theta$",
            "K",
        ),
        "pressure": ("Pressure", r"$P$", "mbar"),
        "q": ("Meltwater Runoff", r"$R_{M}$", units["energy_flux"]),
        "qps": (
            "Penetrating Shortwave",
            r"$Q_{PS}$",
            units["energy_flux"],
        ),
        "qrr": ("Rain Heat Flux", r"$Q_{R}$", units["energy_flux"]),
        "rain": ("Rain", r"$RA$", "mm"),
        "refreeze": ("Refreeze", r"$REFR$", units["mass_flux"]),
        "rh2": ("2m Relative Humidity", r"$RH_{2m}$", units["empty"]),
        "rho_air": ("Air Density", r"$\rho_{air}$", r"kg$\cdot$m$^{3}$"),
        "rrr": ("Rain", r"$Rain$", "mm"),
        "saturated_temperature": (
            "Parcel Temperature (Saturated)",
            r"$T_{sat}$",
            "K",
        ),
        "shf": ("Sensible Heat Flux", r"$Q_{H}$", units["energy_flux"]),
        "snow": ("Snow Height", r"$h_{SNOW}$", "m"),
        "snowfall": ("Snowfall", r"$SN$", "mm"),
        "snowheight": ("Snow Height", r"$h_{SNOW}$", "m"),
        "sublimation": ("Sublimation", r"$S$", units["mass_flux"]),
        "subm": ("Subsurface Melt Rate", r"$M_{sub}$", units["mwe_mm"]),
        "surfmb": ("Surface Mass Balance", r"$MB_{s}$", units["mwe_mm"]),
        "surfm": ("Surface Melt Rate", r"$M_{s}$", units["mwe_mm"]),
        "t": ("Temperature", r"$T$", r"˚C"),
        "temperature": ("Temperature", r"$T$", r"˚C"),
        "t2": ("2m Temperature", r"$T_{2m}$", r"˚C"),
        "theta_star": ("Temperature Scale", r"$\theta^{*}$", "K"),
        "totalheight": ("Total Height", r"$h$", "m"),
        "totalmelt": ("Total Melt Rate", r"$M$", units["mwe_mm"]),
        "ts": ("Surface Temperature", r"$T_{s}$", "K"),
        "u2": ("2m Wind Speed", r"$U_{2m}$", units["speed"]),
        "u_star": ("Friction Velocity", r"$u^{*}$", r"m$\cdot$s$^{-2}$"),
        "unsaturated_temperature": (
            "Parcel Temperature (Unsaturated)",
            r"$T_{unsat}$",
            "K",
        ),
        "virtual_temperature": ("Virtual Temperature", r"$T_{v}$", "K"),
        "water_vapour_pressure": ("Water Vapour Pressure", r"$e$", "Pa"),
        "wind_speed": ("Wind Speed", r"$u$", units["speed"]),
        "z0": ("Roughness length", r"$Z_{0}$", "mm"),
    }

    name = dependent.lower()
    if name in implemented_labels:
        label = implemented_labels[name]
    else:
        label = (dependent.title(), f"${name}$", "")

    return label


def get_date_and_timezone(data):
    """Return first time index and timezone.

    Args:
        data (pd.DataFrame or pd.Series): TZ-aware dataframe with
            DatetimeIndex.

    Returns:
        dict[str, datetime.tzinfo]: Date formatted as
        "DD Month YYYY", and timezone object.
    """

    date = data.index[0].strftime("%d %B %Y")
    timezone = data.index.tz

    return {"date": date, "tzone": timezone}


def title_plot(title, timestamp, location=""):
    """Constructs title and legend.

    Args:
        title (str): Prefix to include in title.
        timestamp (Union[str, pd.Timestamp]): Date or time of data
            collection.
        location (str): Location of data collection. Default empty
            string.

    Returns:
        str: Title of plot with location and time.
    """

    if not location:
        location = ""  # Otherwise None interpreted literally in f-strings
    else:
        location = f"\nat {location}"

    if not isinstance(timestamp, str):
        timestamp = timestamp.strftime("%d %B %Y")

    title_string = f"{title}{location}, {timestamp}"
    plt.title(title_string, fontweight="bold")
    # plt.legend(loc="upper left")

    return title_string


def merge_label_with_unit(label, shortname=False):
    """Merges variable name with its unit if applicable.

    Args:
        label (tuple[str, str, str]): Contains the name, symbol, and
            unit of a variable. Supports both TeX and empty strings.
            Strings containing TeX must be passed as raw strings::

                label = ("Name", "Symbol", r"$Unit_{TeX}$")
                label = ("Name", r"$Symbol_{TeX}$", "")

    Returns:
        str: Formatted string with the name and unit of a variable.
    """

    if not shortname:
        idx = 0
    else:
        idx = 1
    if not label[2]:  # if no unit available
        merged_label = f"{label[idx]}"
    elif shortname == "unit":
        merged_label = f"{label[2]}"
    else:
        merged_label = f"{label[idx]} [{label[2]}]"

    return merged_label


def merge_multiple_labels(labels):
    """Merges multiple labels into a single formatted string.

    Args:
        labels (list[str]): Labels, which may contain duplicates.

    Returns:
        str: A formatted, punctuated string with no duplicates.
    """

    unique_text = list(dict.fromkeys(labels))
    if len(unique_text) < 2:
        merged = f"{unique_text[0]}"
    elif len(unique_text) == 2:
        merged = " and ".join(unique_text)
    else:
        merged = ", ".join(unique_text)

    return merged


def set_xy_labels(ax, name):
    """Sets labels for X (time), Y (variable) axis.

    Args:
        ax (plt.Axes): Plot's axes.
        timezone (datetime.tzinfo): Local timezone of data.
        name (str): Name or abbreviation of dependent variable.

    Returns:
        plt.Axes: Plot axes with labels for local time on the x-axis
        and for the dependent variable with units on the y-axis.
        Ticks on the x-axis are formatted at hourly intervals.
    """

    label_text = label_selector(dependent=name)
    x_label = merge_label_with_unit(label=label_text)
    y_label = merge_label_with_unit(label=label_text)

    plt.xlabel(f"Modelled {x_label}")
    plt.ylabel(f"Observed {y_label}")

    return ax


def get_ablation_season(
    filename: str = "../../data/input/Suldenferner/Suldenferner_aws_2016_surface_height.csv",
    name: str = "DELTAHEIGHT",
    start=None,
    end=None,
    lat=None,
    lon=None,
) -> tuple:
    data = read_data(filename=filename)
    data = get_plot_data(
        data=data,
        name=name,
        start=start,
        end=end,
        latitude=lat,
        longitude=lon,
    )
    daily_mean = data.resample("1D").mean()
    start_year = daily_mean.index[0].year
    ablation_start = daily_mean[f"{start_year}-05":].idxmax()
    ablation_end = daily_mean[ablation_start:].idxmin()

    if start is not None:
        ablation_start = max(ablation_start, start)
    if end is not None:
        ablation_end = max(ablation_end, end)

    return ablation_start, ablation_end


def plot_ablation_season(axis, colour: str = None):
    if not colour:
        colours = get_colours()
        colour = colours["pink"]
    ablation_start, ablation_end = get_ablation_season()
    axis.axvspan(ablation_start, ablation_end, alpha=0.2, color=colour)

    return axis


def set_timeseries_labels(
    axis,
    label: str,
    shortname: bool = False,
    show_ablation: bool = False,
    format_dates: bool = False,
) -> tuple:
    colours = get_colours()
    label_text = label_selector(dependent=label)
    y_label = merge_label_with_unit(label=label_text, shortname=shortname)
    axis.set_ylabel(y_label)
    plt.xlabel(f"Date")

    axis.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(axis.xaxis.get_major_locator())
    )
    if format_dates:
        format_date_ticks(axes=[axis])

    if show_ablation:
        plot_ablation_season(axis=axis, colour=colours["pink"])

    return label_text, y_label


def plot_constant_lines(axis, hlines=None, vlines=None):
    """Plots horizontal or vertical lines onto axis.

    Args:
        axis (plt.Axes): Target axes.
        hlines (dict[str, float]): Name and x-axis value for which
            to plot a horizontal line. Default None.
        vlines (dict[str, float]): Name and x-axis value for which
            to plot a vertical line. Default None.
    """

    if hlines:
        for key in hlines:
            if hlines[key]:
                axis.axhline(
                    hlines[key],
                    color="grey",
                    linestyle="--",
                    label=label_selector(dependent=key)[0],
                )
    if vlines:
        for key in vlines:
            if vlines[key]:
                axis.axvline(
                    vlines[key],
                    color="red",
                    label=label_selector(dependent=key)[0],
                )


def save_figure(
    figure,
    timestamp: str,
    suffix: str = "",
    img_format: str = "svg",
    output_dir: str = "",
    dpi: int = 300,
    bbox_inches: str = "tight",
    parse_timestamp: bool = False,
):
    """Saves figure to disk.

    Args:
        figure (plt.Figure): Matplotlib figure object.
        timestamp (pd.Timestamp): Date or time of data collection.
        suffix (str): Additional string to add to file name.
        img_format (str): Image file format. Default SVG.
        output_dir (str): Location to save images.
            Default "./reports/figures/".
    """

    if not output_dir:
        pass
    elif not os.path.isdir(output_dir):
        try:
            os.mkdir(output_dir)
        except OSError as error:
            print(error)

    if not parse_timestamp:
        plot_id = timestamp
    else:
        plot_id = timestamp.strftime("%Y%m%d")
        plot_id = re.sub(r"\W+", "", str(plot_id))

    if suffix:
        suffix = f"_{suffix}"

    figure.savefig(
        fname=f"{output_dir}{plot_id}{suffix}",
        format=img_format,
        dpi=dpi,
        bbox_inches=bbox_inches,
    )
    plt.close()


def set_delta_annotation(data: pd.DataFrame, idx: int, label: str) -> dict:
    data_values = data.values.flatten()

    annotation = {
        "text": f"{label}: {data_values[-1]-data_values[0]:.6f}",
        "x": data.time[idx],
        "y": data.values[idx],
    }
    plt.text(x=annotation["x"], y=annotation["y"], s=annotation["text"])
    return annotation


def set_annotation(
    axis,
    data: pd.DataFrame,
    idx: int,
    label: str,
    size: float = 20,
    weight: str = None,
) -> dict:

    if isinstance(idx, str):
        if idx == "min":
            idx = data.values.argmin()
        elif idx == "max":
            idx = data.values.argmax()
        else:
            raise ValueError("`idx` takes integer, 'min', or 'max'.")

    y_val = data.values[idx]
    ymin, ymax = axis.get_ylim()
    offset = max(abs(ymax), abs(ymin)) / 10
    if data.values[idx + 1] > y_val:
        offset = -offset

    annotation = {
        "text": f"{label}",
        "x": data.index[idx],
        "y": y_val,
        "size": size,
        "weight": weight,
    }
    axis.annotate(
        annotation["text"],
        (annotation["x"], annotation["y"]),
        size=annotation["size"],
        weight=annotation["weight"],
        transform=axis.transAxes,
        textcoords="offset points",
        xytext=(np.abs(offset), offset),
    )
    return annotation


def get_obs_heights(
    filename: str, var: str, period: list = None, offset: float = False
) -> pd.DataFrame:
    df = pd.read_csv(filename, index_col="TIMESTAMP", parse_dates=True)
    if period:
        df = df[period[0] : period[1]]
    if offset:
        start_point = df[var].iloc[0]
        data_values = start_point - df[var]
        height_drop = data_values.iloc[-1] - data_values.iloc[0]
    else:
        data_values = df[var]
        height_drop = data_values.iloc[-1]
    plt.figure()
    plt.plot(
        df.index,
        data_values,
        color="black",
        label=f"{var} [Ref]",
    )
    ax = plt.gca()
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
    )

    plt.text(
        df.index[0],
        data_values.iloc[0],
        f"Obs: {height_drop:.6f}",
    )
    output_name = get_output_path(
        var_name=var, start_time=period[0], end_time=period[1]
    )
    plt.savefig(f"obs_{output_name}")

    return df


def get_output_path(
    var_name: str, start_time: str = None, end_time: str = None
):
    if start_time and end_time:
        datestamp = f"{start_time.replace('-','')}-{end_time.replace('-','')}_"
    else:
        datestamp = ""
    output_name = f"{datestamp}{var_name}"

    if not var_name:
        raise ValueError("Must supply a variable name.")
    elif not output_name:
        raise ValueError("Output path is an empty string.")

    return output_name


def set_output_path(output_path: str = None, **kwargs) -> str:
    """Set output path for image.

    Args:
        output_path (str): Output file path. Default None.

    Keyword Args:
        var_name (str): Variable name.
        start_time (str): Start of timestamp.
        end_time (str): End of timestamp.

    Returns: File path of output image.
    """
    var_name = kwargs.get("var_name")
    start_time = kwargs.get("start_time")
    end_time = kwargs.get("end_time")

    if not output_path and not var_name:
        raise ValueError("Must supply an output path or variable name.")
    elif not output_path:
        output_path = get_output_path(
            var_name=var_name, start_time=start_time, end_time=end_time
        )

    return output_path


def find_nearest(array, value):
    array = np.asarray(array)
    idx = np.nanargmin(np.abs(array - value))
    return (idx, array[idx])


def find_nearest_2d(array, values):
    array = np.asarray(array)

    # the last dim must be 1 to broadcast in (array - values) below.
    values = np.expand_dims(values, axis=-1)
    indices = np.nanargmin(np.abs(array - values), axis=-1)
    dist = np.nanmin(np.abs(array - values), axis=-1)

    return indices, dist


def naive_fast(latvar, lonvar, lat0, lon0):
    # Read latitude and longitude from file into numpy arrays
    latvals = latvar[:]
    lonvals = lonvar[:]
    dist_sq = (latvals - lat0) ** 2 + (lonvals - lon0) ** 2
    minindex_flattened = dist_sq.argmin()  # 1D index of min element
    iy_min, ix_min = np.unravel_index(minindex_flattened, latvals.shape)
    return iy_min, ix_min


def get_datetime_hours(date_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.to_datetime(date_index, format="%H", utc=True)


def get_spatiotemporal_domain(
    data: xr.DataArray | pd.DataFrame,
    start: str = None,
    end: str = None,
    latitude: str = None,
    longitude: str = None,
) -> xr.DataArray | pd.DataFrame:
    if (start is not None) & (end is not None):
        if not isinstance(data, pd.DataFrame):
            data = data.sel(time=slice(start, end))
        else:
            data = data[start:end]

    # Select location
    if not isinstance(data, pd.DataFrame):
        if (latitude is not None) & (longitude is not None):
            data = data.sel(lat=latitude, lon=longitude, method="nearest")
        else:
            data = data.drop_vars(("lat", "lon"))
            # data = data.squeeze(("lat", "lon"))
        # if (latitude is None) & (longitude is None):
        #     latitude = data.lat.values[0]
        #     longitude = data.lon.values[0]
        # data = data.sel(lat=latitude, lon=longitude, method="nearest")

    return data


def get_squeezed_data(
    data: xr.Dataset, name: str, latitude: str = None, longitude: str = None
):
    if (latitude is None) & (longitude is None):
        data = data[name].squeeze(["lat", "lon"])
    else:
        data = data[name][:, :].values

    return data


def read_data(filename: str):
    if filename[-4:] == ".csv":
        data = pd.read_csv(filename, index_col="TIMESTAMP", parse_dates=True)
    else:
        data = xr.open_dataset(filename)
    return data


def get_univariate_data(data, name: str):
    array = data[name]
    if isinstance(data, xr.DataArray):
        array = array.flatten()
    return array


def get_data_as_dataframe(data, name: str) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        dataframe = (
            get_univariate_data(data=data, name=name)
            .to_dataframe()
            .droplevel(["lat", "lon"])
        )
    else:
        dataframe = get_univariate_data(data=data, name=name)

    return dataframe


def get_parsed_data(data, name):
    is_dataframe = isinstance(data, pd.DataFrame)
    if name == "DELTAHEIGHT":
        if not is_dataframe:
            data = get_data_as_dataframe(data=data, name="TOTALHEIGHT")
            var_data = data - data.values[0]
        else:
            var_data = get_data_as_dataframe(data=data, name="HEIGHT")
    elif name == "TOTALMELT":
        surface_melt = get_data_as_dataframe(data=data, name="surfM")
        subsurface_melt = get_data_as_dataframe(data=data, name="subM")
        var_data = 1000 * surface_melt["surfM"].add(subsurface_melt["subM"])
        # var_data = subsurface_melt
    elif name == "PCPN":
        rain_data = get_data_as_dataframe(data=data, name="RRR")
        snowfall_data = get_data_as_dataframe(data=data, name="SNOWFALL")
        var_data = rain_data + snowfall_data.values
    else:
        var_data = get_data_as_dataframe(data=data, name=name)

    return var_data


def get_var_data(
    data,
    name: str,
    start: str = None,
    end: str = None,
    latitude=None,
    longitude=None,
):
    var_data = get_spatiotemporal_domain(
        data=data, start=start, end=end, latitude=latitude, longitude=longitude
    )

    var_data = get_parsed_data(data=var_data, name=name)

    return var_data


def get_linear_regression(x_data, y_data) -> tuple:
    model = sm.OLS(y_data, sm.add_constant(x_data))
    results = model.fit()
    print(results.params)
    print(results.summary())
    return model, results


def get_best_fit_line(axis, results, colour=None):
    intercept, slope = results.params
    print(results.rsquared_adj)
    r_squared = results.rsquared_adj
    # label = f"$y = ({slope:.3E})x {intercept:+.1e}$"
    label = f"$R^{2}: {r_squared:.3f}$"
    axis.axline(
        xy1=(0, intercept),
        slope=slope,
        label=label,
        color=colour,
        linewidth=5.0,
        in_layout=False,
    )

    return axis


def remove_outliers(data, method: str = "zscore", threshold: int = 3):
    if method == "zscore":
        zscore = (data - data.mean()) / data.std()
        mask = zscore < threshold
        data = data[mask]
    else:
        raise NotImplementedError(f"{method} is not implemented.")
    return data, mask


def round_to_base(x, base=5):
    return base * math.ceil(x / base)


def get_convert_units(data, name: str):
    if name in ["LWout", "LE"]:
        data = -data
    elif name in ["LAYER_T", "T2", "TS"]:
        data = data - 273.15
    elif "MB" in name or name in [
        "SNOWFALL",
        "REFREEZE",
        "CONDENSATION",
        "DEPOSITION",
        "SUBLIMATION",
        "EVAPORATION",
    ]:
        data = data * 1000
    elif name in ["surfM", "subM"]:
        data = data * -1000

    return data


def get_plot_data(
    data,
    name: str,
    start=None,
    end=None,
    latitude=None,
    longitude=None,
    convert_units: bool = True,
) -> pd.DataFrame:
    plot_data = get_var_data(
        data=data,
        name=name,
        start=start,
        end=end,
        latitude=latitude,
        longitude=longitude,
    )

    if convert_units:
        plot_data = get_convert_units(data=plot_data, name=name)

    return plot_data


def get_model_name(filename: str) -> str:
    if "_DEB_" in filename:
        model_name = "COSIPY-DEB"
    elif "_CI_" in filename:
        model_name = "COSIPY-CI"
    elif ".csv" in filename or "30m_qc" in filename:
        model_name = "Observations"
    else:
        model_name = filename.split("/")[-1][:-24]

    return model_name


def plot_1D_scatter(
    prediction: str,
    prediction_var: str,
    reference: str,
    reference_var: str,
    start: str = None,
    end: str = None,
    lat=None,
    lon=None,
):
    # Get dataset
    prediction_data = read_data(filename=prediction)
    reference_data = read_data(filename=reference)

    x_data = get_spatiotemporal_domain(
        data=prediction_data, start=start, end=end, latitude=lat, longitude=lon
    )
    y_data = get_spatiotemporal_domain(
        data=reference_data, start=start, end=end, latitude=lat, longitude=lon
    )

    x_data = (
        get_univariate_data(data=x_data, name=prediction_var)
        .to_dataframe()
        .droplevel(["lat", "lon"])
    )
    y_data = (
        get_univariate_data(data=y_data, name=reference_var)
        .resample("h")
        .mean()
    )
    max_bin = math.ceil(
        max(np.nanmax(x_data.values), np.nanmax(y_data.values))
    )
    bin_intervals = np.arange(0, max_bin, 0.1)
    x_data_bin = pd.cut(x_data[prediction_var], bins=bin_intervals)
    y_data_bin = pd.cut(y_data, bins=bin_intervals)
    # x_group = x_data_bin.groupby(prediction_var)
    # y_group = y_data_bin.groupby(reference_var)
    # print(x_group)
    # print(y_group)
    # x_count = [np.mean(n) for n in x_data_bin]
    # y_count = [np.mean(n) for n in y_data_bin]
    # print(x_count)
    # print(y_count)
    # y_count = [n if n > 0 else float('nan') for n in y_data_bin]

    initialise_formatting()

    plt.figure(figsize=(10, 10))
    # plt.plot(x_count, y_count)

    # fig, ax = plt.subplots(tight_layout=True)
    # hist = ax.hist2d(x_data[prediction_var], y_data, bins=bin_intervals,norm=matplotlib.colors.LogNorm())

    # plt.scatter(
    #     x_data,
    #     y_data,
    #     # marker=".",
    #     s=2.0,
    #     color="black",
    # )

    key_labels = [
        label_selector(prediction_var)[0],
        label_selector(reference_var)[0],
    ]
    title_name = merge_multiple_labels(labels=key_labels)
    title_string = f"Modelled and Observed {title_name}"
    title_plot(
        title=title_string, timestamp=f"{start} {end}", location="Suldenferner"
    )
    axes = plt.gca()
    set_xy_labels(ax=axes, name=reference_var)

    output_name = get_output_path(
        var_name=prediction_var, start_time=start, end_time=end
    )
    plt.savefig(output_name, format="png", bbox_inches="tight")


def plot_profile_1D(filename, pdate, d=None, lat=None, lon=None):
    """This creates a simple plot showing the 2D fields"""

    DATA = xr.open_dataset(filename)
    DATA = DATA.sel(time=pdate)

    if (lat is not None) & (lon is not None):
        DATA = DATA.sel(lat=lat, lon=lon, method="nearest")

    plt.figure(figsize=(5, 5))
    depth = np.append(0, np.cumsum(DATA.LAYER_HEIGHT.values))

    if (lat is None) & (lon is None):
        rho = np.append(DATA.LAYER_RHO[:, :, 0], DATA.LAYER_RHO.values)
        t = np.append(DATA.LAYER_T[:, :, 0], DATA.LAYER_T.values)
    else:
        rho = np.append(DATA.LAYER_RHO[0], DATA.LAYER_RHO.values)
        t = np.append(DATA.LAYER_T[0], DATA.LAYER_T.values)

    print("Date: %s" % (pdate))
    print(
        "T2: %.2f \t RH: %.2f \t U: %.2f \t G: %.2f"
        % (DATA.T2, DATA.RH2, DATA.U2, DATA.G)
    )
    if d is not None:
        for dmeas in d:
            # idx, val = find_nearest(depth,d)
            idx, val = find_nearest(depth, dmeas)
            print(
                "nearest depth: %.3f \t density: %.2f \t T: %.2f"
                % (val, rho[idx], t[idx])
            )

    plt.step(rho, depth)
    ax1 = plt.gca()
    ax1.invert_yaxis()
    ax1.set_ylabel("Depth [m]")
    ax1.tick_params(axis="x", labelcolor="blue")
    ax1.set_xlabel("Density [kg m^-3]", color="blue")
    ax2 = ax1.twiny()
    ax2.plot(t, depth, color="red")
    ax2.set_xlabel("Temperature [K]", color="red")
    ax2.tick_params(axis="x", labelcolor="red")
    plt.show()


if __name__ == "__main__":

    args = get_user_arguments()

    plot_1D_scatter(
        prediction=args.file,
        prediction_var=args.var,
        reference=args.reference,
        reference_var=args.ref_var,
        start=args.start,
        end=args.end,
        lat=args.lat,
        lon=args.lon,
    )
