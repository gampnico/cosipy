import matplotlib

matplotlib.use("TkAgg")
import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotting_utilities as pu


def get_cumulated_array(data, **kwargs):
    cum = data.clip(**kwargs)
    cum = np.cumsum(cum, axis=0)
    d = np.zeros(np.shape(data))
    d[1:] = cum[:-1]
    return d


def plot_stacked_bar(
    data,
    axis,
    names: dict,
    start: str = None,
    end: str = None,
    lat: str = None,
    lon: str = None,
    convert_units: bool = False,
    resample_period: str = "1D",
):

    # bar_width = datetime.timedelta(days=0.9)
    bottom_offset = None
    all_data = []
    colours = []
    labels = []
    for flux, colour in names.items():
        flux_data = pu.get_plot_data(
            data=data,
            name=flux,
            start=start,
            end=end,
            latitude=lat,
            longitude=lon,
            convert_units=convert_units,
        )
        flux_data_mean = flux_data.resample(
            resample_period, label="left"
        ).mean()
        all_data.append(flux_data_mean[flux])
        if bottom_offset is None:
            bottom_offset = np.zeros(flux_data_mean.shape[0])
        label_text = pu.label_selector(dependent=flux)
        colours.append(colour)
        labels.append(label_text)
    all_data = np.array(all_data)
    cumulated_data = get_cumulated_array(all_data, min=0)
    cumulated_data_neg = get_cumulated_array(all_data, max=0)
    row_mask = all_data < 0
    cumulated_data[row_mask] = cumulated_data_neg[row_mask]
    data_stack = cumulated_data
    data_shape = np.shape(all_data)
    for i in np.arange(0, data_shape[0]):
        initial_width = flux_data_mean.index[1] - flux_data_mean.index[0]
        axis.bar(
            flux_data_mean.index,
            all_data[i],
            bottom=data_stack[i],
            color=colours[i],
            label=labels[i][0],
            align="edge",
            width=[*np.diff(flux_data_mean.index), initial_width],
        )
        # axis.bar(
        #     flux_data_daily_mean.index,
        #     flux_data_daily_mean[flux],
        #     width=bar_width,
        #     align="edge",
        #     color=colour,
        #     label=label_text[0],
        #     bottom=bottom_offset,
        # )
        # bottom_offset += flux_data_daily_mean[flux]
    axis.set_xlim([flux_data_mean.index[0], flux_data_mean.index[-1]])
    axis.legend()
    return axis


def plot_1D_cumulative_fluxes(
    filename: str,
    start=None,
    end=None,
    lat=None,
    lon=None,
    resample_period: str = "1D",
    output_path: str = None,
):

    cmaps = pu.get_colourmap()
    energy_fluxes = {
        "B": cmaps["energy_fluxes"][0],
        "LE": cmaps["energy_fluxes"][1],
        "H": cmaps["energy_fluxes"][2],
        "QPS": cmaps["energy_fluxes"][3],
        "QRR": cmaps["energy_fluxes"][4],
    }
    mass_fluxes = {
        "SNOWFALL": cmaps["blues"][0],
        "REFREEZE": cmaps["blues"][1],
        "CONDENSATION": cmaps["blues"][2],
        "DEPOSITION": cmaps["blues"][3],
        "surfM": cmaps["reds"][0],
        "subM": cmaps["reds"][1],
        "EVAPORATION": cmaps["reds"][2],
        "SUBLIMATION": cmaps["reds"][3],
    }
    y_data = pu.read_data(filename=filename)

    # pu.initialise_formatting()
    pu.initialise_formatting(scale_pts=8)
    fig, axes = plt.subplots(figsize=(20, 20), nrows=2, ncols=1)

    plot_stacked_bar(
        data=y_data,
        axis=axes[0],
        names=energy_fluxes,
        start=start,
        end=end,
        lat=lat,
        lon=lon,
        convert_units=True,
        resample_period=resample_period,
    )
    line_plots = {}
    for name in ["SWnet", "LWin", "LWout", "T2", "TS"]:
        line_data = pu.get_plot_data(
            data=y_data,
            name=name,
            start=start,
            end=end,
            latitude=lat,
            longitude=lon,
            convert_units=False,
        )
        line_plots[name] = line_data.resample(
            resample_period, label="left"
        ).mean()
    line_plots["LWnet"] = (
        line_plots["LWin"]["LWin"] + line_plots["LWout"]["LWout"]
    )
    axes[0].plot(line_plots["LWnet"], color="black", linewidth=3.0)
    pu.set_annotation(
        axis=axes[0],
        data=line_plots["LWnet"],
        idx="min",
        label=r"Q$_{L}$",
        size=30,
        weight="bold",
    )
    # line_plots["DeltaT"] = line_plots["T2"]["T2"] - line_plots["TS"]["TS"]
    # axes[0].plot(line_plots["DeltaT"], color="black", linewidth=3.0)
    # pu.set_annotation(
    #     axis=axes[0],
    #     data=line_plots["DeltaT"],
    #     idx=0,
    #     label=r"$\Delta T$",
    #     size=30,
    #     weight="bold",
    # )

    axes[0].plot(line_plots["SWnet"], color="black", linewidth=3.0)
    pu.set_annotation(
        axis=axes[0],
        data=line_plots["SWnet"],
        idx="max",
        label=r"Q$_{S}$",
        size=30,
        weight="bold",
    )

    axes[0].plot()
    pu.set_timeseries_labels(
        axis=axes[0], label="energy_flux", shortname=False, show_ablation=False
    )

    plot_stacked_bar(
        data=y_data,
        axis=axes[1],
        names=mass_fluxes,
        start=start,
        end=end,
        lat=lat,
        lon=lon,
        convert_units=True,
        resample_period=resample_period,
    )
    # axes[1].set_xlim([y_data.index[0], y_data.index[-1]])

    pu.set_timeseries_labels(
        axis=axes[1], label="mass_flux", shortname=False, show_ablation=False
    )
    # date_fmt = mdates.DateFormatter(fmt="%b %Y")
    pu.format_date_ticks(axes=axes, fmt="%b\n%Y", minor_fmt="%b",bar=True)
    for axis in axes:
        # axis.xaxis.set_major_formatter(date_fmt)
        y_label = axis.get_ylabel()
        if resample_period[1:] == "ME":
            mean_type = "Monthly "
        elif resample_period[1:] == "D":
            mean_type = "Daily "
        else:
            mean_type = ""
        axis.set_ylabel(f"{mean_type}Mean {y_label}")
        axis.axhline(0, color="grey", linestyle="--")
        axis.set_xlabel(f"Month (2016)")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    if not output_path:
        output_path = pu.get_output_path(
            var_name=f"fluxes-bar-",
            start_time=start,
            end_time=end,
        )

    pu.save_figure(figure=fig, timestamp=output_path, img_format="svg")
    # pu.set_delta_annotation(data=data_ref, idx=0, label="Ref")
    # pu.set_delta_annotation(data=data, idx=0, label="Run")


def get_user_arguments():
    parser = pu.get_arg_parser()
    parser.add_argument(
        "--resample",
        dest="resample_period",
        default="1D",
        help="Resampling frequency",
        type=str,
    )

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":

    args = get_user_arguments()
    print(args)

    plot_1D_cumulative_fluxes(
        filename=args.file,
        start=args.start,
        end=args.end,
        lat=args.lat,
        lon=args.lon,
        resample_period=args.resample_period,
        output_path=args.output_path,
    )
