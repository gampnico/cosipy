import matplotlib

matplotlib.use("TkAgg")
import argparse
import math

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotting_utilities as pu


def plot_1D_meteo_series(
    filename, start=None, end=None, lat=None, lon=None, output_path: str = None
):
    variables = ["T2", "RH2", "U2", "G", "PCPN", "SNOWFALL"]
    pu.initialise_formatting(scale_pts=10)
    colours = pu.get_colours()
    fig, ax = plt.subplots(
        nrows=len(variables), ncols=1, sharex=True, figsize=(20, 25)
    )
    # fig.tight_layout()
    raw_data = pu.read_data(filename=filename)

    for i in range(len(ax)):
        data = pu.get_plot_data(
            data=raw_data,
            name=variables[i],
            start=start,
            end=end,
            latitude=lat,
            longitude=lon,
        )
        if variables[i] != "SNOWFALL":
            data_hourly_mean = data.resample("1h", label="left").mean()
            ax[i].plot(
                data_hourly_mean.index,
                data_hourly_mean,
                color=colours["dark_silver"],
                label=f"{variables[i]}",
            )
            data_daily_mean = data.resample("1D", label="left").mean()
        else:
            data_daily_mean = data.resample("1D", label="left").sum() / 100
        ax[i].plot(
            data_daily_mean.index,
            data_daily_mean,
            color="black",
            label=f"{variables[i]}",
        )

        if np.nanmin(data_hourly_mean) < 0:
            ax[i].axhline(0, color="grey", linestyle="--")
        ymin, ymax = ax[i].get_ylim()
        ymax = math.ceil(ymax + (ymax - ymin) / 5)
        if variables[i] == "RH2":
            ymax = 105
            ax[i].yaxis.set_major_formatter(
                matplotlib.ticker.PercentFormatter(decimals=0, symbol="%")
            )
        ax[i].set_ylim((min(ymin, 0), ymax))
        print(ymin, ymax)
        ax[i].yaxis.set_major_locator(
            plt.MaxNLocator(
                nbins=4,
                min_n_ticks=4,
                prune=None,
                integer=True,
                steps=[1, 2, 4, 5, 10],
            )
        )
        pu.set_timeseries_labels(
            axis=ax[i], label=variables[i], shortname=True, show_ablation=True
        )
        ax[i].set_title(pu.label_selector(variables[i])[0])

    plt.xlabel(f"Date")

    ax[0].xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax[0].xaxis.get_major_locator())
    )
    pu.format_date_ticks(axes=ax)

    pu.set_output_path(
        output_path=output_path,
        var_name="meteo",
        start_time=start,
        end_time=end,
    )
    pu.save_figure(figure=fig, timestamp=output_path, img_format="svg")
    # plt.savefig(output_name, format="png", bbox_inches="tight")


if __name__ == "__main__":

    args = pu.get_user_arguments()

    plot_1D_meteo_series(
        filename=args.file,
        start=args.start,
        end=args.end,
        lat=args.lat,
        lon=args.lon,
        output_path=args.output_path,
    )
