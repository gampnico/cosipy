import matplotlib

matplotlib.use("TkAgg")
import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotting_utilities as pu


def plot_1D_instabilities(
    filename: str,
    prediction_var: str,
    start=None,
    end=None,
    lat=None,
    lon=None,
    resample_period: str = "1D",
    output_path: str = None,
):
    data = pu.read_data(filename=filename)
    y_data = pu.get_plot_data(
        data=data,
        name=prediction_var,
        start=start,
        end=end,
        latitude=lat,
        longitude=lon,
        convert_units=False,
    )
    rolling_sd = y_data.rolling(window=datetime.timedelta(hours=24)).std()

    pu.initialise_formatting(scale_pts=12)
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(20, 15))
    # fig.tight_layout()

    axes[0].plot(y_data.index, y_data, color="black")
    ylabel, _ = pu.set_timeseries_labels(
        axis=axes[0], label=prediction_var, shortname=False, show_ablation=True
    )
    axes[1].plot(y_data.index, rolling_sd, color="black")
    axes[1].set_ylabel(f"Moving Standard Deviation, [{ylabel[2]}]")
    pu.plot_ablation_season(axis=axes[1])
    ymax = pu.round_to_base(x=np.nanmax(rolling_sd.values), base=5)
    axes[1].set_ylim((0, ymax))

    pu.format_date_ticks(axes=axes,fmt="%b\n%Y",minor_fmt="%b")
    plt.xlabel(f"Date")

    if not output_path:
        output_path = pu.get_output_path(
            var_name=f"instabilities", start_time=start, end_time=end
        )

    pu.save_figure(figure=fig, timestamp=output_path, img_format="svg")


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

    plot_1D_instabilities(
        filename=args.file,
        prediction_var=args.var,
        start=args.start,
        end=args.end,
        lat=args.lat,
        lon=args.lon,
        resample_period=args.resample_period,
        output_path=args.output_path,
    )
