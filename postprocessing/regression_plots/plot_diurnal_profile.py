import matplotlib

matplotlib.use("TkAgg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotting_utilities as pu
import xarray as xr


def get_diurnal_mean(
    data: xr.Dataset,
    name: str,
    start: str = None,
    end: str = None,
    latitude: str = None,
    longitude: str = None,
    convert_units: bool = False,
) -> pd.DataFrame:
    data = pu.get_spatiotemporal_domain(
        data=data, start=start, end=end, latitude=latitude, longitude=longitude
    )

    if isinstance(data, pd.DataFrame):
        data = data.groupby(data.index.hour).mean()
        diurnal = data[name]

    else:
        data = data.groupby("time.hour").mean()
        diurnal = pu.get_parsed_data(data=data, name=name)

    if convert_units:
        diurnal = pu.get_convert_units(data=diurnal, name=name)

    return diurnal


def plot_1D_diurnal_profile(
    prediction: str,
    prediction_var: str,
    timestamps: list,
    dividing_var: str,
    start: str = None,
    end: str = None,
    lat=None,
    lon=None,
    convert_units: bool = False,
    output_path: str = None,
):
    model_name = pu.get_model_name(filename=prediction)
    raw_data = pu.read_data(filename=prediction)

    diurnal_mean = get_diurnal_mean(
        data=raw_data,
        name=prediction_var,
        start=start,
        end=end,
        latitude=lat,
        longitude=lon,
        convert_units=convert_units,
    )

    profile_data = diurnal_mean[prediction_var]
    print(profile_data.keys())

    pu.initialise_formatting(scale_pts=10)
    colours = pu.get_colours()
    legend_artists = []
    fig = plt.figure(figsize=(20, 20))
    axes = plt.gca()

    y_data = {}
    for time_key in timestamps:
        y_data[time_key] = profile_data[int(time_key)]
        print(y_data[time_key])
        artist = axes.plot(
            y_data[time_key],
            y_data[time_key].index,
            label=f"{time_key:01}:00",
        )

        if np.nanmin(y_data[time_key]) < 0:
            axes.axvline(0, color="grey", linestyle="--")
    axes.invert_yaxis()
    label_text = pu.label_selector(dependent=f"{prediction_var[6:]}")
    x_label = pu.merge_label_with_unit(label=label_text)
    label_text = pu.label_selector(dependent="hour")
    y_label = pu.merge_label_with_unit(label=label_text)

    axes.set_xlabel(f"{x_label}")
    axes.set_ylabel(f"{y_label}")
    plt.legend()

    if not output_path:
        output_path = pu.get_output_path(
            var_name=f"diurnal-profile-{prediction_var}",
            start_time=start,
            end_time=end,
        )
    pu.save_figure(figure=fig, timestamp=output_path, img_format="svg")


def get_user_arguments():
    parser = pu.get_arg_parser()
    parser.add_argument(
        "-t",
        "--times",
        dest="timestamps",
        nargs="+",
        default=None,
        help="Profile timestamps",
        type=str,
    )
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
    print(args)

    plot_1D_diurnal_profile(
        prediction=args.file,
        prediction_var=args.var,
        timestamps=args.timestamps,
        dividing_var=args.split_var,
        start=args.start,
        end=args.end,
        lat=args.lat,
        lon=args.lon,
        output_path=args.output_path,
        convert_units=args.convert_units,
    )
