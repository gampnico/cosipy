import matplotlib
import matplotlib.ticker

matplotlib.use("TkAgg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
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


def plot_1D_diurnal(
    prediction: list,
    prediction_var: str,
    reference: str,
    reference_var: str,
    start: str = None,
    end: str = None,
    lat=None,
    lon=None,
    twin: bool = False,
    convert_units: bool = False,
    output_path: str = None,
):
    reference_data = pu.read_data(filename=reference)

    x_data = get_diurnal_mean(
        data=reference_data,
        name=reference_var,
        start=start,
        end=end,
        latitude=lat,
        longitude=lon,
        convert_units=convert_units,
    )
    x_name = pu.get_model_name(filename=reference)

    y_data = {}
    for simulation in prediction:
        model_name = pu.get_model_name(filename=simulation)
        raw_data = pu.read_data(filename=simulation)
        diurnal_mean = get_diurnal_mean(
            data=raw_data,
            name=prediction_var,
            start=start,
            end=end,
            latitude=lat,
            longitude=lon,
            convert_units=convert_units,
        )
        y_data[model_name] = diurnal_mean
    if x_name in y_data.keys():
        x_name = "Observations"
    pu.initialise_formatting(scale_pts=10)
    colours = pu.get_colours()
    legend_artists = []
    if not twin:
        fig = plt.figure(figsize=(20, 20))
        axes = plt.gca()
        artist = axes.plot(
            pu.get_datetime_hours(x_data.index),
            x_data,
            label=x_name,
            color="black",
        )
    else:
        fig, axes = plt.subplots(figsize=(20, 20))
        ax_ref = axes.twinx()
        ax_colour = colours["tree_green"]
        ax_colour = "firebrick"
        artist = ax_ref.plot(
            pu.get_datetime_hours(x_data.index),
            x_data,
            label=x_name,
            color=ax_colour,
            linestyle="--",
            linewidth=3.0,
        )
        label_text = pu.label_selector(dependent=reference_var)
        y_label = pu.merge_label_with_unit(label=label_text)
        ax_ref.set_ylabel(f"{y_label}", color=ax_colour)
        ax_ref.tick_params(axis="y", labelcolor=ax_colour)
        # ax_ref.set_ylabel(f"{y_label}", color="black")
        # ax_ref.tick_params(axis="y", labelcolor="black")

    legend_artists.append(artist)
    model_colours = {"COSIPY-DEB": "debris", "COSIPY-CI": "clean_ice"}
    for key, simulation in y_data.items():
        if key in model_colours.keys():
            line_colour = colours[model_colours[key]]
        else:
            line_colour = None
        artist = axes.plot(
            pu.get_datetime_hours(simulation.index),
            simulation,
            color=line_colour,
            # color="black",
            label=key,
            linewidth=3.0,
        )
        legend_artists.append(artist)

    axes.axhline(0, color="grey", linestyle="--")
    # axes.xaxis.set_major_formatter(
    #     mdates.ConciseDateFormatter(axes.xaxis.get_major_locator())
    # )
    axes.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3)))
    axes.xaxis.set_major_formatter(mdates.DateFormatter(fmt="%H:%M", tz="UTC"))
    axes.xaxis.set_minor_locator(mdates.HourLocator())
    label_text = pu.label_selector(dependent="hour")
    x_label = pu.merge_label_with_unit(label=label_text)
    label_text = pu.label_selector(dependent=prediction_var)
    y_label = pu.merge_label_with_unit(label=label_text)

    axes.set_xlabel(f"{x_label}")
    axes.set_ylabel(f"{y_label}")
    # axes.legend(handles=legend_artists)
    fig.legend(bbox_to_anchor=(0.9, 0.88))

    if not output_path:
        output_path = pu.get_output_path(
            var_name=f"diurnal-{prediction_var}",
            start_time=start,
            end_time=end,
        )
    pu.save_figure(figure=fig, timestamp=output_path, img_format="svg")


def get_user_arguments():
    parser = pu.get_arg_parser()
    parser.add_argument(
        "-i",
        "--models",
        dest="model_paths",
        nargs="+",
        default=None,
        help="Simulation outputs",
        type=str,
    )
    parser.add_argument(
        "--twin_axes",
        dest="twin_axes",
        action="store_true",
        default=False,
        help="Plot reference on twin axis",
    )

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":

    args = get_user_arguments()
    print(args)

    plot_1D_diurnal(
        prediction=args.model_paths,
        prediction_var=args.var,
        reference=args.reference,
        reference_var=args.ref_var,
        start=args.start,
        end=args.end,
        lat=args.lat,
        lon=args.lon,
        twin=args.twin_axes,
        output_path=args.output_path,
        convert_units=args.convert_units,
    )
