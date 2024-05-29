import matplotlib

matplotlib.use("TkAgg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import plotting_utilities as pu


def plot_1D_model_comparison(
    prediction: list,
    prediction_var,
    reference: str,
    reference_var,
    start=None,
    end=None,
    lat=None,
    lon=None,
    output_path: str = None,
):

    x_data = pu.read_data(filename=reference)
    x_data = pu.get_plot_data(
        data=x_data,
        name=reference_var,
        start=start,
        end=end,
        latitude=lat,
        longitude=lon,
    )
    x_data_daily_mean = x_data.resample("1D").mean()

    y_data = {}
    for simulation in prediction:
        model_name = pu.get_model_name(filename=simulation)
        raw_data = pu.read_data(filename=simulation)

        raw_data = pu.get_plot_data(
            data=raw_data,
            name=prediction_var,
            start=start,
            end=end,
            latitude=lat,
            longitude=lon,
        )

        data_daily_mean = raw_data.resample("1D").mean()
        y_data[model_name] = data_daily_mean

    pu.initialise_formatting(scale_pts=10)
    colours = pu.get_colours()
    fig = plt.figure(figsize=(20, 10))
    ax = plt.gca()
    plt.plot(
        x_data_daily_mean.index,
        x_data_daily_mean,
        color="black",
        label=f"Observations",
        linewidth=2.0,
    )

    model_colours = {"COSIPY-DEB": "debris", "COSIPY-CI": "clean_ice"}
    for key, simulation in y_data.items():
        if key in model_colours.keys():
            line_colour = colours[model_colours[key]]
        else:
            line_colour = None
        plt.plot(
            simulation.index,
            simulation,
            color=line_colour,
            label=key,
            linewidth=2.0,
        )

        if np.nanmin(simulation) < 0:
            ax.axhline(0, color="grey", linestyle="--")

    plt.legend()

    _, y_label = pu.set_timeseries_labels(
        axis=ax,
        label=prediction_var,
        shortname=False,
        show_ablation=True,
        format_dates=False,
    )
    pu.format_y_ticks(axis=ax, label=y_label, base=1)
    pu.format_date_ticks(axes=[ax], fmt="%B", minor_fmt="")
    ax.spines[["right", "top"]].set_visible(False)
    # ax.set_ylim(0, 2.5)

    if not output_path:
        output_path = pu.get_output_path(
            var_name=f"model-comp-{prediction_var}",
            start_time=start,
            end_time=end,
        )

    pu.save_figure(figure=fig, timestamp=output_path, img_format="svg")
    # pu.set_delta_annotation(data=data_ref, idx=0, label="Ref")
    # pu.set_delta_annotation(data=data, idx=0, label="Run")


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

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":

    args = get_user_arguments()
    print(args)

    plot_1D_model_comparison(
        prediction=args.model_paths,
        prediction_var=args.var,
        reference=args.reference,
        reference_var=args.ref_var,
        start=args.start,
        end=args.end,
        lat=args.lat,
        lon=args.lon,
        output_path=args.output_path,
    )
