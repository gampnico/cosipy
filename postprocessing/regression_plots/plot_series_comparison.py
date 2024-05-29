import matplotlib

matplotlib.use("TkAgg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import plotting_utilities as pu


def plot_1D_timeseries_comparison(
    prediction,
    prediction_var,
    reference,
    reference_var,
    start=None,
    end=None,
    lat=None,
    lon=None,
    output_path: str = None,
):
    x_data = pu.read_data(filename=reference)
    y_data = pu.read_data(filename=prediction)
    x_data = pu.get_plot_data(
        data=x_data,
        name=reference_var,
        start=start,
        end=end,
        latitude=lat,
        longitude=lon,
    )
    y_data = pu.get_plot_data(
        data=y_data,
        name=prediction_var,
        start=start,
        end=end,
        latitude=lat,
        longitude=lon,
    )

    colours = pu.initialise_formatting()
    fig = plt.figure(figsize=(20, 10))
    plt.plot(
        x_data.index,
        x_data,
        color="black",
        label=f"{reference_var} [Ref]",
    )
    plt.plot(
        y_data.index, y_data, color=colours["debris"], label=prediction_var
    )

    plt.legend()
    ax = plt.gca()
    ax = pu.format_date_ticks(axes=[ax], fmt="%B\n%Y", minor_fmt="%B")
    # ax.xaxis.set_major_formatter(
    #     mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
    # )

    if not output_path:
        output_path = pu.get_output_path(
            var_name=f"series-comp-{prediction_var}",
            start_time=start,
            end_time=end,
        )

    pu.save_figure(figure=fig, timestamp=output_path, img_format="svg")
    # pu.set_delta_annotation(data=data_ref, idx=0, label="Ref")
    # pu.set_delta_annotation(data=data, idx=0, label="Run")


if __name__ == "__main__":

    args = pu.get_user_arguments()

    plot_1D_timeseries_comparison(
        prediction=args.file,
        prediction_var=args.var,
        reference=args.reference,
        reference_var=args.ref_var,
        start=args.start,
        end=args.end,
        lat=args.lat,
        lon=args.lon,
        output_path=args.output_path,
    )
