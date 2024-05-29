import matplotlib

matplotlib.use("TkAgg")
import argparse
import os

import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm
from scipy import interpolate
from scipy.interpolate import griddata


def plot_profile(filename, pdate, lat, lon):
    """This creates a simple plot showing the 2D fields"""
    DATA = xr.open_dataset(filename)

    (c_y, c_x) = naive_fast(DATA.lat.values, DATA.lon.values, lat, lon)
    DATA = DATA.sel(time=pdate, west_east=c_x, south_north=c_y)

    plt.figure(figsize=(20, 12))

    depth = np.append(0, np.cumsum(DATA.LAYER_HEIGHT.values))
    rho = np.append(DATA.LAYER_RHO[0], DATA.LAYER_RHO.values)
    plt.step(rho, depth)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.show()


def plot_1D_timeseries(
    filename, var, start=None, end=None, lat=None, lon=None
):
    # Get dataset
    ds = xr.open_dataset(filename)

    if (start is not None) & (end is not None):
        ds = ds.sel(time=slice(start, end))

    # Select location
    if (lat is not None) & (lon is not None):
        ds = ds.sel(lat=lat, lon=lon, method="nearest")

    data = ds[var]

    plt.figure(figsize=(15, 8))
    plt.plot(data.time, data.values.flatten(), color="black", label=var)
    plt.legend()
    ax = plt.gca()
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
    )

    set_delta_annotation(data=data, idx=0, label="Delta")
    plt.savefig("test.png")


def set_delta_annotation(data: pd.DataFrame, idx: int, label: str):
    data_values = data.values.flatten()
    plt.text(
        data.time[idx],
        data.values[idx],
        f"{label}: {data_values[-1]-data_values[0]:.6f}",
    )


def get_obs_heights(filename, var, period=None, offset=False):
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


def plot_1D_timeseries_comparison(
    filename,
    var,
    reference,
    reference_var,
    start=None,
    end=None,
    lat=None,
    lon=None,
):
    # Get dataset
    ds = xr.open_dataset(filename)
    ds_ref = xr.open_dataset(reference)
    #    get_obs_heights(filename="../../data/input/Suldenferner/Suldenferner_aws_qc_2015.csv", var="TCDT")
    get_obs_heights(
        filename="../../data/input/Suldenferner/Suldenferner_aws_2016_surface_height.csv",
        var="HEIGHT",
        period=[start, end],
    )
    get_obs_heights(
        filename="../../data/input/Suldenferner/Suldenferner_aws_2016_30m_qc.csv",
        var="SNOW",
        period=[start, end],
    )
    if (start is not None) & (end is not None):
        ds = ds.sel(time=slice(start, end))
        ds_ref = ds_ref.sel(time=slice(start, end))

    # Select location
    if (lat is not None) & (lon is not None):
        ds = ds.sel(lat=lat, lon=lon, method="nearest")
        ds_ref = ds_ref.sel(lat=lat, lon=lon, method="nearest")

    data = ds[var]
    data_ref = ds_ref[reference_var]
    data_values = data.values.flatten()
    data_ref_values = data_ref.values.flatten()

    plt.figure()
    plt.plot(
        data_ref.time,
        data_ref_values,
        color="black",
        label=f"{reference_var} [Ref]",
    )
    plt.plot(data.time, data_values, color="firebrick", label=var)

    plt.legend()
    ax = plt.gca()
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
    )
    set_delta_annotation(data=data_ref, idx=0, label="Ref")
    set_delta_annotation(data=data, idx=0, label="Run")

    output_name = get_output_path(var_name=var, start_time=start, end_time=end)
    plt.savefig(output_name)


def get_output_path(
    var_name: str, start_time: str = None, end_time: str = None
):
    if start_time and end_time:
        datestamp = f"{start_time.replace('-','')}-{end_time.replace('-','')}_"
    else:
        datestamp = ""
    output_name = f"{datestamp}{var_name}.png"
    return output_name


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
):
    # Get dataset
    ds = xr.open_dataset(filename)

    if (start is not None) & (end is not None):
        ds = ds.sel(time=slice(start, end))

    # Select location
    if (lat is not None) & (lon is not None):
        ds = ds.sel(lat=lat, lon=lon, method="nearest")

    # Get first layer height
    fl = ds.attrs["First_layer_height_log_profile"]

    # Get data
    if var == "T":
        if (lat is None) & (lon is None):
            V = ds.LAYER_T[:, 0, 0, :].values
        else:
            V = ds.LAYER_T[:, :].values
        cmap = plt.get_cmap("YlGnBu_r")
        barLabel = "Temperature [K]"
    if var == "RHO":
        if (lat is None) & (lon is None):
            V = ds.LAYER_RHO[:, 0, 0, :].values
        else:
            V = ds.LAYER_RHO[:, :].values
        cmap = plt.get_cmap("YlGnBu_r")
        barLabel = "Density [kg m^-3]"
    if var == "IF":
        if (lat is None) & (lon is None):
            V = ds.LAYER_ICE_FRACTION[:, 0, 0, :].values
        else:
            V = ds.LAYER_ICE_FRACTION[:, :].values
        cmap = plt.get_cmap("YlGnBu_r")
        barLabel = "Ice fraction [-]"
    if var == "REF":
        if (lat is None) & (lon is None):
            V = ds.LAYER_REFREEZE[:, 0, 0, :].values
        else:
            V = ds.LAYER_REFREEZE[:, :].values
        cmap = plt.get_cmap("YlGnBu_r")
        barLabel = "Refreezing [m w.e.]"
    if var == "LWC":
        if (lat is None) & (lon is None):
            V = ds.LAYER_LWC[:, 0, 0, :].values
        else:
            V = ds.LAYER_LWC[:, :].values
        cmap = plt.get_cmap("YlGnBu_r")
        barLabel = "Liquid Water Content [-]"
    if var == "POR":
        if (lat is None) & (lon is None):
            V = ds.LAYER_POROSITY[:, 0, 0, :].values
        else:
            V = ds.LAYER_POROSITY[:, :].values
        cmap = plt.get_cmap("YlGnBu_r")
        barLabel = "Air Porosity [-]"
    if var == "DEPTH":
        if (lat is None) & (lon is None):
            V = ds.LAYER_HEIGHT[:, 0, 0, :].values.cumsum(axis=1)
        else:
            V = ds.LAYER_HEIGHT[:, :].values.cumsum(axis=1)
        cmap = plt.get_cmap("YlGnBu_r")
        barLabel = "Depth [m]"
    if var == "HEIGHT":
        if (lat is None) & (lon is None):
            V = ds.LAYER_HEIGHT[:, 0, 0, :].values
        else:
            V = ds.LAYER_HEIGHT[:, :].values
        cmap = plt.get_cmap("YlGnBu_r")
        barLabel = "Layer height [m]"

    if (lat is None) & (lon is None):
        D = ds.LAYER_HEIGHT[:, 0, 0, :].values.cumsum(axis=1)
    else:
        D = ds.LAYER_HEIGHT[:, :].values.cumsum(axis=1)

    # Get dimensions
    time = np.arange(ds.dims["time"])

    if (lat is None) & (lon is None):
        depth = ds.SNOWHEIGHT[:, 0, 0].values
    else:
        depth = ds.SNOWHEIGHT[:].values

    # Calc plotting domain height
    Dn = np.int(np.floor(ds.SNOWHEIGHT.max())) + 1

    ## Create new grid
    xi = time
    if domainy is None:
        domainy = 0.0

    yi = np.arange(domainy, Dn, fl)
    X, Y = np.meshgrid(xi, yi)
    data = np.full_like(X, np.nan, dtype=np.double)

    # Re-calc depth data top=zero
    D = -(D.transpose() - depth).transpose()

    def find_nearest(array, values):
        array = np.asarray(array)

        # the last dim must be 1 to broadcast in (array - values) below.
        values = np.expand_dims(values, axis=-1)

        indices = np.nanargmin(np.abs(array - values), axis=-1)
        dist = np.nanmin(np.abs(array - values), axis=-1)

        return indices, dist

    for i in range(len(xi)):
        sel = np.where(yi < depth[i])
        idx, dist = find_nearest(D[i, :], yi[sel])
        data[sel, i] = V[i, idx]

    fig, ax = plt.subplots(figsize=(20, 10))
    CS = ax.pcolormesh(X, Y, data, cmap=cmap)
    # CS = ax.pcolormesh(X,Y,data, vmin=0, vmax=0.1)

    N = pd.date_range(ds.time[0].values, ds.time[-1].values, freq="m")
    M = pd.date_range(ds.time[0].values, ds.time[-1].values, freq="H")

    if (stake_file != None) & (pit_name != None):
        df = pd.read_csv(stake_file, sep="\t", index_col="TIMESTAMP")
        df = df[df[pit_name] != -9999]

        for index, row in df.iterrows():
            res = (M == pd.Timestamp(index)).argmax()
            if res != 0:
                plt.scatter(res, row[pit_name])

    labIdx = []
    label = []
    for q in N:
        o = np.where(M == q)
        labIdx.append(o[0][0])
        label.append(q.strftime("%Y-%m-%d"))

    plt.xticks(labIdx, label, rotation=45, fontsize=16, weight="normal")
    plt.ylabel("Depth [m]", fontsize=16, weight="normal")
    plt.xlabel("Date", fontsize=16, weight="normal")
    plt.title(var + "-Profile", fontsize=16, weight="normal")

    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = plt.colorbar(CS)
    cbar.ax.set_ylabel(
        barLabel, fontsize=16, fontname="Helvetica", weight="normal"
    )
    cbar.ax.set_yticks(barLabel, fontname="Helvetica", weight="normal")

    plt.show()


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


def find_nearest(array, value):
    array = np.asarray(array)
    idx = np.nanargmin(np.abs(array - value))
    return (idx, array[idx])


def naive_fast(latvar, lonvar, lat0, lon0):
    # Read latitude and longitude from file into numpy arrays
    latvals = latvar[:]
    lonvals = lonvar[:]
    dist_sq = (latvals - lat0) ** 2 + (lonvals - lon0) ** 2
    minindex_flattened = dist_sq.argmin()  # 1D index of min element
    iy_min, ix_min = np.unravel_index(minindex_flattened, latvals.shape)
    return iy_min, ix_min


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quick plot of the results file."
    )
    parser.add_argument(
        "-f", "--file", dest="file", help="Path to the result file"
    )
    parser.add_argument(
        "-r", "--reference", dest="reference", help="Path to observation data"
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

    args = parser.parse_args()

    if args.reference:
        plot_1D_timeseries_comparison(
            filename=args.file,
            var=args.var,
            reference=args.reference,
            reference_var=args.ref_var,
            start=args.start,
            end=args.end,
            lat=args.lat,
            lon=args.lon,
        )
    else:
        plot_1D_timeseries(
            filename=args.file,
            var=args.var,
            start=args.start,
            end=args.end,
            lat=args.lat,
            lon=args.lon,
        )
