import matplotlib

matplotlib.use("TkAgg")
import argparse
import os

import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.interpolate as scinterp
import xarray as xr
from matplotlib.colors import BoundaryNorm


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

    plt.rc("font", size=small)  # controls default text sizes
    plt.rc("axes", titlesize=large)  # fontsize of the axes title
    plt.rc("axes", labelsize=medium)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=small)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=small)  # fontsize of the tick labels
    plt.rc("legend", fontsize=small)  # legend fontsize
    plt.rc("figure", titlesize=extra)  # fontsize of the figure title
    # plt.rc("text", usetex=True)


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
    output_path: str = None,
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

    # assert ds.SNOWHEIGHT.max() == 0.0
    if ds.SNOWHEIGHT.max() > 0.0:
        if (lat is None) & (lon is None):
            depth = ds.SNOWHEIGHT[:, 0, 0].values
        else:
            depth = ds.SNOWHEIGHT[:].values

        # Calc plotting domain height
        Dn = int(np.floor(ds.SNOWHEIGHT.max())) + 1
    else:
        if (lat is None) & (lon is None):
            depth = ds.TOTALHEIGHT[:, 0, 0].values / 100
        else:
            depth = ds.TOTALHEIGHT[:].values / 100

        # Calc plotting domain height
        Dn = int(np.floor(ds.TOTALHEIGHT.max() / 100)) + 1

    print(Dn)

    ## Create new grid
    xi = time
    if domainy is None:
        domainy = 0.0

    yi = np.arange(domainy, Dn, fl)
    X, Y = np.meshgrid(xi, yi)
    data = np.full_like(X, np.nan, dtype=np.double)

    # Re-calc depth data top=zero
    D = -(D.transpose() - depth).transpose()

    for i in range(len(xi)):
        sel = np.where(yi < depth[i])
        idx, dist = find_nearest(D[i, :], yi[sel])
        data[sel, i] = V[i, idx]

    fig, ax = plt.subplots(figsize=(20, 10))
    CS = ax.pcolormesh(X, Y, data, cmap=cmap)
    # CS = ax.pcolormesh(X,Y,data, vmin=0, vmax=0.1)

    N = pd.date_range(ds.time.values[0], ds.time.values[-1], freq="ME")
    M = pd.date_range(ds.time[0].values, ds.time[-1].values, freq="h")

    if (stake_file != None) & (pit_name != None):
        df = pd.read_csv(stake_file, sep="\t", index_col="TIMESTAMP")
        df = df[df[pit_name] != -9999]

        for index, row in df.iterrows():
            res = (M == pd.Timestamp(index)).argmax()
            if res != 0:
                plt.scatter(res, row[pit_name])

    label = []
    for times in np.array(ds.time.values):
        label.append(np.datetime_as_string(times, unit="m")[5:-3])

    # assert labIdx
    plt.xticks(
        ticks=np.arange(0, ds.time.shape[0], 3),
        labels=label[::3],
        fontsize=8,
        rotation=45,
        weight="normal",
    )
    plt.ylabel("Depth [m]", fontsize=16, weight="normal")
    plt.xlabel("Date", fontsize=16, weight="normal")
    plt.title(var + "-Profile", fontsize=16, weight="normal")
    ax = plt.gca()
    ax.invert_yaxis()
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = plt.colorbar(CS)
    cbar.ax.set_ylabel(barLabel, fontsize=16)

    # cbar.ax.set_yticks(barLabel)

    plt.show()


def get_output_path(
    var_name: str, start_time: str = None, end_time: str = None
):
    if start_time and end_time:
        datestamp = f"{start_time.replace('-','')}-{end_time.replace('-','')}_"
    else:
        datestamp = ""
    output_name = f"{datestamp}{var_name}.png"
    return output_name


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
        barLabel = f"{data[name].long_name} [{data[name].units}]"
    else:
        barLabel = name
    return name, barLabel


def get_colourmap(bar_label: str):
    colourmap = {}
    colourmap["c_map"] = plt.get_cmap("YlGnBu_r")
    colourmap["c_map"].set_bad("gray")
    colourmap["c_map_center"] = None
    colourmap["cbar_args"] = {"label": bar_label}
    colourmap["vmin"] = None
    colourmap["vmax"] = None

    return colourmap


def set_colourmap(cmap: dict, name: str):
    if name == "LAYER_T":
        cmap["c_map"] = plt.get_cmap("bwr")
        cmap["c_map_center"] = 273.16
    elif name == "LAYER_G_PENETRATING":
        cmap["c_map"] = plt.get_cmap("bwr")
        cmap["c_map_center"] = 0.0
    elif name == "LAYER_NTYPE":
        cmap["c_map"] = plt.get_cmap("gist_stern_r", 3)
        cmap["c_labels"] = ["ice", "debris", "snow"]
        cmap["cbar_args"]["format"] = plt.FuncFormatter(
            lambda val, loc: cmap["c_labels"][int(val)]
        )
        cmap["cbar_args"]["ticks"] = [-1, 0, 1]
        cmap["vmin"] = -1
        cmap["vmax"] = 1
    elif name not in ["LAYER_CC", "LAYER_REFREEZE"]:
        cmap["vmin"] = 0

    return cmap


def get_spatiotemporal_domain(
    data: xr.Dataset,
    start: str = None,
    end: str = None,
    lat: str = None,
    lon: str = None,
) -> xr.Dataset:
    if (start is not None) & (end is not None):
        data = data.sel(time=slice(start, end))

    # Select location
    if (lat is not None) & (lon is not None):
        data = data.sel(lat=lat, lon=lon, method="nearest")

    return data


def get_squeezed_data(
    data: xr.Dataset, name: str, latitude: str = None, longitude: str = None
):
    if (latitude is None) & (longitude is None):
        data = data[name].squeeze(["lat", "lon"])
    else:
        data = data[name][:, :].values

    return data


def get_minimum_layer_height(data: xr.Dataset):
    # height = data.attrs["First_layer_height_log_profile"]
    height = get_squeezed_data(data=data, name="LAYER_HEIGHT").min(skipna=True)
    return height


def plot_profile_1D_timeseries_test(
    filename,
    var,
    domainy=None,
    start=None,
    end=None,
    lat=None,
    lon=None,
    stake_file=None,
    pit_name=None,
    output_path: str = None,
):
    # Get dataset
    ds = xr.open_dataset(filename)
    ds = get_spatiotemporal_domain(
        data=ds, start=start, end=end, lat=lat, lon=lon
    )
    # Get first layer height
    fl = ds.attrs["First_layer_height_log_profile"]

    # Get data
    var, barLabel = get_bar_label(data=ds, name=var)
    V = get_squeezed_data(data=ds, name=var, latitude=lat, longitude=lon)
    if (lat is None) & (lon is None):
        # D = np.nancumsum(ds.LAYER_HEIGHT.squeeze(["lat", "lon"]).values, axis=1)
        D = ds.LAYER_HEIGHT.squeeze(["lat", "lon"]).values.cumsum(axis=1)
    else:
        D = ds.LAYER_HEIGHT[:, :].values.cumsum(axis=1)

    colourmap = get_colourmap(bar_label=barLabel)
    colourmap = set_colourmap(cmap=colourmap, name=var)

    if var == "LAYER_NTYPE":
        density = get_squeezed_data(
            data=ds, name="LAYER_RHO", latitude=lat, longitude=lon
        )
        V = xr.where((density < ds.ice_density) & (V != 1), -1, V)
        V = xr.where((V == 1), V - 0.001, V)  # until the cbar gets fixed

    # Get dimensions
    time = np.arange(ds.dims["time"])

    if ds.SNOWHEIGHT.max() > 0.0:
        if (lat is None) & (lon is None):
            depth = ds.SNOWHEIGHT[:, 0, 0].values
        else:
            depth = ds.SNOWHEIGHT[:].values

        # Calc plotting domain height
        Dn = int(np.floor(ds.SNOWHEIGHT.max())) + 1
    else:
        if (lat is None) & (lon is None):
            depth = ds.LAYER_HEIGHT.squeeze(["lat", "lon"]).values
        else:
            depth = ds.TOTALHEIGHT[:].values

        # Calc plotting domain height
        Dn = int(np.floor(ds.TOTALHEIGHT.max())) + 1

    if not domainy:
        domainy = 0.0
    y_size = V.layer.shape[0]
    max_y = ds.TOTALHEIGHT.max(skipna=True)
    xi = V.time.values
    # xi = np.linspace(ds.time.min(), ds.time.max(), ds.time.shape[0])
    # yi = np.linspace(domainy, float(max_y.values), y_size)
    yi = np.linspace(0, y_size, y_size)
    print(f"\n---\n")
    print(xi.shape)
    print(type(xi))
    print(yi.shape)
    print(type(yi))
    print(V.time.shape)
    print(type(V.time))
    print(V.layer.shape)
    print(type(V.layer))
    print(V.values.shape)
    print(type(V.values))
    print(V)
    print(f"\n---\n")
    x, y, z = (V.time.values, V.layer.values, V.values)
    x, y = np.meshgrid(x, y)
    # xi,yi = np.meshgrid(xi,yi)
    print(x.shape)
    print(y.shape)
    print(z.shape)
    print(xi.shape)
    print(yi.shape)
    zi = scinterp.griddata((x, y), z.transpose(), (xi, yi), method="cubic")
    # zi = scinterp.griddata((V.time, D), V, (xi[None,:], yi[:,None]), method='cubic')
    # zi = griddata((V.time.values, V.layer.values), V, D, method="cubic")
    fig, ax = plt.subplots()
    c = ax.pcolormesh(xi, yi, zi, shading="auto")

    if not output_path:
        output_path = get_output_path(
            var_name=var, start_time=start, end_time=end
        )
    plt.savefig(fname=output_path, format="png", dpi=300, bbox_inches="tight")
    # plt.close()


def plot_profile_1D_timeseries_corrected(
    filename,
    var,
    domainy=None,
    start=None,
    end=None,
    lat=None,
    lon=None,
    stake_file=None,
    pit_name=None,
    output_path: str = None,
):
    # Get dataset
    ds = xr.open_dataset(filename)
    ds = get_spatiotemporal_domain(
        data=ds, start=start, end=end, lat=lat, lon=lon
    )
    # Get first layer height
    fl = ds.attrs["First_layer_height_log_profile"]

    # Get data
    var, barLabel = get_bar_label(data=ds, name=var)
    V = get_squeezed_data(data=ds, name=var, latitude=lat, longitude=lon)
    if (lat is None) & (lon is None):
        # D = np.nancumsum(ds.LAYER_HEIGHT.squeeze(["lat", "lon"]).values, axis=1)
        D = ds.LAYER_HEIGHT.squeeze(["lat", "lon"]).values.cumsum(axis=1)
    else:
        D = ds.LAYER_HEIGHT[:, :].values.cumsum(axis=1)

    colourmap = get_colourmap(bar_label=barLabel)
    colourmap = set_colourmap(cmap=colourmap, name=var)

    if var == "LAYER_NTYPE":
        density = get_squeezed_data(
            data=ds, name="LAYER_RHO", latitude=lat, longitude=lon
        )
        V = xr.where((density < ds.ice_density) & (V != 1), -1, V)
        V = xr.where((V == 1), V - 0.001, V)  # until the cbar gets fixed

    # Get dimensions
    time = np.arange(ds.dims["time"])

    if ds.SNOWHEIGHT.max() > 0.0:
        if (lat is None) & (lon is None):
            depth = ds.SNOWHEIGHT[:, 0, 0].values
        else:
            depth = ds.SNOWHEIGHT[:].values

        # Calc plotting domain height
        Dn = int(np.floor(ds.SNOWHEIGHT.max())) + 1
    else:
        if (lat is None) & (lon is None):
            depth = ds.LAYER_HEIGHT.squeeze(["lat", "lon"]).values
        else:
            depth = ds.TOTALHEIGHT[:].values

        # Calc plotting domain height
        Dn = int(np.floor(ds.TOTALHEIGHT.max())) + 1
    
    initialise_formatting()

    plt.figure(figsize=(20, 10))
    ax = plt.gca()

    # V.plot.contourf(
    #         "time", "layer", ax=ax, yincrease=False, cbar_kwargs={"label": barLabel}, cmap=c_map
    #     )
    # y_idx = ds.TOTALHEIGHT.idxmax("time", skipna=True)
    # heights = np.nancumsum(
    #     ds.LAYER_HEIGHT.sel(time=y_idx).squeeze(["lat", "lon"]).values
    # )
    # print(heights)
    # Xout = np.linspace(ds.time.min(), ds.time.max(), ds.time.shape[0])
    # Xout = ds.time.values
    # max_y = ds.TOTALHEIGHT.max(skipna=True)
    # step_y = ds.LAYER_HEIGHT.layer[-1] + 1
    # print(max_y.values)
    # print(step_y.values)
    if not domainy:
        domainy = 0.0
    # y_size = V.layer.shape[0]
    # Yout = np.linspace(domainy, float(max_y.values), y_size)
    # # Yout = np.arange(domainy, Dn, fl)
    # print(Yout)
    # print(f"\n---\n")
    # print(D)
    # print(D.shape)
    # print(depth)
    # print(depth.shape)
    # print(f"\n---\n")

    # Xout_2d, Yout_2d = np.meshgrid(Xout, Yout)
    # data_2d = np.full_like(Xout_2d, np.nan, dtype=np.double)

    # for i in range(len(Xout)):
    #     sel = np.where(Yout < depth[i])
    #     idx, dist = find_nearest_2d(D[i, :], Yout[sel])
    #     data_2d[sel, i] = V[i, idx]

    # print(D.shape)
    # print(V.shape)
    # print(Xout_2d.shape)
    # print(Yout_2d.shape)

    # print(Yout_2d)
    # plt.contourf(D.transpose(), V.transpose().values, cmap=c_map, ylim=float(max_y.values))
    # ax = plt.gca()
    # ax.invert_yaxis()
    max_layers = ds.LAYER_HEIGHT.count(dim="layer").max()
    divnorm = matplotlib.colors.TwoSlopeNorm(
        vmin=colourmap["vmin"],
        vcenter=colourmap["c_map_center"],
        vmax=colourmap["vmax"],
    )

    # print(max_y)
    # print(V.time.shape)
    # print(D.shape)
    # y_coord = D
    # x_coord = V.time
    # print(x_coord.shape)
    # print(y_coord.shape)
    # print(V.shape)

    # D = -(D.transpose()).transpose()
    # xi = V.time.values
    # xi = np.linspace(ds.time.min(), ds.time.max(), ds.time[1]-ds.time[0])
    # step_y = get_minimum_layer_height(data=ds)
    # print(step_y)
    # yi = np.linspace(
    #     domainy,
    #     float(max_y.values),
    # )
    # yi = np.linspace(domainy, float(max_y.values), max(1,int(fl)))
    # print(f"\n---\n")
    # print(xi.shape)
    # print(type(xi))
    # print(yi.shape)
    # print(type(yi))
    # print(V.time.shape)
    # print(type(V.time))
    # print(V.layer.shape)
    # print(type(V.layer))
    # print(V.values.shape)
    # print(type(V.values))
    # print(V)
    # print(f"\n---\n")
    # zi = griddata(V.time.values, V.layer.values, V.values, xi, yi)
    # zi = griddata((V.time, D), V, (xi[None,:], yi[:,None]), method='cubic')
    # zi = griddata((V.time.values, V.layer.values), V, D, method="cubic")
    # fig, ax = plt.subplots()
    # c = ax.pcolormesh(xi, yi, zi, shading="auto")
    # plt.pcolormesh(
    #     x_coord,
    #     y_coord,
    #     V,
    #     # Xout_2d,
    #     # Yout_2d,
    #     # data_2d,
    #     # ax=ax,
    #     # yincrease=False,
    #     # cbar_kwargs=cbar_args,
    #     cmap=c_map,
    #     # center=c_map_center,
    #     # levels=64,
    #     # yticks=V.layer,
    #     # ylim=float(max_y.values) + 10,
    #     # ylim=max_layers,
    #     vmin=vmin,
    #     vmax=vmax,
    #     norm=divnorm
    # )
    # ax.invert_yaxis()
    V.plot.pcolormesh(
        "time",
        "layer",
        ax=ax,
        yincrease=False,
        cbar_kwargs=colourmap["cbar_args"],
        cmap=colourmap["c_map"],
        center=colourmap["c_map_center"],
        levels=64,
        yticks=V.layer,
        # ylim=float(max_y.values) + 10,
        ylim=max_layers,
        vmin=colourmap["vmin"],
        vmax=colourmap["vmax"],
    )
    # plt.colorbar(mappable=cm.ScalarMappable(c_map=c_map,cax=c_map_center)
    if not output_path:
        output_path = get_output_path(
            var_name=var, start_time=start, end_time=end
        )
    plt.savefig(fname=output_path, format="png", dpi=300, bbox_inches="tight")
    # plt.close()


def plot_profile_1D_corrected(filename, pdate, d=None, lat=None, lon=None):
    """This creates a simple plot showing the 2D fields"""

    DATA = xr.open_dataset(filename)
    DATA = DATA.sel(time=pdate)

    if (lat is not None) & (lon is not None):
        DATA = DATA.sel(lat=lat, lon=lon, method="nearest")

    plt.figure(figsize=(5, 5))
    # depth = np.append(0, np.cumsum(DATA.LAYER_HEIGHT.values))

    if (lat is None) & (lon is None):
        rho = DATA.LAYER_RHO.squeeze(["lat", "lon"])
        t = DATA.LAYER_T.squeeze(["lat", "lon"])
        depth = np.cumsum(DATA.LAYER_HEIGHT.squeeze(["lat", "lon"]), axis=1)
        # rho = np.append(DATA.LAYER_RHO[:, :, 0], DATA.LAYER_RHO.values)
        # t = np.append(DATA.LAYER_T[:, :, 0], DATA.LAYER_T.values)
    else:
        rho = np.append(DATA.LAYER_RHO[0], DATA.LAYER_RHO.values)
        t = np.append(DATA.LAYER_T[0], DATA.LAYER_T.values)

    print("Date: %s" % (pdate))
    # print('T2: %.2f \t RH: %.2f \t U: %.2f \t G: %.2f' % (DATA.T2,DATA.RH2,DATA.U2,DATA.G))
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

    args = parser.parse_args()

    if args.pdate is None:
        plot_profile_1D_timeseries_corrected(
            filename=args.file,
            var=args.var,
            domainy=args.d,
            start=args.start,
            end=args.end,
            stake_file=args.stake_file,
            pit_name=args.pit_name,
            output_path=args.output_path,
        )
    else:
        plot_profile_1D_corrected(
            args.file, args.pdate, args.d, args.lat, args.lon
        )
