from types import SimpleNamespace

import numpy as np
from numba import njit
from scipy.optimize import OptimizeResult, minimize, newton

import constants
from config import use_debris


def get_minimisation_bounds(grid) -> tuple:
    """Get bounds for surface temperature minimisation.

    Args:
        grid (Grid): Glacier mesh.

    Returns:
        tuple[float, float]: Lower and upper minimisation bounds.
    """
    lower = 220.0
    if grid.get_node_ntype(0) != 0:
        upper = constants.debris_max_temperature
    else:
        upper = constants.zero_temperature

    return lower, upper


# fmt: off
def pack_minimisation_arguments(
    GRID, dt: int, z: float, z0: float, T2: float, rH2: float,
    p: float, SWnet: float, u2: float, RAIN: float, SLOPE: float,
    B_Ts: np.ndarray, LWin: float, N: float,
) -> dict:  # fmt: on
    """Pack variables used for minimisation arguments into a dictionary.

    Dict lookups are faster than for tuples [O(1) vs. O(n)].
    """
    return locals()


# fmt: off
def minimize_surface_energy_balance(
    optim_func, grid, dt: int, z: float, z_0: float,
    temperature_2m: float, rel_humidity: float, pressure: float,
    shortwave_net: float, wind_velocity: float, rain: float, slope: float,
    subsurface_temperatures: np.ndarray, longwave_in: float,
    cloud_fraction: float, lower_bound: float, upper_bound: float,
) -> OptimizeResult:
    """ Minimise the surface energy balance by summing the fluxes."""
    res = minimize(
        fun=optim_func,
        x0=grid.get_node_temperature(0),
        method=constants.sfc_temperature_method,
        bounds=((lower_bound, upper_bound),),
        tol=1e-2,
        args=(
            grid, dt, z, z_0, temperature_2m, rel_humidity, pressure,
            shortwave_net, wind_velocity, rain, slope, subsurface_temperatures,
            longwave_in, cloud_fraction,
        ),
    )
    return res
# fmt: on


# fmt: off
def minimize_newton(
    optim_func, grid, dt: int, z: float, z_0: float,
    temperature_2m: float, rel_humidity: float, pressure: float,
    shortwave_net: float, wind_velocity: float, rain: float, slope: float,
    subsurface_temperatures: np.ndarray, longwave_in: float,
    cloud_fraction: float,
) -> np.ndarray:
    residual = newton(
        func=optim_func,
        x0=np.array([grid.get_node_temperature(0)]),
        tol=1e-2,
        maxiter=50,
        args=(
            grid, dt, z, z_0, temperature_2m, rel_humidity, pressure,
            shortwave_net, wind_velocity, rain, slope, subsurface_temperatures,
            longwave_in, cloud_fraction,
        ),
    )

    return residual
# fmt: on


# fmt: off
def update_surface_temperature(
    GRID, dt: int, z: float, z0: float, T2: float, rH2: float, p: float,
    SWnet: float, u2: float, RAIN: float, SLOPE: float,
    LWin: float = None, N: float = None,
) -> tuple:
# fmt: on
    """Update the surface temperature and get the surface fluxes.

    Args:
        GRID: Grid structure.
        dt: Integration time [s] -- can vary in WRF_X_CSPY.
        z: Measurement height [m] -- varies in WRF_X_CSPY.
        z0: Roughness length [m].
        T2: Air temperature [K].
        rH2: Relative humidity [%].
        p: Air pressure [hPa].
        SWnet: Incoming shortwave radiation [W m^-2].
        u2: Wind velocity [m S^-1].
        RAIN: RAIN (mm).
        SLOPE: Slope of the surface [degree].
        LWin: Incoming longwave radiation [W m^-2].
        N: Fractional cloud cover [-].

    Returns:
        fun: minimisation function.
        x (float): surface temperature.
        Li (float): Incoming longwave radiation [W m^-2].
        Lo (float): Outgoing longwave radiation [W m^-2].
        H (float): Sensible heat flux [W m^-2].
        L (float): Latent heat flux [W m^-2].
        B (float): Ground heat flux [W m^-2].
        Qrr (float): Rain heat flux [W m^-2].
        rho (float): Air density [kg m^-3].
        Lv (float): Latent heat of vaporization [J kg^-1].
        MOL (float): Monin-Obukhov length.
        Cs_t (float): Stanton number [-].
        Cs_q (float): Dalton number [-].
        q0 (float): Mixing ratio at the surface [kg kg^-1].
        q2 (float): Mixing ratio at measurement height [kg kg^-1].
    """

    # Interpolate subsurface temperatures to selected subsurface depths for GHF computation
    B_Ts = interp_subT(GRID)

    # Set minimisation bounds
    lower_bnd_ts, upper_bnd_ts = get_minimisation_bounds(GRID)
    if use_debris:
        optimisation_function = eb_optim_debris
    else:
        optimisation_function = eb_optim

    # Update surface temperature
    # fmt: off
    if constants.sfc_temperature_method in ["L-BFGS-B", "SLSQP"]:
        # Get surface temperature by minimizing the energy balance function (SWnet+Li+Lo+H+L=0)
        res = minimize_surface_energy_balance(
            optimisation_function, GRID, dt, z, z0, T2, rH2, p, SWnet, u2,
            RAIN, SLOPE, B_Ts, LWin, N, lower_bnd_ts, upper_bnd_ts,
        )
    elif constants.sfc_temperature_method == "Newton":
        try:
            res = minimize_newton(
                optimisation_function, GRID, dt, z, z0, T2, rH2, p,
                SWnet, u2, RAIN, SLOPE, B_Ts, LWin, N,
            )
            if res < lower_bnd_ts:
                raise ValueError("TS Solution is out of bounds")
            res = SimpleNamespace(
                **{"x": min(np.array([upper_bnd_ts]), res), "fun": None}
            )
            # min_res=min(np.array([constants.zero_temperature]),res)
            # res = SimpleNamespace(x=min_res, fun=None)
        except (RuntimeError, ValueError):
            # Workaround for non-convergence and unboundedness
            res = minimize(
                optimisation_function,
                GRID.get_node_temperature(0),
                method="SLSQP",
                bounds=((lower_bnd_ts, upper_bnd_ts),),
                tol=1e-2,
                args=(
                    optimisation_function, GRID, dt, z, z0, T2, rH2, p,
                    SWnet, u2, RAIN, SLOPE, B_Ts, LWin, N,
                ),
            )
    else:
        raise ValueError("Invalid method for minimizing the residual.")

    # Set surface temperature
    GRID.set_node_temperature(0, res.x)

    (Li, Lo, H, L, B, Qrr, rho, Lv, MOL, Cs_t, Cs_q, q0, q2) = eb_fluxes(
        GRID, res.x, dt, z, z0, T2, rH2, p, u2, RAIN, SLOPE, B_Ts, LWin, N,
    )

    # Consistency check
    if not (lower_bnd_ts < res.x < upper_bnd_ts):
        print(
            "Surface temperature is outside bounds:",
            GRID.get_node_temperature(0),
        )

    # Return fluxes
    return (
        res.fun, res.x, Li, Lo, H, L, B, Qrr, rho, Lv, MOL, Cs_t, Cs_q, q0, q2,
    )
# fmt: on


@njit
def get_subsurface_temperature(
    GRID, cumulative_depth: np.ndarray, zlt: float
) -> float:
    """Get subsurface temperature.

    Args:
        GRID (Grid): Gridded data instance.
        cumulative_depth: Cumulative glacier layer heights [m].
        zlt: Interpolation depth [m].

    Returns:
        Subsurface temperature at the interpolation depth.
    """

    # Find indexes of two depths for temperature interpolation
    idx1_depth = np.abs(cumulative_depth - zlt).argmin()
    depth = cumulative_depth.flat[idx1_depth]

    if depth > zlt:
        idx2_depth = idx1_depth - 1
    else:
        idx2_depth = idx1_depth + 1

    temperature_idx1 = GRID.get_node_temperature(idx1_depth)
    t_z = temperature_idx1 + (
        (temperature_idx1 - GRID.get_node_temperature(idx2_depth))
        / (cumulative_depth[idx1_depth] - cumulative_depth[idx2_depth])
    ) * (zlt - cumulative_depth[idx1_depth])

    return t_z


@njit
def interp_subT(GRID) -> np.ndarray:
    """Interpolate subsurface temperature to depths used for ground heat flux."""

    # Cumulative layer depths
    layer_heights_cum = np.cumsum(np.array(GRID.get_height()))

    t_z1 = get_subsurface_temperature(GRID, layer_heights_cum, constants.zlt1)
    t_z2 = get_subsurface_temperature(GRID, layer_heights_cum, constants.zlt2)

    return np.array([t_z1, t_z2])


@njit
def interp_subT_debris(GRID) -> np.ndarray:
    """Get debris temperatures at its layer extents."""
    layer_heights_cum = np.cumsum(np.array(GRID.get_height()))
    bottom_debris_idx, top_ice_idx = GRID.get_debris_extents()
    t_z1 = get_subsurface_temperature(
        GRID, layer_heights_cum, bottom_debris_idx
    )
    t_z2 = get_subsurface_temperature(GRID, layer_heights_cum, top_ice_idx)

    return np.array([t_z1, t_z2])


@njit
def get_saturation_vapor_pressure(T_0: float, T_2: float) -> tuple:
    """Get saturation vapour pressure.

    Args:
        T_0: Surface temperature [K].
        T_2: 2m air temperature [K].

    Returns:
        tuple[float, float]: Surface and 2m saturation vapour pressure.
    """

    if constants.saturation_water_vapour_method == "Sonntag90":
        Ew = method_EW_Sonntag(T_2)
        Ew0 = method_EW_Sonntag(T_0)
    else:
        msg = (
            f"Method for saturation water vapour",
            f"{constants.saturation_water_vapour_method}",
            "not available, using default",
        )
        print(" ".join(msg))
        Ew = method_EW_Sonntag(T_2)
        Ew0 = method_EW_Sonntag(T_0)

    return Ew, Ew0


@njit
def eb_fluxes(
    GRID, T0, dt, z, z0, T2, rH2, p, u2, RAIN, SLOPE, B_Ts, LWin=None, N=None
):
    """This functions returns the surface fluxes with Monin-Obukhov stability correction.

    Given:

        GRID    ::  Grid structure
        T0      ::  Surface temperature [K]
        dt      ::  Integration time [s]
        z       ::  Measurement height [m]
        z0      ::  Roughness length [m]
        T2      ::  Air temperature [K]
        rH2     ::  Relative humidity [%]
        p       ::  Air pressure [hPa]
        u2      ::  Wind velocity [m S^-1]
        RAIN    ::  RAIN (mm)
        SLOPE   ::  Slope of the surface [degree]
        B_Ts    ::  Subsurface temperatures at interpolation depths [K]
        LWin    ::  Incoming longwave radiation [W m^-2]
        N       ::  Fractional cloud cover [-]

    Returns:

        Li      ::  Incoming longwave radiation [W m^-2]
        Lo      ::  Outgoing longwave radiation [W m^-2]
        H       ::  Sensible heat flux [W m^-2]
        L       ::  Latent heat flux [W m^-2]
        B       ::  Ground heat flux [W m^-2]
        Qrr     ::  Rain heat flux [W m^-2]
        SWnet   ::  Shortwave radiation budget [W m^-2]
        rho     ::  Air density [kg m^-3]
        Lv      ::  Latent heat of vaporization [J kg^-1]
        MOL     ::  Monin Obhukov length
        Cs_t    ::  Stanton number [-]
        Cs_q    ::  Dalton number [-]
        q0      ::  Mixing ratio at the surface [kg kg^-1]
        q2      ::  Mixing ratio at measurement height [kg kg^-1]
        phi     ::  Stability correction term [-]
    """

    # Saturation vapour pressure (hPa)
    if constants.saturation_water_vapour_method == "Sonntag90":
        Ew = method_EW_Sonntag(T2)
        Ew0 = method_EW_Sonntag(T0)
    else:
        print(
            "Method for saturation water vapour ",
            constants.saturation_water_vapour_method,
            " not available, using default",
        )
        Ew = method_EW_Sonntag(T2)
        Ew0 = method_EW_Sonntag(T0)

    # latent heat of vaporization
    if T0 >= constants.zero_temperature:
        Lv = constants.lat_heat_vaporize
    else:
        Lv = constants.lat_heat_sublimation

    # Water vapour at height z in  m (hPa)
    Ea = (rH2 * Ew) / 100.0

    # Calc incoming longwave radiation, if not available Ea has to be in Pa (Konzelmann 1994)
    # numba has no implementation for power(none, int)
    if (LWin is None) and (N is not None):
        eps_cs = 0.23 + 0.433 * np.power(100 * Ea / T2, 1.0 / 8.0)
        eps_tot = eps_cs * (1 - np.power(N, 2)) + 0.984 * np.power(N, 2)
        Li = eps_tot * constants.sigma * np.power(T2, 4.0)
    else:
        # otherwise use LW data from file
        Li = LWin

    # turbulent Prandtl number
    Pr = 0.8

    # Mixing Ratio at surface and at measurement height  or calculate with other formula? 0.622*e/p = q
    q2 = (rH2 * 0.622 * (Ew / (p - Ew))) / 100.0
    q0 = (100.0 * 0.622 * (Ew0 / (p - Ew0))) / 100.0

    # Air density
    rho = (p * 100.0) / (287.058 * (T2 * (1 + 0.608 * q2)))

    # Bulk transfer coefficient
    z0t = z0 / 100  # Roughness length for sensible heat
    z0q = z0 / 10  # Roughness length for moisture
    L = None

    # Avoid recalculating
    slope_radians = np.radians(SLOPE)
    cos_slope_radians = np.cos(slope_radians)

    # Monin-Obukhov stability correction
    if constants.stability_correction == "MO":
        L = 0.0
        H0 = T0 * 0.0 + np.inf  # numba: consistent typing of H0
        diff = np.inf
        optim = True
        niter = 0

        # Optimize Obukhov length
        while optim:
            # ustar with initial condition of L == x
            ust = ustar(u2, z, z0, L)

            # Sensible heat flux for neutral conditions
            delta_phi_tq = phi_tq(z, L) - phi_tq(z0, L)
            Cd = np.power(0.41, 2.0) / np.power(
                np.log(z / z0) - phi_m(z, L) - phi_m(z0, L), 2.0
            )
            Cs_t = 0.41 * np.sqrt(Cd) / (np.log(z / z0t) - delta_phi_tq)
            Cs_q = 0.41 * np.sqrt(Cd) / (np.log(z / z0q) - delta_phi_tq)

            # Surface heat flux
            H = (
                rho
                * constants.spec_heat_air
                * Cs_t
                * u2
                * (T2 - T0)
                * cos_slope_radians
            )

            # Latent heat flux
            LE = rho * Lv * Cs_q * u2 * (q2 - q0) * cos_slope_radians

            # Monin-Obukhov length
            L = MO(rho, ust, T2, H)

            # Heat flux differences between iterations
            diff = np.abs(H0 - H)

            # Termination criterion
            if (diff < 1e-1) | (niter > 5):
                optim = False
            niter = niter + 1

            # Store last heat flux in H0
            H0 = H

    # Richardson-Number stability correction
    elif constants.stability_correction == "Ri":
        # Bulk transfer coefficient
        Cs_t = np.power(0.41, 2.0) / (
            np.log(z / z0) * np.log(z / z0t)
        )  # Stanton-Number
        Cs_q = np.power(0.41, 2.0) / (
            np.log(z / z0) * np.log(z / z0q)
        )  # Dalton-Number

        # Bulk Richardson number
        Ri = 0
        if u2 != 0:
            Ri = (
                (9.81 * (T2 - T0) * z) / (T2 * np.power(u2, 2))
            ).item()  # numba can't compare literal & array below

        # Stability correction
        phi = 1
        if 0.01 < Ri <= 0.2:
            phi = np.power(1 - 5 * Ri, 2)
        elif Ri > 0.2:
            phi = 0

        # Sensible heat flux
        H = (
            rho
            * constants.spec_heat_air
            * Cs_t
            * u2
            * (T2 - T0)
            * phi
            * cos_slope_radians
        )

        # Latent heat flux
        LE = rho * Lv * Cs_q * u2 * (q2 - q0) * phi * cos_slope_radians

    else:
        # msg = f"Stability correction {constants.stability_correction} is not supported."
        raise ValueError("Stability correction is not supported.")

    # Outgoing longwave radiation
    Lo = (
        -constants.surface_emission_coeff * constants.sigma * np.power(T0, 4.0)
    )

    # Get thermal conductivity
    lam = GRID.get_node_thermal_conductivity(0)

    # Ground heat flux
    hminus = constants.zlt1
    hplus = constants.zlt2 - constants.zlt1
    Tz1, Tz2 = B_Ts
    B = lam * (
        (hminus / (hplus + hminus)) * ((Tz2 - Tz1) / hplus)
        + (hplus / (hplus + hminus)) * ((Tz1 - T0) / hminus)
    )

    # Rain heat flux
    QRR = (
        constants.water_density
        * constants.spec_heat_water
        * (RAIN / 1000 / dt)
        * (T2 - T0)
    )

    # Return surface fluxes
    # Numba: No implementation of function Function(<class 'float'>) found for signature: >>> float(array(float64, 1d, C))
    # fmt: off
    return (
        Li.item(), Lo.item(), H.item(), LE.item(), B.item(), QRR.item(),
        rho, Lv, L, Cs_t, Cs_q, q0, q2,
    )  # fmt: on


@njit
def phi_m_stable(z: float, L: float) -> float:
    """Get integrated stability function for stable conditions.

    Args:
        z: Height, [m].
        L: Obukhov length, [m].

    Returns:
        Stability function for momentum under stable conditions.
    """
    zeta = z / L
    if (zeta > 0.0) & (zeta <= 1.0):  # weak stability
        return -5 * zeta
    elif zeta > 1.0:  # strong stability
        return (1 - 5) * (1 + np.log(zeta)) - zeta
    else:
        return 0.0


@njit
def phi_m(z: float, L: float) -> float:
    """Get integrated stability function for the momentum flux.

    Args:
        z: Height, [m].
        L: Obukhov length, [m].

    Returns:
        Integrated stability function for the momentum flux.
    """
    if L > 0:
        return phi_m_stable(z, L)
    elif L < 0:
        x = np.power((1 - 16 * z / L), 0.25)
        return (
            2 * np.log((1 + x) / 2.0)
            + np.log((1 + np.power(x, 2.0)) / 2.0)
            - 2 * np.arctan(x)
            + np.pi / 2.0
        )
    else:
        return 0.0


@njit
def phi_tq(z: float, L: float) -> float:
    """Stability function for the heat and moisture flux."""
    if L > 0:
        return phi_m_stable(z, L)
    elif L < 0:
        x = np.power((1 - 19.3 * z / L), 0.25)
        return 2 * np.log((1 + np.power(x, 2.0)) / 2.0)
    else:
        return 0.0


@njit
def ustar(u2, z, z0, L):
    """Friction velocity."""
    return (0.41 * u2) / (np.log(z / z0) - phi_m(z, L))


@njit
def MO(rho, ust, T2, H):
    """Monin-Obukhov length"""
    if H != 0:
        return (
            (rho * constants.spec_heat_air * np.power(ust, 3) * T2)
            / (0.41 * 9.81 * H)
        ).item()  # numba: expects a float
    else:
        return 0.0


@njit
def method_EW_Sonntag(T: float) -> float:
    """Get saturation vapour pressure.

    Args:
        T: Temperature [K]
    """
    if T >= 273.16:
        # over water
        Ew = 6.112 * np.exp((17.67 * (T - 273.16)) / ((T - 29.66)))
    else:
        # over ice
        Ew = 6.112 * np.exp((22.46 * (T - 273.16)) / ((T - 0.55)))
    return Ew


# fmt: off
@njit
def eb_optim(
    T0: float, GRID, dt: int, z: float, z0: float, T2: float, rH2: float,
    p: float, SWnet: float, u2: float, RAIN: float, SLOPE: float, B_Ts: float,
    LWin: float = None, N: float = None,
) -> float:
    """Optimization function to solve for surface temperature T0"""

    # Get surface fluxes for surface temperature T0
    (Li, Lo, H, L, B, Qrr, _, _, _, _, _, _, _) = eb_fluxes(
        GRID, T0, dt, z, z0, T2, rH2, p, u2, RAIN, SLOPE, B_Ts, LWin, N
    )

    # Return the residual (is minimized by the optimization function)
    if constants.sfc_temperature_method == "Newton":
        return SWnet + Li + Lo + H + L + B + Qrr
    else:
        return np.abs(SWnet + Li + Lo + H + L + B + Qrr)


@njit
def eb_optim_debris(
    T0: float, GRID, dt: int, z: float, z0: float, T2: float, rH2: float,
    p: float, SWnet: float, u2: float, RAIN: float, SLOPE: float, B_Ts: float,
    LWin: float = None, N: float = None,
) -> float:
    """Optimization function to solve for surface temperature T0."""

    # Get surface fluxes for surface temperature T0
    (Li, Lo, H, L, B, Qrr, _, _, _, _, _, _, _) = eb_fluxes(
        GRID, T0, dt, z, z0, T2, rH2, p, u2, RAIN, SLOPE, B_Ts, LWin, N
    )

    # Return the residual (is minimized by the optimization function)
    if constants.sfc_temperature_method == "Newton":
        return SWnet + T0 * ((Li + Lo) + H + L + B + Qrr)
    else:
        return np.abs(SWnet + T0 * ((Li + Lo) + H + L + B + Qrr))
# fmt: on
