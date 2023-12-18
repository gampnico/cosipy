import numpy as np
import pytest
from scipy.optimize import OptimizeResult

import constants
import cosipy.modules.surfaceTemperature as module_surface_temperature


class TestParamSurfaceTemperature:
    """Tests methods for parametrising surface temperature."""

    dt = 3600
    z = 2
    z0 = 0.24 / 1000
    T2 = 275
    rH2 = 50
    p = 1000
    SWnet = 789
    u2 = 3.5
    RAIN = 0.1
    SLOPE = 0.0
    B_Ts = np.array([270.15, 268.15])
    LWin = None
    N = 0.5

    test_args = {
        "dt": dt,
        "z": z,
        "z0": z0,
        "T2": T2,
        "rH2": rH2,
        "p": p,
        "SWnet": SWnet,
        "u2": u2,
        "RAIN": RAIN,
        "SLOPE": SLOPE,
        "B_Ts": B_Ts,
        "LWin": LWin,
        "N": N,
    }

    def test_pack_minimisation_arguments(self, conftest_mock_grid):
        test_grid = conftest_mock_grid
        test_args = self.test_args
        test_args["GRID"] = test_grid

        compare_args = module_surface_temperature.pack_minimisation_arguments(
            GRID=test_grid,
            dt=self.dt,
            z=self.z,
            z0=self.z0,
            T2=self.T2,
            rH2=self.rH2,
            p=self.p,
            SWnet=self.SWnet,
            u2=self.u2,
            RAIN=self.RAIN,
            SLOPE=self.SLOPE,
            B_Ts=self.B_Ts,
            LWin=self.LWin,
            N=self.N,
        )
        assert isinstance(compare_args, dict)
        assert compare_args == test_args

    @pytest.mark.parametrize("arg_optim", ["L-BFGS-B", "SLSQP"])
    def test_minimize_surface_energy_balance(
        self, monkeypatch, conftest_mock_grid, conftest_boilerplate, arg_optim
    ):
        conftest_boilerplate.patch_variable(
            monkeypatch,
            module_surface_temperature.constants,
            {"sfc_temperature_method": arg_optim},
        )
        test_grid = conftest_mock_grid
        bounds = module_surface_temperature.get_minimisation_bounds(test_grid)

        residual = module_surface_temperature.minimize_surface_energy_balance(
            optim_func=module_surface_temperature.eb_optim,
            grid=test_grid,
            dt=self.dt,
            z=self.z,
            z_0=self.z0,
            temperature_2m=self.T2,
            rel_humidity=self.rH2,
            pressure=self.p,
            shortwave_net=self.SWnet,
            wind_velocity=self.u2,
            rain=self.RAIN,
            slope=self.SLOPE,
            subsurface_temperatures=self.B_Ts,
            longwave_in=self.LWin,
            cloud_fraction=self.N,
            lower_bound=bounds[0],
            upper_bound=bounds[1],
        )

        assert isinstance(residual, OptimizeResult)
        conftest_boilerplate.check_output(
            residual.x[0], float, constants.zero_temperature
        )
        assert bounds[0] < residual.x[0] < bounds[1]

    @pytest.mark.parametrize(
        "arg_debris",
        [
            module_surface_temperature.eb_optim,
            module_surface_temperature.eb_optim_debris,
        ],
    )
    def test_minimize_newton(
        self, monkeypatch, conftest_mock_grid, conftest_boilerplate, arg_debris
    ):
        conftest_boilerplate.patch_variable(
            monkeypatch,
            module_surface_temperature.constants,
            {"sfc_temperature_method": "Newton"},
        )
        test_grid = conftest_mock_grid
        bounds = module_surface_temperature.get_minimisation_bounds(test_grid)

        residual = module_surface_temperature.minimize_newton(
            optim_func=arg_debris,
            grid=test_grid,
            dt=self.dt,
            z=self.z,
            z_0=self.z0,
            temperature_2m=self.T2,
            rel_humidity=self.rH2,
            pressure=self.p,
            shortwave_net=self.SWnet,
            wind_velocity=self.u2,
            rain=self.RAIN,
            slope=self.SLOPE,
            subsurface_temperatures=self.B_Ts,
            longwave_in=self.LWin,
            cloud_fraction=self.N,
        )

        assert isinstance(residual, np.ndarray)
        assert bounds[0] < residual[0] < bounds[1]

    @pytest.mark.parametrize("arg_optim", ["L-BFGS-B", "SLSQP"])
    def test_minimize_surface_energy_balance_debris(
        self, monkeypatch, conftest_mock_grid, conftest_boilerplate, arg_optim
    ):
        conftest_boilerplate.patch_variable(
            monkeypatch,
            module_surface_temperature.constants,
            {"sfc_temperature_method": arg_optim},
        )
        test_grid = conftest_mock_grid
        test_grid.add_fresh_debris(0.1, 2840.0, 273.15, 0.0)
        bounds = module_surface_temperature.get_minimisation_bounds(test_grid)

        residual = module_surface_temperature.minimize_surface_energy_balance(
            optim_func=module_surface_temperature.eb_optim_debris,
            grid=test_grid,
            dt=self.dt,
            z=self.z,
            z_0=self.z0,
            temperature_2m=self.T2,
            rel_humidity=self.rH2,
            pressure=self.p,
            shortwave_net=self.SWnet,
            wind_velocity=self.u2,
            rain=self.RAIN,
            slope=self.SLOPE,
            subsurface_temperatures=self.B_Ts,
            longwave_in=self.LWin,
            cloud_fraction=self.N,
            lower_bound=bounds[0],
            upper_bound=bounds[1],
        )

        assert isinstance(residual, OptimizeResult)
        assert bounds[0] < residual.x[0] < bounds[1]
        test_grid.set_node_temperature(0, residual.x[0])

        (
            Li,
            Lo,
            H,
            L,
            B,
            Qrr,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = module_surface_temperature.eb_fluxes(
            test_grid,
            test_grid.get_node_temperature(0),
            self.dt,
            self.z,
            self.z0,
            self.T2,
            self.rH2,
            self.p,
            self.u2,
            self.RAIN,
            self.SLOPE,
            self.B_Ts,
            self.LWin,
            self.N,
        )

        compare_flux_sum = np.abs(
            self.SWnet + residual.x[0] * ((Li + Lo) + H + L + B + Qrr)
        )
        assert -0.5 < compare_flux_sum < 0.5

    def test_minimize_newton_debris(
        self, monkeypatch, conftest_mock_grid, conftest_boilerplate
    ):
        conftest_boilerplate.patch_variable(
            monkeypatch,
            module_surface_temperature.constants,
            {"sfc_temperature_method": "Newton"},
        )
        test_grid = conftest_mock_grid
        test_grid.add_fresh_debris(0.1, 2840.0, 273.15, 0.0)
        bounds = module_surface_temperature.get_minimisation_bounds(test_grid)

        residual = module_surface_temperature.minimize_newton(
            optim_func=module_surface_temperature.eb_optim_debris,
            grid=test_grid,
            dt=self.dt,
            z=self.z,
            z_0=self.z0,
            temperature_2m=self.T2,
            rel_humidity=self.rH2,
            pressure=self.p,
            shortwave_net=self.SWnet,
            wind_velocity=self.u2,
            rain=self.RAIN,
            slope=self.SLOPE,
            subsurface_temperatures=self.B_Ts,
            longwave_in=self.LWin,
            cloud_fraction=self.N,
        )

        assert isinstance(residual, np.ndarray)
        assert bounds[0] < residual[0] < bounds[1]
        test_grid.set_node_temperature(0, residual[0])

        (
            Li,
            Lo,
            H,
            L,
            B,
            Qrr,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = module_surface_temperature.eb_fluxes(
            test_grid,
            test_grid.get_node_temperature(0),
            self.dt,
            self.z,
            self.z0,
            self.T2,
            self.rH2,
            self.p,
            self.u2,
            self.RAIN,
            self.SLOPE,
            self.B_Ts,
            self.LWin,
            self.N,
        )

        compare_flux_sum = self.SWnet + residual[0] * (
            (Li + Lo) + H + L + B + Qrr
        )

        # infinite iterations should sum to zero
        assert -0.5 < compare_flux_sum < 0.5

    def test_surface_temperature_parameterisation(self, conftest_mock_grid):
        GRID = conftest_mock_grid
        test_bounds = module_surface_temperature.get_minimisation_bounds(GRID)

        (
            fun,
            surface_temperature,
            lw_radiation_in,
            lw_radiation_out,
            sensible_heat_flux,
            latent_heat_flux,
            ground_heat_flux,
            rain_heat_flux,
            rho,
            Lv,
            monin_obukhov_length,
            Cs_t,
            Cs_q,
            q0,
            q2,
        ) = module_surface_temperature.update_surface_temperature(
            # Old args: GRID, 0.6, (0.24 / 1000), 275, 0.6, 789, 1000, 4.5, 0.0, 0.1
            GRID=GRID,
            dt=3600,
            z=2,
            z0=(0.24 / 1000),
            T2=275,
            rH2=50,
            p=1000,
            SWnet=789,
            u2=3.5,
            RAIN=0.1,
            SLOPE=0.0,
            LWin=None,
            N=0.5,  # otherwise tries to retrieve LWin from non-existent file
        )

        assert constants.zero_temperature >= surface_temperature >= 220.0
        assert test_bounds[0] < surface_temperature < test_bounds[1]
        assert 400 >= lw_radiation_in >= 0
        assert 0 >= lw_radiation_out >= -400
        assert 250 >= sensible_heat_flux >= -250
        assert 200 >= latent_heat_flux >= -200
        assert 100 >= ground_heat_flux >= -100
