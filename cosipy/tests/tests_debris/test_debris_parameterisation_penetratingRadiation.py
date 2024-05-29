import numpy as np
import pytest

import constants
import cosipy.modules.penetratingRadiation as pRad


class TestParamRadiationDebris:
    """Tests radiation methods."""

    melt_water = 1.0
    timedelta = 3600

    @pytest.mark.parametrize("arg_direction", [1, -1])
    def test_get_conductive_heat_flux(
        self, conftest_mock_grid, conftest_debris_boilerplate, arg_direction
    ):
        test_grid = conftest_mock_grid
        debris_index = 4
        debris_offset = min(0, arg_direction)
        conftest_debris_boilerplate.add_debris_to_grid(
            test_grid, debris=debris_index + 1, snow=0
        )
        assert test_grid.get_number_debris_layers() == debris_index + 1

        debris_index -= debris_offset
        temperature_gradient = test_grid.get_node_temperature(
            debris_index + arg_direction
        ) - test_grid.get_node_temperature(debris_index)

        compare_flux = pRad.get_conductive_heat_flux(
            grid=test_grid, idx=debris_index, direction=arg_direction
        )
        assert np.sign(compare_flux) == np.sign(arg_direction)
        assert np.sign(compare_flux) != np.sign(temperature_gradient)

    def test_get_decay_curve(
        self,
        conftest_debris_boilerplate,
        conftest_boilerplate,
        conftest_mock_grid,
    ):
        test_grid = conftest_mock_grid
        debris_index = 4
        conftest_debris_boilerplate.add_debris_to_grid(
            test_grid, debris=debris_index + 1, snow=0
        )
        assert test_grid.get_number_debris_layers() == debris_index + 1

        compare_si, compare_decay = pRad.get_decay_curve(
            test_grid, shortwave=300.0
        )

        conftest_boilerplate.check_output(compare_si, float, 300.0)
        assert isinstance(compare_decay, np.ndarray)
        assert all(x == 1 for x in compare_decay[:debris_index])
        assert all(x < 1 for x in compare_decay[debris_index + 1 :])

    @pytest.mark.parametrize(
        "arg_density", [250.0, constants.snow_ice_threshold + 1]
    )
    @pytest.mark.parametrize("arg_bury", [True, False])
    def test_method_Bintanja_debris(
        self,
        monkeypatch,
        conftest_mock_grid,
        conftest_boilerplate,
        arg_density,
        arg_bury,
    ):
        conftest_boilerplate.patch_variable(
            monkeypatch,
            pRad.constants,
            {"penetrating_method": "Bintanja95"},
        )

        test_grid = conftest_mock_grid
        for i in range(6):
            test_grid.add_fresh_debris(0.1, 2840.0, 273.15, 0.0)

        if arg_bury:
            test_grid.add_fresh_snow(0.1, arg_density, 270.15, 0.0)

        test_swnet = 800.0
        if not arg_bury:
            test_si = 0.0
        elif arg_density <= constants.snow_ice_threshold:
            test_si = test_swnet * 0.1
        else:
            test_si = test_swnet * 0.2

        melt_si = pRad.method_Bintanja_debris(
            GRID=test_grid, SWnet=test_swnet, dt=constants.dt
        )
        assert isinstance(melt_si, tuple)
        compare_melt = melt_si[0]
        assert isinstance(compare_melt, float)
        compare_si = melt_si[1]
        conftest_boilerplate.check_output(compare_si, float, test_si)
        assert compare_si >= 0.0
