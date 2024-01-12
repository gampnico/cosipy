import re

import numpy as np
import pytest

import constants
import cosipy.modules.albedo as module_albedo
from COSIPY import start_logging


class TestParamAlbedoUpdate:
    """Tests get/set methods for albedo properties."""

    def test_updateAlbedo(
        self, monkeypatch, conftest_boilerplate, conftest_mock_grid
    ):
        conftest_boilerplate.patch_variable(
            monkeypatch, module_albedo, {"use_debris": False}
        )

        grid = conftest_mock_grid

        albedo = module_albedo.updateAlbedo(grid)
        assert isinstance(albedo, float)
        assert constants.albedo_firn <= albedo <= constants.albedo_fresh_snow

    def test_updateAlbedo_ice(
        self, monkeypatch, conftest_mock_grid_ice, conftest_boilerplate
    ):
        conftest_boilerplate.patch_variable(
            monkeypatch, module_albedo, {"use_debris": False}
        )
        grid_ice = conftest_mock_grid_ice
        albedo = module_albedo.updateAlbedo(grid_ice)
        assert conftest_boilerplate.check_output(
            albedo, float, constants.albedo_ice
        )

    @pytest.mark.parametrize("arg_height", [0.0, 0.05, 0.1, 0.5])
    @pytest.mark.parametrize(
        "arg_temperature",
        [constants.temperature_bottom, constants.zero_temperature, 280.0],
    )
    @pytest.mark.parametrize(
        "arg_time", [0, constants.albedo_mod_snow_aging * 24]
    )
    def test_get_surface_properties(
        self,
        conftest_mock_grid,
        conftest_boilerplate,
        arg_height,
        arg_temperature,
        arg_time,
    ):
        """Get snow height, timestamp, and time since last snowfall."""

        grid = conftest_mock_grid
        surface = module_albedo.get_surface_properties(GRID=grid)

        assert isinstance(surface, tuple)
        assert len(surface) == 3
        assert all(isinstance(parameter, float) for parameter in surface)
        conftest_boilerplate.check_output(surface[0], float, 0.0)
        conftest_boilerplate.check_output(surface[1], float, 0.0)

        grid.add_fresh_snow(
            arg_height,
            constants.constant_density,
            arg_temperature,
            0.0,
        )
        grid.set_fresh_snow_props_update_time(3600 * arg_time)
        fresh_surface = module_albedo.get_surface_properties(GRID=grid)

        assert isinstance(fresh_surface, tuple)
        assert all(isinstance(parameter, float) for parameter in fresh_surface)
        conftest_boilerplate.check_output(fresh_surface[0], float, arg_height)
        conftest_boilerplate.check_output(
            fresh_surface[1], float, arg_time * 3600
        )


class TestParamAlbedoSelection:
    """Tests user selection of parametrisation method."""

    @pytest.mark.parametrize("arg_method", ["Oerlemans98", "Lejeune13"])
    def test_updateAlbedo_method(
        self, monkeypatch, conftest_mock_grid, conftest_boilerplate, arg_method
    ):
        """Set method from constants.py when calculating albedo."""

        grid = conftest_mock_grid

        conftest_boilerplate.patch_variable(
            monkeypatch, module_albedo.constants, {"albedo_method": arg_method}
        )
        assert constants.albedo_method == arg_method
        albedo = module_albedo.updateAlbedo(grid)
        assert isinstance(albedo, float)

    def test_updateAlbedo_method_error(
        self, monkeypatch, conftest_mock_grid, conftest_boilerplate
    ):
        grid = conftest_mock_grid
        valid_methods = ["Oerlemans98", "Lejeune13"]
        test_method = "Wrong Method"
        conftest_boilerplate.patch_variable(
            monkeypatch,
            module_albedo.constants,
            {"albedo_method": test_method},
        )
        conftest_boilerplate.patch_variable(
            monkeypatch, module_albedo, {"use_debris": False}
        )

        assert constants.albedo_method == test_method
        error_message = " ".join(
            (
                f'Albedo method = "{test_method}"',
                f"is not allowed, must be one of",
                f'{", ".join(valid_methods)}',
            )
        )

        with pytest.raises(ValueError, match=re.escape(error_message)):
            module_albedo.updateAlbedo(grid)


class TestParamAlbedoMethods:
    """Tests methods for parametrising albedo."""

    def test_method_Oerlemans(self, conftest_mock_grid):
        """Get surface albedo without accounting for snow depth."""

        grid = conftest_mock_grid
        albedo_limit = constants.albedo_firn + (
            constants.albedo_fresh_snow - constants.albedo_firn
        ) * np.exp(0 / (constants.albedo_mod_snow_aging * 24.0))

        compare_albedo = module_albedo.method_Oerlemans(grid)

        assert isinstance(compare_albedo, float)
        assert 0.0 <= compare_albedo <= 1.0
        assert compare_albedo <= albedo_limit

    @pytest.mark.parametrize("arg_hours", [0.0, 12.0, 25.0])
    def test_get_simple_albedo(self, arg_hours):
        """Get surface albedo without accounting for snow depth."""

        albedo_limit = constants.albedo_firn + (
            constants.albedo_fresh_snow - constants.albedo_firn
        ) * np.exp((0) / (constants.albedo_mod_snow_aging * 24.0))

        compare_albedo = module_albedo.get_simple_albedo(
            elapsed_time=arg_hours
        )

        assert isinstance(compare_albedo, float)
        assert 0.0 <= compare_albedo <= 1.0
        assert compare_albedo <= albedo_limit

    @pytest.mark.parametrize("arg_depth", [0.0, 0.05, 0.1, 1.0])
    def test_albedo_weighting_lejeune(self, arg_depth):
        """Get albedo weight."""

        weight = module_albedo.get_albedo_weight_lejeune(snow_depth=arg_depth)
        assert isinstance(weight, float)

        if arg_depth >= 0.1:
            assert weight == 1.0
        else:
            assert 0 <= weight < 1.0

    def test_method_lejeune(self, conftest_mock_grid, conftest_boilerplate):
        """Get surface albedo for snow-covered debris."""

        grid = conftest_mock_grid

        compare_albedo = module_albedo.method_lejeune(GRID=grid)
        assert 0.0 <= compare_albedo <= 1.0
        assert conftest_boilerplate.check_output(
            compare_albedo, float, constants.albedo_debris
        )

    def test_method_lejeune_ice(
        self, conftest_mock_grid_ice, conftest_boilerplate
    ):
        """Get surface albedo for bare debris."""

        grid = conftest_mock_grid_ice

        compare_albedo = module_albedo.method_lejeune(GRID=grid)
        assert 0.0 <= compare_albedo <= 1.0
        assert conftest_boilerplate.check_output(
            compare_albedo, float, constants.albedo_debris
        )
