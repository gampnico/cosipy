import numpy as np

import pytest
import constants
from COSIPY import start_logging
import cosipy.modules.albedo as module_albedo


class TestParamAlbedoUpdate:
    """Tests get/set methods for albedo properties."""

    def test_updateAlbedo(self, conftest_mock_grid):
        grid = conftest_mock_grid

        albedo = module_albedo.updateAlbedo(grid)
        assert isinstance(albedo, float)
        assert (
            albedo >= constants.albedo_firn
            and albedo <= constants.albedo_fresh_snow
        )

    def test_updateAlbedo_ice(
        self, conftest_mock_grid_ice, conftest_boilerplate
    ):
        grid_ice = conftest_mock_grid_ice
        albedo = module_albedo.updateAlbedo(grid_ice)
        assert conftest_boilerplate.check_output(
            albedo, float, constants.albedo_ice
        )


class TestParamAlbedoSelection:
    """Tests user selection of parametrisation method."""

    @pytest.mark.parametrize("arg_method", ["Oerlemans98"])
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

    @pytest.mark.parametrize("arg_method", ["Wrong Method", "", None])
    def test_updateAlbedo_method_error(
        self, monkeypatch, conftest_mock_grid, conftest_boilerplate, arg_method
    ):
        grid = conftest_mock_grid
        valid_methods = ["Oerlemans98"]

        conftest_boilerplate.patch_variable(
            monkeypatch, module_albedo.constants, {"albedo_method": arg_method}
        )
        assert constants.albedo_method == arg_method
        error_message = (
            f'Albedo method = "{constants.albedo_method}"',
            f"is not allowed, must be one of",
            f'{", ".join(valid_methods)}',
        )

        with pytest.raises(ValueError, match=" ".join(error_message)):
            module_albedo.updateAlbedo(grid)


class TestParamAlbedoMethods:
    """Tests methods for parametrising albedo."""

    def test_method_Oerlemans(self, conftest_mock_grid):
        """Get surface albedo without accounting for snow depth."""

        grid = conftest_mock_grid
        albedo_limit = constants.albedo_firn + (
            constants.albedo_fresh_snow - constants.albedo_firn
        ) * np.exp((0) / (constants.albedo_mod_snow_aging * 24.0))

        compare_albedo = module_albedo.method_Oerlemans(grid)

        assert isinstance(compare_albedo, float)
        assert 0.0 <= compare_albedo <= 1.0
        assert compare_albedo <= albedo_limit
