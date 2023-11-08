import numpy as np
import pytest

import constants
import cosipy.modules.roughness as module_roughness


class TestParamRoughness:
    """Tests methods for parametrising roughness."""

    @pytest.mark.parametrize("arg_method", ["Wrong Method", "", None])
    def test_updateRoughness_method_error(
        self, monkeypatch, conftest_mock_grid, conftest_boilerplate, arg_method
    ):
        grid = conftest_mock_grid
        valid_methods = ["Moelg12"]

        conftest_boilerplate.patch_variable(
            monkeypatch,
            module_roughness.constants,
            {"roughness_method": arg_method},
        )
        assert module_roughness.constants.roughness_method == arg_method
        error_message = (
            f'Roughness method = "{module_roughness.constants.roughness_method}"',
            f"is not allowed, must be one of",
            f'{", ".join(valid_methods)}',
        )

        with pytest.raises(ValueError, match=" ".join(error_message)):
            module_roughness.updateRoughness(grid)

    def test_method_moelg(
        self,
        monkeypatch,
        conftest_boilerplate,
        conftest_mock_grid,
        conftest_mock_grid_ice,
    ):
        conftest_boilerplate.patch_variable(
            monkeypatch, module_roughness, {"use_debris": False}
        )
        test_grid = conftest_mock_grid
        compare_roughness = module_roughness.method_Moelg(test_grid)

        assert (
            constants.roughness_fresh_snow / 1000
            <= compare_roughness
            <= constants.roughness_firn / 1000
        )

        test_grid_ice = conftest_mock_grid_ice
        ice_roughness = module_roughness.method_Moelg(test_grid_ice)
        assert np.isclose(ice_roughness, constants.roughness_ice / 1000)

    @pytest.mark.parametrize("arg_bury", [True, False])
    def test_method_moelg_debris(
        self,
        monkeypatch,
        conftest_boilerplate,
        conftest_mock_grid,
        conftest_mock_grid_ice,
        arg_bury,
    ):
        conftest_boilerplate.patch_variable(
            monkeypatch, module_roughness, {"use_debris": True}
        )

        test_grid = conftest_mock_grid
        test_grid.add_fresh_debris(0.1, 2840.0, 273.15, 0.0)
        if arg_bury:
            test_grid.add_fresh_snow(0.1, 250.0, 270.0, 0.0)

        compare_roughness = module_roughness.method_Moelg_debris(test_grid)
        if arg_bury:
            assert (
                constants.roughness_fresh_snow / 1000
                <= compare_roughness
                <= constants.roughness_firn / 1000
            )
        else:
            conftest_boilerplate.check_output(
                compare_roughness, float, constants.roughness_debris / 1000
            )

        test_grid_ice = conftest_mock_grid_ice
        ice_roughness = module_roughness.method_Moelg(test_grid_ice)
        assert np.isclose(ice_roughness, constants.roughness_ice / 1000)

    @pytest.mark.parametrize("arg_method", ["Moelg12"])
    @pytest.mark.parametrize("arg_debris", [True, False])
    def test_updateRoughness(
        self,
        monkeypatch,
        conftest_boilerplate,
        conftest_mock_grid,
        conftest_mock_grid_ice,
        arg_method,
        arg_debris,
    ):
        conftest_boilerplate.patch_variable(
            monkeypatch, module_roughness, {"use_debris": arg_debris}
        )
        conftest_boilerplate.patch_variable(
            monkeypatch,
            module_roughness.constants,
            {"roughness_method": arg_method},
        )
        assert module_roughness.constants.roughness_method == arg_method

        GRID = conftest_mock_grid
        roughness = module_roughness.updateRoughness(GRID)
        assert (
            constants.roughness_fresh_snow / 1000
            <= roughness
            <= constants.roughness_firn / 1000
        )

        GRID_ice = conftest_mock_grid_ice
        ice_roughness = module_roughness.updateRoughness(GRID_ice)
        assert np.isclose(ice_roughness, constants.roughness_ice / 1000)
