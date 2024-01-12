import numpy as np
import pytest

import cosipy.modules.heatEquation as module_heatEquation


class TestParamHeatEquation:
    """Tests heat equation solvers."""

    melt_water = 1.0
    timedelta = 3600

    @pytest.mark.parametrize("arg_range", [True, False])
    def test_get_new_temperatures_cds2(self, conftest_mock_grid, arg_range):
        test_grid = conftest_mock_grid
        test_temperatures = test_grid.get_temperature()
        test_layers = test_grid.get_number_layers()

        test_centre = np.arange(1, test_layers - 1)
        test_lower = np.arange(2, test_layers)
        test_upper = np.arange(0, test_layers - 2)
        if not arg_range:  # test other iterables
            test_centre = np.array([*test_centre])
            test_lower = np.array([*test_lower])
            test_upper = np.array([*test_upper])

        compare_temperatures = module_heatEquation.get_new_temperatures_cds2(
            grid=test_grid,
            lower=test_lower,
            central=test_centre,
            upper=test_upper,
            dt=1,
        )

        assert isinstance(compare_temperatures, np.ndarray)
        assert not np.array_equal(compare_temperatures, test_temperatures)

    def test_solveHeatEquation(self, conftest_mock_grid):
        test_grid = conftest_mock_grid
        test_temperature = test_grid.get_temperature()

        module_heatEquation.solveHeatEquation(GRID=test_grid, dt=1)

        compare_temperature = test_grid.get_temperature()

        assert not np.array_equal(compare_temperature, test_temperature)
