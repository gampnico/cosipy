import math
import re

import numpy as np
import pytest

import constants
import cosipy.modules.heatEquation as module_heatEquation


class TestParamHeatEquationDebris:
    """Tests heat equations for debris."""

    def add_debris_to_grid(self, grid_obj, debris: int = 5, snow: int = 2):
        """Add debris layers with snowpack."""
        for i in range(0, debris):
            grid_obj.add_fresh_debris(0.1, 2840.0, 274.15, 0.0)
        for i in range(0, snow):
            grid_obj.add_fresh_snow(
                0.05, 300, min(270.15 + i / 2, 273.15), 0.0
            )

    @pytest.mark.parametrize("arg_snow", [0, 2])
    @pytest.mark.parametrize("arg_debris", [0, 3])
    def test_add_debris_to_grid(
        self, conftest_mock_grid, arg_snow, arg_debris
    ):
        """Add debris layer using fixture."""
        test_grid = conftest_mock_grid
        test_nodes = test_grid.number_nodes
        test_snow = test_grid.get_number_snow_layers()

        self.add_debris_to_grid(
            grid_obj=test_grid, debris=arg_debris, snow=arg_snow
        )

        assert test_grid.get_number_debris_layers() == arg_debris
        assert test_grid.get_number_snow_layers() == arg_snow + test_snow

        for i in range(0, arg_snow):
            assert test_grid.get_node_ntype(i) == 0
        for i in range(arg_snow, arg_snow + arg_debris):
            assert test_grid.get_node_ntype(i) == 1
        assert test_grid.number_nodes == test_nodes + arg_debris + arg_snow

    def test_get_contact_temperature(self, conftest_mock_grid):
        test_grid = conftest_mock_grid
        test_grid.add_fresh_debris(0.1, 2840.0, 274.15, 0.0)
        test_grid.add_fresh_snow(0.05, 300, 270.15, 0.0)

        test_contact = module_heatEquation.get_contact_temperature(
            test_grid, 0
        )
        assert isinstance(test_contact, float)
        assert (
            test_grid.get_node_temperature(0)
            < test_contact
            < test_grid.get_node_temperature(1)
        )
        diffusion_time = (
            test_grid.get_node_height(0) ** 2
        ) / test_grid.get_node_thermal_diffusivity(0)
        assert 3600.0 > diffusion_time > 0.0

    def test_get_skin_temperature(
        self, conftest_mock_grid, conftest_boilerplate
    ):
        test_grid = conftest_mock_grid
        test_grid.add_fresh_debris(0.1, 2840.0, 274.15, 0.0)
        test_grid.add_fresh_snow(0.05, 300, 270.15, 0.0)

        test_contact = module_heatEquation.get_contact_temperature(
            test_grid, 0
        )
        test_skin = test_contact + (
            test_grid.get_node_thermal_effusivity(0)
            / test_grid.get_node_thermal_effusivity(1)
        ) * (test_contact - test_grid.get_node_temperature(0))

        compare_skin = module_heatEquation.get_skin_temperature(
            grid=test_grid, idx=0, contact_temperature=test_contact
        )
        conftest_boilerplate.check_output(compare_skin, float, test_skin)

    def test_get_midpoint_temperature(self, conftest_boilerplate):
        test_lower = 1.0
        test_upper = 32.0
        test_height = 0.1
        test_midpoint = (
            (test_upper - test_lower) * (test_height / 2)
        ) + test_lower

        compare_midpoint = module_heatEquation.get_midpoint_temperature(
            skin_lower=test_lower, skin_upper=test_upper, height=test_height
        )
        conftest_boilerplate.check_output(
            compare_midpoint, float, test_midpoint
        )

    def test_set_midpoint_temperatures(
        self, conftest_mock_grid, conftest_boilerplate
    ):
        test_grid = conftest_mock_grid
        self.add_debris_to_grid(grid_obj=test_grid)
        test_temperatures = test_grid.get_temperature()
        test_contact = module_heatEquation.get_contact_temperature(
            grid=test_grid, idx=1
        )
        module_heatEquation.set_midpoint_temperatures(
            grid=test_grid, contact_temperature=test_contact, idx=1
        )

        compare_temperatures = test_grid.get_temperature()

        conftest_boilerplate.check_output(
            compare_temperatures[0], float, test_temperatures[0]
        )
        assert compare_temperatures[1] > test_temperatures[1]  # snow warms
        assert compare_temperatures[2] < test_temperatures[2]  # debris cools

    def test_get_ivhc_tensor(self, conftest_mock_grid):
        test_grid = conftest_mock_grid
        test_extents = (1, 3)
        test_shape = (test_extents[1] - test_extents[0]) + 1
        test_dt = 2
        test_ivhc = (
            1
            * test_grid.get_thermal_conductivity()[
                test_extents[0] : test_extents[1] + 1
            ]
            / (
                2
                * constants.spec_heat_debris
                * np.power(
                    test_grid.get_height()[
                        test_extents[0] : test_extents[1] + 1
                    ],
                    2,
                )
                * test_grid.get_density()[
                    test_extents[0] : test_extents[1] + 1
                ]
            )
        )
        assert isinstance(test_ivhc, np.ndarray)
        assert test_ivhc.shape == (test_shape,)

        compare_ivhc_default = module_heatEquation.get_ivhc_tensor(
            grid=test_grid, extents=test_extents
        )
        assert isinstance(compare_ivhc_default, np.ndarray)
        assert compare_ivhc_default.shape == (test_shape,)

        compare_ivhc_dt = module_heatEquation.get_ivhc_tensor(
            grid=test_grid, extents=test_extents, dt=test_dt
        )
        assert isinstance(compare_ivhc_dt, np.ndarray)
        assert compare_ivhc_dt.shape == (test_shape,)

        np.testing.assert_allclose(compare_ivhc_default, test_ivhc)
        np.testing.assert_allclose(
            compare_ivhc_dt, compare_ivhc_default * test_dt
        )

        for array in (compare_ivhc_default, compare_ivhc_dt):
            assert array.all() > 0
            assert np.isfinite(array).all()

    def test_get_d_tensor(self, conftest_mock_grid):
        test_grid = conftest_mock_grid
        min_debris = 5
        min_snow = 2
        self.add_debris_to_grid(
            grid_obj=test_grid, debris=min_debris, snow=min_snow
        )
        test_extents = (2, 7)
        test_ivhc = module_heatEquation.get_ivhc_tensor(
            grid=test_grid, extents=test_extents, dt=1
        )
        compare_tensor = module_heatEquation.get_d_tensor(
            grid=test_grid,
            ivhc_tensor=test_ivhc,
            skin_temperature=272.15,
            extents=test_extents,
        )
        assert isinstance(compare_tensor, np.ndarray)
        assert compare_tensor.shape == (1 + test_extents[1] - test_extents[0],)
        assert compare_tensor.shape == test_ivhc.shape

        assert not compare_tensor.take(0, -1).all()
        assert compare_tensor[1:-1].all() > 0
        assert np.isfinite(compare_tensor).all()

    def test_get_a_tensor(self, conftest_mock_grid):
        test_grid = conftest_mock_grid
        min_debris = 5
        min_snow = 2
        self.add_debris_to_grid(
            grid_obj=test_grid, debris=min_debris, snow=min_snow
        )
        test_extents = (min_snow, min_snow + min_debris)
        test_ivhc = module_heatEquation.get_ivhc_tensor(
            grid=test_grid, extents=test_extents, dt=1
        )
        compare_tensor = module_heatEquation.get_a_tensor(
            ivhc_tensor=test_ivhc
        )

        assert isinstance(compare_tensor, np.ndarray)
        assert compare_tensor.shape == test_ivhc.shape
        assert not compare_tensor.take(0, -1).all()
        assert compare_tensor[1:-1].all() > 0
        assert np.isfinite(compare_tensor).all()

    def test_get_s_tensor(self, conftest_mock_grid):
        test_grid = conftest_mock_grid
        min_debris = 5
        min_snow = 2
        self.add_debris_to_grid(
            grid_obj=test_grid, debris=min_debris, snow=min_snow
        )
        test_extents = (min_snow, min_snow + min_debris)
        test_ivhc = module_heatEquation.get_ivhc_tensor(
            grid=test_grid, extents=test_extents, dt=1
        )
        test_a_tensor = module_heatEquation.get_a_tensor(ivhc_tensor=test_ivhc)
        compare_tensor = module_heatEquation.get_s_tensor(
            grid=test_grid,
            a_tensor=test_a_tensor,
            ivhc_tensor=test_ivhc,
            previous_temperature=272.15,
            extents=test_extents,
        )

        assert isinstance(compare_tensor, np.ndarray)
        assert compare_tensor.shape == test_a_tensor.shape
        assert not compare_tensor.take(0, -1).all()
        assert compare_tensor[1:-1].all() > 0
        assert np.isfinite(compare_tensor).all()

    @pytest.mark.parametrize("arg_snow", [0, 1, 2, 3])
    def test_set_upper_boundary_condition(self, conftest_mock_grid, arg_snow):
        test_grid = conftest_mock_grid
        self.add_debris_to_grid(grid_obj=test_grid, debris=5, snow=arg_snow)

        test_snow = test_grid.get_number_snow_layers()
        test_debris = test_grid.get_number_debris_layers()
        test_temperatures = test_grid.get_temperature()

        module_heatEquation.set_upper_boundary_condition(
            grid=test_grid, top_idx=arg_snow
        )

        compare_temperatures = test_grid.get_temperature()
        assert test_grid.get_number_snow_layers() == test_snow
        assert test_grid.get_number_debris_layers() == test_debris
        if arg_snow:
            assert (
                compare_temperatures[arg_snow] != test_temperatures[arg_snow]
            )
        if arg_snow > 1:
            assert (
                compare_temperatures[arg_snow - 1]
                > test_temperatures[arg_snow - 1]
            )
            assert compare_temperatures[arg_snow] < test_temperatures[arg_snow]

    @pytest.mark.parametrize("arg_snow", [0, 1, 2, 3])
    def test_get_internal_debris_temperature_error(
        self, conftest_mock_grid, arg_snow
    ):
        test_grid = conftest_mock_grid
        self.add_debris_to_grid(grid_obj=test_grid, debris=3, snow=arg_snow)
        error_message = " ".join(
            ("Reid & Brock (2010) requires 5+ debris layers.", f"There are 3.")
        )

        with pytest.raises(ValueError, match=re.escape(error_message)):
            module_heatEquation.get_internal_debris_temperature(grid=test_grid)

    @pytest.mark.parametrize("arg_snow", [0, 1, 2, 3])
    def test_get_internal_debris_temperature(
        self, conftest_mock_grid, arg_snow
    ):
        test_grid = conftest_mock_grid
        min_debris = 5
        self.add_debris_to_grid(
            grid_obj=test_grid, debris=min_debris, snow=arg_snow
        )
        test_layers = test_grid.get_number_debris_layers()

        compare_temperatures = (
            module_heatEquation.get_internal_debris_temperature(grid=test_grid)
        )

        assert isinstance(compare_temperatures, np.ndarray)
        assert compare_temperatures.shape == (test_layers,)
        assert not compare_temperatures.take(0, -1).all()
        assert compare_temperatures[1:-1].all() > 0
        assert np.isfinite(compare_temperatures).all()

    @pytest.mark.parametrize("arg_snow", [0, 1, 2, 3])
    def test_set_internal_debris_temperature(
        self, conftest_mock_grid, arg_snow
    ):
        test_grid = conftest_mock_grid
        min_debris = 5
        self.add_debris_to_grid(
            grid_obj=test_grid, debris=min_debris, snow=arg_snow
        )
        test_temperature = test_grid.get_temperature()
        test_extents = test_grid.get_debris_extents()

        deb_temperatures = module_heatEquation.get_internal_debris_temperature(
            test_grid
        )
        module_heatEquation.set_internal_debris_temperature(
            grid=test_grid, temperatures=deb_temperatures
        )

        compare_temperature = test_grid.get_temperature()

        assert len(compare_temperature) == len(test_temperature)

        if arg_snow < 2:
            np.testing.assert_allclose(
                compare_temperature[: test_extents[0]],
                test_temperature[: test_extents[0]],
            )
        else:
            np.testing.assert_allclose(
                compare_temperature[: test_extents[0] - 1],
                test_temperature[: test_extents[0] - 1],
            )
        np.testing.assert_allclose(
            compare_temperature[test_extents[1] :],
            test_temperature[test_extents[1] :],
        )

    def test_solveHeatEquation_debris_nodebris(
        self, conftest_mock_grid, conftest_boilerplate
    ):
        test_grid = conftest_mock_grid
        test_grid.add_fresh_snow(0.05, 300, 270.15, 0.0)
        test_temperature = test_grid.get_temperature()
        assert test_grid.get_number_debris_layers() == 0

        # error raised if debris present
        module_heatEquation.solveHeatEquation_debris(grid=test_grid, dt=1)

        compare_temperature = test_grid.get_temperature()
        assert not np.allclose(compare_temperature, test_temperature)

    @pytest.mark.parametrize("arg_snow", [0, 1, 2, 3])
    def test_set_heat_conservation_debris(
        self, conftest_mock_grid, conftest_boilerplate, arg_snow
    ):
        test_grid = conftest_mock_grid
        min_debris = 5
        self.add_debris_to_grid(
            grid_obj=test_grid, debris=min_debris, snow=arg_snow
        )
        test_temperature = test_grid.get_temperature()
        test_extents = test_grid.get_debris_extents()

        compare_temperature = module_heatEquation.set_heat_conservation_debris(
            grid=test_grid, dt=1
        )

        compare_temperature = test_grid.get_temperature()

        assert len(compare_temperature) == len(test_temperature)
        if arg_snow < 2:
            np.testing.assert_allclose(
                compare_temperature[: test_extents[0]],
                test_temperature[: test_extents[0]],
            )
        else:
            np.testing.assert_allclose(
                compare_temperature[: test_extents[0] - 1],
                test_temperature[: test_extents[0] - 1],
            )
        np.testing.assert_allclose(
            compare_temperature[test_extents[1] :],
            test_temperature[test_extents[1] :],
        )

    
