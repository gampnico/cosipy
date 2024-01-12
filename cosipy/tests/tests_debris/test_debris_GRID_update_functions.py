import numpy as np
import pytest

import cosipy.cpkernel.grid as cpgrid
from cosipy.cpkernel.node import BaseNode


class TestGridSetter:
    """Tests update methods for Grid objects.

    Do not duplicate tests in `test_GRID_update_functions.py`.
    """

    pass


class TestGridInteractions:
    """Tests interactions between layers."""

    @pytest.mark.parametrize("arg_height", [0.05, 0.1, 0.5])
    @pytest.mark.parametrize("arg_temperature", [273.16, 270.16, 280.0])
    @pytest.mark.parametrize("arg_lwc", [0.0, 0.5])
    def test_add_fresh_snow(
        self,
        conftest_mock_grid,
        conftest_boilerplate,
        arg_height,
        arg_temperature,
        arg_lwc,
    ):
        """Add fresh snow layer."""

        test_grid = conftest_mock_grid
        test_number_nodes = test_grid.number_nodes

        test_grid.add_fresh_snow(arg_height, 250.0, arg_temperature, arg_lwc)
        assert test_grid.number_nodes == test_number_nodes + 1
        compare_grid_data = test_grid.grid
        assert isinstance(compare_grid_data[0], BaseNode)

        compare_node = compare_grid_data[0]
        conftest_boilerplate.check_output(
            compare_node.height, float, arg_height
        )
        conftest_boilerplate.check_output(
            compare_node.temperature, float, arg_temperature
        )
        conftest_boilerplate.check_output(
            compare_node.liquid_water_content, float, arg_lwc
        )
        conftest_boilerplate.check_output(compare_node.ntype, int, 0)

    @pytest.mark.parametrize("arg_height", [0.05, 0.1, 0.5])
    @pytest.mark.parametrize("arg_temperature", [273.16, 270.16, 280.0])
    @pytest.mark.parametrize("arg_lwc", [0.0, 0.5])
    def test_add_fresh_debris(
        self,
        conftest_mock_grid,
        conftest_boilerplate,
        arg_height,
        arg_temperature,
        arg_lwc,
    ):
        """Add fresh debris layer."""

        test_grid = conftest_mock_grid
        test_number_nodes = test_grid.number_nodes
        test_snow = test_grid.get_fresh_snow_props()
        assert isinstance(test_snow, tuple)
        assert all(isinstance(parameter, float) for parameter in test_snow)
        assert test_snow[0] == 0

        test_grid.add_fresh_debris(arg_height, 250.0, arg_temperature, arg_lwc)
        assert test_grid.number_nodes == test_number_nodes + 1
        compare_grid_data = test_grid.grid
        assert isinstance(compare_grid_data[0], BaseNode)
        assert test_grid.get_number_debris_layers() == 1

        fresh_snow = test_grid.get_fresh_snow_props()
        assert isinstance(fresh_snow, tuple)
        assert all(isinstance(parameter, float) for parameter in fresh_snow)
        assert conftest_boilerplate.check_output(fresh_snow[0], float, 0.0)
        compare_node = compare_grid_data[0]
        assert isinstance(compare_node, BaseNode)
        conftest_boilerplate.check_output(compare_node.ntype, int, 1)

        conftest_boilerplate.check_output(
            compare_node.height, float, arg_height
        )
        conftest_boilerplate.check_output(
            compare_node.temperature, float, arg_temperature
        )
        conftest_boilerplate.check_output(
            compare_node.liquid_water_content, float, 0.0  # dry debris
        )


class TestGridRemeshing:
    """Tests if layers can remesh and merge."""

    def test_merge_nodes(
        self,
        conftest_mock_grid,
        conftest_boilerplate,
        conftest_debris_boilerplate,
    ):
        # TODO: Test that debris doesn't merge
        pass

    @pytest.mark.parametrize("arg_bury", [True, False])
    def test_log_profile_debris(
        self, conftest_mock_grid, conftest_boilerplate, arg_bury
    ):
        test_grid = conftest_mock_grid
        test_grid.add_fresh_debris(0.2, 2840.0, 273.15, 0.0)
        compare_grid_data = test_grid.grid
        assert compare_grid_data[0].ntype == 1
        if arg_bury:
            test_grid.add_fresh_snow(0.1, 250.0, 273.15, 0.0)
        test_nodes = test_grid.number_nodes
        test_snowheight = test_grid.get_total_snowheight()
        test_debris_height = test_grid.get_total_debris_height()
        test_total_height = test_grid.get_total_height()
        test_ice_height = (
            test_grid.get_total_height()
            - test_grid.get_total_snowheight()
            - test_grid.get_total_debris_height()
        )

        test_grid.log_profile_debris()

        conftest_boilerplate.check_output(
            test_grid.get_total_height(), float, test_total_height
        )
        assert test_grid.number_nodes > test_nodes
        assert test_grid.get_number_debris_layers() == 1
        if arg_bury:
            assert test_grid.get_node_ntype(0) == 0
        else:
            assert test_grid.get_node_ntype(0) == 1

        assert np.isclose(test_grid.get_total_snowheight(), test_snowheight)
        conftest_boilerplate.check_output(
            test_grid.get_total_debris_height(), float, test_debris_height
        )
        compare_ice_height = (
            test_grid.get_total_height()
            - test_grid.get_total_snowheight()
            - test_grid.get_total_debris_height()
        )
        assert np.isclose(compare_ice_height, test_ice_height)

        for density in test_grid.get_density():
            assert density < 1000

    def test_log_profile_debris_clean_ice(
        self, conftest_mock_grid_values, conftest_boilerplate
    ):
        """Output of `log_profile_debris` on a grid with no debris is
        identical to `log_profile`.
        """

        data = conftest_mock_grid_values.copy()
        grid_sno = cpgrid.Grid(
            layer_heights=data["layer_heights"],
            layer_densities=data["layer_densities"],
            layer_temperatures=data["layer_temperatures"],
            layer_liquid_water_content=data["layer_liquid_water_content"],
        )
        grid_deb = cpgrid.Grid(
            layer_heights=data["layer_heights"],
            layer_densities=data["layer_densities"],
            layer_temperatures=data["layer_temperatures"],
            layer_liquid_water_content=data["layer_liquid_water_content"],
        )

        test_ice_height = (
            grid_sno.get_total_height() - grid_sno.get_total_snowheight()
        )
        grid_sno.log_profile()
        grid_deb.log_profile_debris()

        conftest_boilerplate.assert_grid_profiles_equal(grid_sno, grid_deb)
        compare_ice_height = (
            grid_deb.get_total_height() - grid_deb.get_total_snowheight()
        )
        conftest_boilerplate.check_output(
            compare_ice_height, float, test_ice_height
        )

    @pytest.mark.parametrize("arg_bury", [True, False])
    def test_adaptive_profile_debris(
        self, monkeypatch, conftest_mock_grid, conftest_boilerplate, arg_bury
    ):
        conftest_boilerplate.patch_variable(
            monkeypatch, cpgrid.constants, {"merge_max": 100}
        )
        test_grid = conftest_mock_grid
        test_grid.add_fresh_snow(0.1, 250.0, 273.15, 0.0)
        test_grid.add_fresh_snow(0.1, 250.0, 273.15, 0.0)
        test_grid.add_fresh_debris(0.2, 2840.0, 273.15, 0.0)
        compare_grid_data = test_grid.grid
        assert compare_grid_data[0].ntype == 1
        if arg_bury:
            test_grid.add_fresh_snow(0.1, 250.0, 273.15, 0.0)
        test_nodes = test_grid.number_nodes
        test_snowheight = test_grid.get_total_snowheight()
        test_debris_height = test_grid.get_total_debris_height()
        test_total_height = test_grid.get_total_height()
        test_ice_height = (
            test_grid.get_total_height()
            - test_grid.get_total_snowheight()
            - test_grid.get_total_debris_height()
        )

        test_grid.adaptive_profile_debris()

        conftest_boilerplate.check_output(
            test_grid.get_total_height(), float, test_total_height
        )
        assert test_grid.number_nodes < test_nodes
        assert test_grid.get_number_debris_layers() == 1
        if arg_bury:
            assert test_grid.get_node_ntype(0) == 0
        else:
            assert test_grid.get_node_ntype(0) == 1

        assert np.isclose(test_grid.get_total_snowheight(), test_snowheight)
        conftest_boilerplate.check_output(
            test_grid.get_total_debris_height(), float, test_debris_height
        )
        compare_ice_height = (
            test_grid.get_total_height()
            - test_grid.get_total_snowheight()
            - test_grid.get_total_debris_height()
        )
        assert np.isclose(compare_ice_height, test_ice_height)

        for density in test_grid.get_density():
            assert density < 1000

    def test_adaptive_profile_debris_clean_ice(
        self, monkeypatch, conftest_mock_grid_values, conftest_boilerplate
    ):
        """Output of `adaptive_profile_debris` on a grid with no debris is
        identical to `adaptive_profile`.
        """
        conftest_boilerplate.patch_variable(
            monkeypatch, cpgrid.constants, {"merge_max": 100}
        )
        data = conftest_mock_grid_values.copy()

        grid_sno = cpgrid.Grid(
            layer_heights=data["layer_heights"],
            layer_densities=data["layer_densities"],
            layer_temperatures=data["layer_temperatures"],
            layer_liquid_water_content=data["layer_liquid_water_content"],
        )
        grid_deb = cpgrid.Grid(
            layer_heights=data["layer_heights"],
            layer_densities=data["layer_densities"],
            layer_temperatures=data["layer_temperatures"],
            layer_liquid_water_content=data["layer_liquid_water_content"],
        )

        test_ice_height = (
            grid_sno.get_total_height() - grid_sno.get_total_snowheight()
        )
        grid_sno.adaptive_profile()
        grid_deb.adaptive_profile_debris()

        conftest_boilerplate.assert_grid_profiles_equal(grid_sno, grid_deb)
        compare_ice_height = (
            grid_deb.get_total_height() - grid_sno.get_total_snowheight()
        )
        conftest_boilerplate.check_output(
            compare_ice_height, float, test_ice_height
        )


class TestGridUpdate:
    """Tests update methods for Grid objects."""

    @pytest.mark.parametrize(
        "arg_profile", ["log_profile", "adaptive_profile"]
    )
    def test_grid_update_functions(
        self,
        monkeypatch,
        conftest_mock_grid,
        conftest_boilerplate,
        arg_profile,
    ):
        conftest_boilerplate.patch_variable(
            monkeypatch, cpgrid.constants, {"remesh_method": arg_profile}
        )
        grid = conftest_mock_grid
        grid.set_node_liquid_water_content(0, 0.04)
        grid.set_node_liquid_water_content(1, 0.03)
        grid.set_node_liquid_water_content(2, 0.03)
        grid.set_node_liquid_water_content(3, 0.02)
        grid.set_node_liquid_water_content(4, 0.01)

        swe_before = np.array(grid.get_height()) / np.array(grid.get_density())
        swe_before_sum = np.nansum(swe_before)
        test_surface_height = grid.get_node_height(0)
        test_snowheight = grid.get_total_snowheight(0)
        test_height = grid.get_total_height()

        grid.update_grid_debris()
        swe_after = np.array(grid.get_height()) / np.array(grid.get_density())
        swe_after_sum = np.nansum(swe_after)
        compare_surface_height = grid.get_node_height(0)
        assert compare_surface_height <= test_surface_height
        conftest_boilerplate.check_output(
            grid.get_total_snowheight(), float, test_snowheight
        )
        conftest_boilerplate.check_output(
            grid.get_total_height(), float, test_height
        )
        assert np.allclose(swe_before_sum, swe_after_sum, atol=1e-4)
