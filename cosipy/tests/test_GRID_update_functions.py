import numpy as np
import pytest

# import constants
from cosipy.cpkernel.node import Node
import cosipy.cpkernel.grid as cpgrid


class TestGridUpdate:
    """Tests update methods for Grid objects."""

    def test_grid_set_node_height(
        self, conftest_mock_grid, conftest_boilerplate
    ):
        GRID = conftest_mock_grid
        test_height = GRID.get_node_height(0)
        GRID.set_node_height(0, test_height + 0.5)
        compare_height = GRID.get_node_height(0)
        conftest_boilerplate.check_output(
            compare_height, float, test_height + 0.5
        )

    def test_grid_set_node_temperature(
        self, conftest_mock_grid, conftest_boilerplate
    ):
        GRID = conftest_mock_grid
        test_temperature = GRID.get_node_temperature(0)
        GRID.set_node_temperature(0, test_temperature + 0.5)
        compare_temperature = GRID.get_node_temperature(0)
        conftest_boilerplate.check_output(
            compare_temperature, float, test_temperature + 0.5
        )

    def test_grid_set_node_ice_fraction(
        self, conftest_mock_grid, conftest_boilerplate
    ):
        GRID = conftest_mock_grid
        test_fraction = GRID.get_node_ice_fraction(0)
        GRID.set_node_ice_fraction(0, test_fraction + 0.5)
        compare_fraction = GRID.get_node_ice_fraction(0)
        conftest_boilerplate.check_output(
            compare_fraction, float, test_fraction + 0.5
        )

    def test_grid_set_node_liquid_water_content(
        self, conftest_mock_grid, conftest_boilerplate
    ):
        GRID = conftest_mock_grid
        test_lwc = GRID.get_node_liquid_water_content(0)
        GRID.set_node_liquid_water_content(0, test_lwc + 0.5)
        compare_lwc = GRID.get_node_liquid_water_content(0)
        conftest_boilerplate.check_output(compare_lwc, float, test_lwc + 0.5)

    def test_grid_set_node_refreeze(
        self, conftest_mock_grid, conftest_boilerplate
    ):
        GRID = conftest_mock_grid
        test_refreeze = GRID.get_node_liquid_water_content(0)
        GRID.set_node_liquid_water_content(0, test_refreeze + 0.5)
        compare_refreeze = GRID.get_node_liquid_water_content(0)
        conftest_boilerplate.check_output(
            compare_refreeze, float, test_refreeze + 0.5
        )

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
        GRID = conftest_mock_grid
        GRID.set_node_liquid_water_content(0, 0.04)
        GRID.set_node_liquid_water_content(1, 0.03)
        GRID.set_node_liquid_water_content(2, 0.03)
        GRID.set_node_liquid_water_content(3, 0.02)
        GRID.set_node_liquid_water_content(4, 0.01)

        SWE_before = np.array(GRID.get_height()) / np.array(GRID.get_density())
        SWE_before_sum = np.nansum(SWE_before)
        test_surface_height = GRID.get_node_height(0)
        test_snowheight = GRID.get_total_snowheight(0)
        test_height = GRID.get_total_height()

        GRID.update_grid()
        SWE_after = np.array(GRID.get_height()) / np.array(GRID.get_density())
        SWE_after_sum = np.nansum(SWE_after)
        compare_surface_height = GRID.get_node_height(0)
        assert compare_surface_height <= test_surface_height
        conftest_boilerplate.check_output(
            GRID.get_total_snowheight(), float, test_snowheight
        )
        conftest_boilerplate.check_output(
            GRID.get_total_height(), float, test_height
        )
        assert np.allclose(SWE_before_sum, SWE_after_sum, atol=1e-4)


class TestGridInteractions:
    """Tests remeshing and interactions between layers."""

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
        test_snow = test_grid.get_fresh_snow_props()
        assert isinstance(test_snow, tuple)
        assert all(isinstance(parameter, float) for parameter in test_snow)
        assert test_snow[0] == 0

        test_grid.add_fresh_snow(arg_height, 250.0, arg_temperature, arg_lwc)
        assert test_grid.number_nodes == test_number_nodes + 1
        assert isinstance(test_grid.grid[0], Node)

        fresh_snow = test_grid.get_fresh_snow_props()
        assert isinstance(fresh_snow, tuple)
        assert all(isinstance(parameter, float) for parameter in fresh_snow)
        assert conftest_boilerplate.check_output(
            fresh_snow[0], float, arg_height
        )
        assert not np.isclose(fresh_snow[0], test_snow[0])

        compare_node = test_grid.grid[0]
        conftest_boilerplate.check_output(
            compare_node.height, float, arg_height
        )
        conftest_boilerplate.check_output(
            compare_node.temperature, float, arg_temperature
        )
        conftest_boilerplate.check_output(
            compare_node.liquid_water_content, float, arg_lwc
        )

    @pytest.mark.parametrize("arg_idx", [None, [-1], [0, 1, 2]])
    def test_grid_remove_node(
        self, conftest_mock_grid_values, conftest_mock_grid, arg_idx
    ):
        """Remove node from grid with or without indices."""

        data = conftest_mock_grid_values.copy()
        GRID = conftest_mock_grid
        if not arg_idx:
            indices = [0]
        else:
            indices = arg_idx
        assert isinstance(indices, list)

        GRID.remove_melt_weq(0.01)
        number_nodes_before = GRID.get_number_layers()
        GRID.remove_node(arg_idx)  # Remove node

        assert GRID.get_number_layers() == number_nodes_before - len(indices)
        assert np.isclose(  # matches new density
            np.nanmean(GRID.get_density()),
            np.nanmean(np.delete(data["layer_densities"], indices)),
        )


class TestGridRemeshing:
    """Tests if layers can remesh and merge."""

    def get_overburden_pressure(
        self, grid_obj, idx: int = 0, single: bool = False
    ):
        """Get overburden pressure for two contiguous layers.

        Args:
            grid_obj (Grid): Grid data instance.
            idx: Layer index. Default 0.
            single: Only calculate pressure for a single layer.
                Default `False`.
        """

        w0 = grid_obj.get_node_height(idx) * grid_obj.get_node_density(idx)
        if not single:
            w0 += grid_obj.get_node_height(
                idx + 1
            ) * grid_obj.get_node_density(idx + 1)

        return w0

    @pytest.mark.parametrize("arg_single", [True, False])
    def test_get_overburden_pressure(
        self, conftest_mock_grid, conftest_boilerplate, arg_single
    ):
        test_grid = conftest_mock_grid
        test_w0 = test_grid.get_node_height(0) * test_grid.get_node_density(0)
        if not arg_single:
            test_w0 += test_grid.get_node_height(
                1
            ) * test_grid.get_node_density(1)
        compare_w0 = self.get_overburden_pressure(
            grid_obj=test_grid, idx=0, single=arg_single
        )
        conftest_boilerplate.check_output(compare_w0, float, test_w0)

    def test_merge_nodes(self, conftest_mock_grid, conftest_boilerplate):
        test_grid = conftest_mock_grid
        test_nodes = test_grid.number_nodes

        # snow-snow
        idx = test_nodes
        test_w0 = self.get_overburden_pressure(test_grid, idx)
        test_height = sum(test_grid.get_height()[idx : idx + 2])

        test_grid.merge_nodes(idx)

        compare_w0 = self.get_overburden_pressure(test_grid, idx, single=True)
        conftest_boilerplate.check_output(compare_w0, float, test_w0)
        conftest_boilerplate.check_output(
            test_grid.get_node_height(idx), float, test_height
        )

        # glacier-glacier
        idx = test_grid.number_nodes - 2  # last two layers are ice
        test_w0 = self.get_overburden_pressure(test_grid, idx)
        test_height = sum(test_grid.get_height()[idx:])

        test_grid.merge_nodes(idx)

        compare_w0 = self.get_overburden_pressure(test_grid, idx, single=True)
        conftest_boilerplate.check_output(compare_w0, float, test_w0)
        conftest_boilerplate.check_output(
            test_grid.get_node_height(idx), float, test_height
        )

    def test_log_profile(self, conftest_mock_grid, conftest_boilerplate):
        test_grid = conftest_mock_grid
        test_nodes = test_grid.number_nodes
        test_snowheight = test_grid.get_total_snowheight()
        test_total_height = test_grid.get_total_height()
        test_ice_height = (
            test_grid.get_total_height() - test_grid.get_total_snowheight()
        )

        test_grid.log_profile()
        assert test_grid.number_nodes > test_nodes
        assert test_grid.get_number_layers() == test_grid.number_nodes
        assert test_grid.get_number_snow_layers() == 14
        conftest_boilerplate.check_output(
            test_grid.get_total_snowheight(), float, test_snowheight
        )
        conftest_boilerplate.check_output(
            test_grid.get_total_height(), float, test_total_height
        )
        compare_ice_height = (
            test_grid.get_total_height() - test_grid.get_total_snowheight()
        )
        conftest_boilerplate.check_output(
            compare_ice_height, float, test_ice_height
        )
