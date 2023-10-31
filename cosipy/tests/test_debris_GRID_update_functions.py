import numpy as np
import pytest

from cosipy.cpkernel.grid import Grid
from cosipy.cpkernel.node import BaseNode


class TestGridSetter:
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
        test_snow = test_grid.get_fresh_snow_props()
        assert isinstance(test_snow, tuple)
        assert all(isinstance(parameter, float) for parameter in test_snow)
        assert test_snow[0] == 0

        test_grid.add_fresh_snow(arg_height, 250.0, arg_temperature, arg_lwc)
        assert test_grid.number_nodes == test_number_nodes + 1
        assert isinstance(test_grid.grid[0], BaseNode)

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
        assert isinstance(test_grid.grid[0], BaseNode)
        assert test_grid.get_number_debris_layers() == 1

        fresh_snow = test_grid.get_fresh_snow_props()
        assert isinstance(fresh_snow, tuple)
        assert all(isinstance(parameter, float) for parameter in fresh_snow)
        assert conftest_boilerplate.check_output(fresh_snow[0], float, 0.0)
        compare_node = test_grid.grid[0]
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

    def add_debris_to_grid(self, grid_obj: Grid):
        """Fixture for adding a debris layer."""
        grid_obj.add_fresh_debris(0.2, 2840.0, 273.15, 0.0)

    def test_add_debris_to_grid(self, conftest_mock_grid):
        """Add debris layer using fixture."""
        test_grid = conftest_mock_grid
        test_nodes = test_grid.number_nodes
        self.add_debris_to_grid(grid_obj=test_grid)
        assert test_grid.get_node_ntype(0) == 1
        for i in range(1, test_grid.number_nodes):
            assert test_grid.get_node_ntype(i) == 0
        assert test_grid.number_nodes == test_nodes + 1
        assert test_grid.get_number_debris_layers() == 1

    def get_hydrostatic_pressure(
        self, grid_obj, idx: int = 0, single: bool = False
    ) -> float:
        """Get hydrostatic pressure for two contiguous layers.

        Args:
            grid_obj (Grid): Grid data instance.
            idx: Layer index. Default 0.
            single: Only calculate pressure for a single layer.
                Default `False`.

        Returns:
            Hydrostatic pressure.
        """

        w0 = grid_obj.get_node_height(idx) * grid_obj.get_node_density(idx)
        if not single:
            w0 += grid_obj.get_node_height(
                idx + 1
            ) * grid_obj.get_node_density(idx + 1)

        return 9.81 * w0

    @pytest.mark.parametrize("arg_single", [True, False])
    def test_get_hydrostatic_pressure(
        self, conftest_mock_grid, conftest_boilerplate, arg_single
    ):
        test_grid = conftest_mock_grid
        test_w0 = test_grid.get_node_height(0) * test_grid.get_node_density(0)
        if not arg_single:
            test_w0 += test_grid.get_node_height(
                1
            ) * test_grid.get_node_density(1)
        compare_w0 = self.get_hydrostatic_pressure(
            grid_obj=test_grid, idx=0, single=arg_single
        )
        conftest_boilerplate.check_output(compare_w0, float, 9.81 * test_w0)

    def test_merge_nodes(self, conftest_mock_grid, conftest_boilerplate):
        """TODO: remove as it's tested in test_GRID_update_functions"""
        test_grid = conftest_mock_grid
        ref_nodes = test_grid.number_nodes
        self.add_debris_to_grid(test_grid)
        test_grid.add_fresh_snow(0.1, 250.0, 273.15, 0.0)
        test_nodes = test_grid.number_nodes
        assert test_nodes == ref_nodes + 2

        # snow-snow
        idx = test_nodes - ref_nodes
        test_w0 = self.get_hydrostatic_pressure(test_grid, idx)
        test_height = sum(test_grid.get_height()[idx : idx + 2])

        test_grid.merge_nodes(idx)

        compare_w0 = self.get_hydrostatic_pressure(test_grid, idx, single=True)
        conftest_boilerplate.check_output(compare_w0, float, test_w0)
        conftest_boilerplate.check_output(
            test_grid.get_node_height(idx), float, test_height
        )

        # glacier-glacier
        idx = test_grid.number_nodes - 2  # last two layers are ice
        test_w0 = self.get_hydrostatic_pressure(test_grid, idx)
        test_height = sum(test_grid.get_height()[idx:])

        test_grid.merge_nodes(idx)

        compare_w0 = self.get_hydrostatic_pressure(test_grid, idx, single=True)
        conftest_boilerplate.check_output(compare_w0, float, test_w0)
        conftest_boilerplate.check_output(
            test_grid.get_node_height(idx), float, test_height
        )

    def test_log_profile_debris_clean_ice(
        self, conftest_mock_grid_values, conftest_boilerplate
    ):
        """Log profile on clean-ice glacier is identical."""

        data = conftest_mock_grid_values.copy()
        grid_sno = Grid(
            layer_heights=data["layer_heights"],
            layer_densities=data["layer_densities"],
            layer_temperatures=data["layer_temperatures"],
            layer_liquid_water_content=data["layer_liquid_water_content"],
        )
        grid_deb = Grid(
            layer_heights=data["layer_heights"],
            layer_densities=data["layer_densities"],
            layer_temperatures=data["layer_temperatures"],
            layer_liquid_water_content=data["layer_liquid_water_content"],
        )

        assert grid_sno is not grid_deb  # don't point to same object

        sno_ice_height = (
            grid_sno.get_total_height() - grid_sno.get_total_snowheight()
        )
        grid_sno.log_profile()
        grid_deb.log_profile_debris()

        assert grid_sno.number_nodes == grid_deb.number_nodes
        assert grid_sno.get_number_layers() == grid_deb.get_number_layers()
        assert (
            grid_sno.get_number_snow_layers()
            == grid_deb.get_number_snow_layers()
        )

        conftest_boilerplate.check_output(
            grid_sno.get_total_snowheight(),
            float,
            grid_deb.get_total_snowheight(),
        )
        conftest_boilerplate.check_output(
            grid_sno.get_total_height(), float, grid_deb.get_total_height()
        )
        compare_ice_height = (
            grid_deb.get_total_height() - grid_sno.get_total_snowheight()
        )
        conftest_boilerplate.check_output(
            compare_ice_height, float, sno_ice_height
        )

    @pytest.mark.parametrize("arg_bury", [True, False])
    def test_log_profile_debris(
        self, conftest_mock_grid, conftest_boilerplate, arg_bury
    ):
        test_grid = conftest_mock_grid
        test_grid.add_fresh_debris(0.2, 2840.0, 273.15, 0.0)
        assert test_grid.grid[0].ntype == 1
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


class TestGridUpdate:
    """Tests update methods for Grid objects."""

    def test_grid_update_functions(
        self, conftest_mock_grid, conftest_boilerplate
    ):
        GRID = conftest_mock_grid
        GRID.set_node_liquid_water_content(0, 0.04)
        GRID.set_node_liquid_water_content(1, 0.03)
        GRID.set_node_liquid_water_content(2, 0.03)
        GRID.set_node_liquid_water_content(3, 0.02)
        GRID.set_node_liquid_water_content(4, 0.01)

        SWE_before = np.array(GRID.get_height()) / np.array(GRID.get_density())
        SWE_before_sum = np.nansum(SWE_before)
        test_density = GRID.get_node_density(0)
        test_height = GRID.get_node_height(0)

        GRID.update_grid()
        SWE_after = np.array(GRID.get_height()) / np.array(GRID.get_density())
        SWE_after_sum = np.nansum(SWE_after)
        compare_density = GRID.get_node_density(0)
        compare_height = GRID.get_node_height(0)

        assert compare_height < test_height
        conftest_boilerplate.check_output(compare_density, float, test_density)

        GRID.adaptive_profile()
        adaptive_density = GRID.get_node_density(0)
        adaptive_height = GRID.get_node_height(0)

        assert adaptive_height <= compare_height
        conftest_boilerplate.check_output(
            adaptive_density, float, compare_density
        )

        SWE_after_adaptive = np.array(GRID.get_height()) / np.array(
            GRID.get_density()
        )
        SWE_after_adaptive_sum = np.nansum(SWE_after_adaptive)

        assert np.allclose(SWE_before_sum, SWE_after_sum, atol=1e-3)
        assert np.allclose(SWE_after_sum, SWE_after_adaptive_sum, atol=1e-3)
