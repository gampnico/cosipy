import pytest

import constants
import cosipy.modules.melting as module_melting


class TestParamMelting:
    """Tests radiation methods."""

    @pytest.mark.parametrize("arg_air_temp", [270.15, 273.16])
    def test_get_surface_melt_energy_eti(
        self, arg_air_temp, conftest_boilerplate
    ):
        test_temperature = arg_air_temp
        test_sw = 800.0
        test_albedo = 0.2
        if arg_air_temp > constants.zero_temperature + 1:
            test_melt_energy = (
                constants.temperature_factor * arg_air_temp
                + (1 - test_albedo)
                * test_sw
                * constants.shortwave_radiation_factor
            )
        else:
            test_melt_energy = 0.0
        compare_melt_energy = module_melting.get_surface_melt_energy_eti(
            air_temperature=test_temperature,
            albedo=test_albedo,
            sw_net=test_sw,
        )
        conftest_boilerplate.check_output(
            compare_melt_energy, float, test_melt_energy
        )
        assert compare_melt_energy >= 0.0

    @pytest.mark.parametrize("arg_debris_temp", [270.15, 280.15])
    def test_get_debris_melt_energy_reid(
        self, conftest_mock_grid, arg_debris_temp
    ):
        # TODO: Test for englacial bands?
        test_grid = conftest_mock_grid
        test_grid.add_fresh_debris(0.2, 2840.0, arg_debris_temp, 0.0)

        compare_chf = module_melting.get_debris_melt_energy_reid(
            grid=test_grid
        )

        assert isinstance(compare_chf, float)
        assert compare_chf >= 0.0

    @pytest.mark.parametrize("arg_dt", [1, 60, 3600])
    def test_get_surface_melt_rate(self, conftest_boilerplate, arg_dt):
        test_melt_energy = 1000
        test_density = 1000.0
        test_melt_rate = (
            test_melt_energy
            * arg_dt
            / (test_density * constants.lat_heat_melting)
        )

        compare_melt_rate = module_melting.get_surface_melt_rate(
            melt_energy=test_melt_energy, dt=arg_dt, density=test_density
        )
        conftest_boilerplate.check_output(
            compare_melt_rate, float, test_melt_rate
        )
        assert compare_melt_rate >= 0.0

    @pytest.mark.parametrize("arg_negative", [1, -1])
    def test_get_surface_melt_energy_sum(
        self, conftest_boilerplate, arg_negative
    ):
        test_sw_net = 500.0
        test_lw_in = 500.0
        test_lw_out = 500.0
        test_ghf = 500.0
        test_rhf = 500.0
        test_shf = 500.0
        test_lhf = 500.0

        if arg_negative:
            test_energy = (
                test_sw_net
                + test_lw_in
                + test_lw_out
                + test_ghf
                + test_rhf
                + test_shf
                + test_lhf
            )
        else:
            test_energy = 0.0

        compare_energy = module_melting.get_surface_melt_energy_sum(
            sw_net=test_sw_net,
            lw_in=test_lw_in,
            lw_out=test_lw_out,
            ghf=test_ghf,
            rhf=test_rhf,
            shf=test_shf,
            lhf=test_lhf,
        )

        conftest_boilerplate.check_output(compare_energy, float, test_energy)
