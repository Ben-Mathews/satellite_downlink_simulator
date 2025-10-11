"""Smoke tests for Carrier class."""

import pytest
import numpy as np
from satellite_downlink_simulator.objects.carrier import Carrier, STATIC_CW_BANDWIDTH_HZ
from satellite_downlink_simulator.objects.enums import CarrierType, ModulationType


class TestCarrierInstantiation:
    """Test carrier instantiation with various configurations."""

    def test_create_fdma_carrier(self, simple_fdma_carrier):
        """Test creating a basic FDMA carrier."""
        assert simple_fdma_carrier is not None
        assert simple_fdma_carrier.carrier_type == CarrierType.FDMA
        assert simple_fdma_carrier.modulation == ModulationType.QPSK

    def test_create_tdma_carrier(self, simple_tdma_carrier):
        """Test creating a basic TDMA carrier."""
        assert simple_tdma_carrier is not None
        assert simple_tdma_carrier.carrier_type == CarrierType.TDMA
        assert simple_tdma_carrier.burst_time_s is not None
        assert simple_tdma_carrier.duty_cycle is not None

    def test_create_static_cw_carrier(self, static_cw_carrier):
        """Test creating a STATIC_CW carrier."""
        assert static_cw_carrier is not None
        assert static_cw_carrier.modulation == ModulationType.STATIC_CW
        assert static_cw_carrier.symbol_rate_sps is None


class TestCarrierValidation:
    """Test carrier parameter validation."""

    def test_static_cw_without_symbol_rate(self):
        """Test that STATIC_CW carriers work without symbol_rate_sps."""
        carrier = Carrier(
            frequency_offset_hz=0.0,
            cn_db=15.0,
            modulation=ModulationType.STATIC_CW,
            carrier_type=CarrierType.FDMA
        )
        assert carrier.symbol_rate_sps is None

    def test_static_cw_rejects_symbol_rate(self):
        """Test that STATIC_CW carriers reject symbol_rate_sps."""
        with pytest.raises(ValueError, match="STATIC_CW carriers should not specify symbol_rate_sps"):
            Carrier(
                frequency_offset_hz=0.0,
                cn_db=15.0,
                symbol_rate_sps=10e6,
                modulation=ModulationType.STATIC_CW,
                carrier_type=CarrierType.FDMA
            )

    def test_modulated_carrier_requires_symbol_rate(self):
        """Test that modulated carriers require symbol_rate_sps."""
        with pytest.raises(ValueError, match="Modulated carriers .* must specify symbol_rate_sps"):
            Carrier(
                frequency_offset_hz=0.0,
                cn_db=15.0,
                modulation=ModulationType.QPSK,
                carrier_type=CarrierType.FDMA
            )

    def test_tdma_requires_burst_parameters(self):
        """Test that TDMA carriers require burst_time_s and duty_cycle."""
        with pytest.raises(ValueError, match="TDMA carriers must specify both burst_time_s and duty_cycle"):
            Carrier(
                frequency_offset_hz=0.0,
                cn_db=15.0,
                symbol_rate_sps=10e6,
                modulation=ModulationType.QPSK,
                carrier_type=CarrierType.TDMA
            )

    def test_fdma_rejects_burst_parameters(self):
        """Test that FDMA carriers reject burst parameters."""
        with pytest.raises(ValueError, match="FDMA carriers should not specify burst_time_s or duty_cycle"):
            Carrier(
                frequency_offset_hz=0.0,
                cn_db=15.0,
                symbol_rate_sps=10e6,
                modulation=ModulationType.QPSK,
                carrier_type=CarrierType.FDMA,
                burst_time_s=0.001,
                duty_cycle=0.3
            )

    def test_positive_cn_db(self):
        """Test that cn_db must be positive."""
        with pytest.raises(ValueError, match="cn_db must be positive"):
            Carrier(
                frequency_offset_hz=0.0,
                cn_db=0.0,
                symbol_rate_sps=10e6,
                modulation=ModulationType.QPSK,
                carrier_type=CarrierType.FDMA
            )

    def test_positive_symbol_rate(self):
        """Test that symbol_rate_sps must be positive."""
        with pytest.raises(ValueError, match="symbol_rate_sps must be positive"):
            Carrier(
                frequency_offset_hz=0.0,
                cn_db=15.0,
                symbol_rate_sps=0.0,
                modulation=ModulationType.QPSK,
                carrier_type=CarrierType.FDMA
            )

    def test_rrc_rolloff_range(self):
        """Test that rrc_rolloff must be between 0 and 1."""
        with pytest.raises(ValueError, match="rrc_rolloff must be between 0.0 and 1.0"):
            Carrier(
                frequency_offset_hz=0.0,
                cn_db=15.0,
                symbol_rate_sps=10e6,
                modulation=ModulationType.QPSK,
                carrier_type=CarrierType.FDMA,
                rrc_rolloff=1.5
            )


class TestCarrierProperties:
    """Test carrier property calculations."""

    def test_bandwidth_modulated_carrier(self, simple_fdma_carrier):
        """Test bandwidth calculation for modulated carrier."""
        expected_bw = simple_fdma_carrier.symbol_rate_sps * (1 + simple_fdma_carrier.rrc_rolloff)
        assert simple_fdma_carrier.bandwidth_hz == expected_bw

    def test_bandwidth_static_cw(self, static_cw_carrier):
        """Test bandwidth for STATIC_CW carrier."""
        assert static_cw_carrier.bandwidth_hz == STATIC_CW_BANDWIDTH_HZ

    def test_frame_period_tdma(self, simple_tdma_carrier):
        """Test frame period calculation for TDMA carrier."""
        expected_period = simple_tdma_carrier.burst_time_s / simple_tdma_carrier.duty_cycle
        assert simple_tdma_carrier.frame_period_s == expected_period

    def test_frame_period_fdma(self, simple_fdma_carrier):
        """Test frame period is None for FDMA carrier."""
        assert simple_fdma_carrier.frame_period_s is None

    def test_guard_time_tdma(self, simple_tdma_carrier):
        """Test guard time calculation for TDMA carrier."""
        expected_guard = simple_tdma_carrier.frame_period_s - simple_tdma_carrier.burst_time_s
        assert simple_tdma_carrier.guard_time_s == expected_guard

    def test_guard_time_fdma(self, simple_fdma_carrier):
        """Test guard time is None for FDMA carrier."""
        assert simple_fdma_carrier.guard_time_s is None


class TestCarrierPowerCalculations:
    """Test carrier power calculation methods."""

    def test_calculate_power_watts(self, simple_fdma_carrier):
        """Test power calculation from C/N."""
        noise_density = 1e-15  # W/Hz
        power = simple_fdma_carrier.calculate_power_watts(noise_density)

        # Verify power is positive
        assert power > 0

        # Verify power calculation formula
        noise_power = noise_density * simple_fdma_carrier.bandwidth_hz
        cn_linear = 10 ** (simple_fdma_carrier.cn_db / 10)
        expected_power = cn_linear * noise_power
        assert np.isclose(power, expected_power)

    def test_calculate_average_power_fdma(self, simple_fdma_carrier):
        """Test average power equals peak power for FDMA."""
        noise_density = 1e-15
        peak_power = simple_fdma_carrier.calculate_power_watts(noise_density)
        avg_power = simple_fdma_carrier.calculate_average_power_watts(noise_density)
        assert avg_power == peak_power

    def test_calculate_average_power_tdma(self, simple_tdma_carrier):
        """Test average power is scaled by duty cycle for TDMA."""
        noise_density = 1e-15
        peak_power = simple_tdma_carrier.calculate_power_watts(noise_density)
        avg_power = simple_tdma_carrier.calculate_average_power_watts(noise_density)
        expected_avg = peak_power * simple_tdma_carrier.duty_cycle
        assert np.isclose(avg_power, expected_avg)


class TestCarrierStringRepresentation:
    """Test carrier string representation."""

    def test_str_fdma_carrier(self, simple_fdma_carrier):
        """Test string representation of FDMA carrier."""
        carrier_str = str(simple_fdma_carrier)
        assert "FDMA" in carrier_str
        assert "QPSK" in carrier_str
        assert simple_fdma_carrier.name in carrier_str

    def test_str_tdma_carrier(self, simple_tdma_carrier):
        """Test string representation of TDMA carrier."""
        carrier_str = str(simple_tdma_carrier)
        assert "TDMA" in carrier_str
        assert "DC=" in carrier_str  # Duty cycle

    def test_str_static_cw_carrier(self, static_cw_carrier):
        """Test string representation of STATIC_CW carrier."""
        carrier_str = str(static_cw_carrier)
        assert "STATIC_CW" in carrier_str
        # Should NOT include symbol rate
        assert "SR=" not in carrier_str
