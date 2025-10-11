"""Smoke tests for utility functions."""

import pytest
import numpy as np
from satellite_downlink_simulator.utils import (
    rrc_filter_time,
    rrc_filter_freq,
    generate_constellation,
    watts_to_dbm,
    dbm_to_watts,
    add_measurement_noise,
    validate_positive,
    validate_range
)
from satellite_downlink_simulator.objects.enums import ModulationType


class TestRRCFilters:
    """Test RRC filter generation."""

    def test_rrc_filter_time_basic(self):
        """Test basic RRC filter time-domain generation."""
        h = rrc_filter_time(
            symbol_rate=10e6,
            rolloff=0.35,
            span=10,
            samples_per_symbol=8
        )

        assert h is not None
        assert len(h) > 0
        assert np.all(np.isfinite(h))

    def test_rrc_filter_time_normalized(self):
        """Test that RRC filter is normalized to unit energy."""
        h = rrc_filter_time(
            symbol_rate=10e6,
            rolloff=0.35,
            span=10,
            samples_per_symbol=8
        )

        energy = np.sum(h ** 2)
        assert np.isclose(energy, 1.0, rtol=1e-6)

    def test_rrc_filter_time_invalid_rolloff(self):
        """Test that invalid rolloff raises error."""
        with pytest.raises(ValueError, match="Rolloff must be between 0 and 1"):
            rrc_filter_time(
                symbol_rate=10e6,
                rolloff=1.5,
                span=10,
                samples_per_symbol=8
            )

    def test_rrc_filter_freq_basic(self):
        """Test basic RRC filter frequency-domain generation."""
        frequencies = np.linspace(-20e6, 20e6, 1000)
        H = rrc_filter_freq(
            frequencies=frequencies,
            symbol_rate=10e6,
            rolloff=0.35
        )

        assert H is not None
        assert len(H) == len(frequencies)
        assert np.all(H >= 0)
        assert np.all(H <= 1)

    def test_rrc_filter_freq_passband(self):
        """Test RRC filter passband is unity."""
        frequencies = np.linspace(-3e6, 3e6, 100)
        H = rrc_filter_freq(
            frequencies=frequencies,
            symbol_rate=10e6,
            rolloff=0.35
        )

        # In passband, H should be 1.0
        assert np.allclose(H, 1.0)

    def test_rrc_filter_freq_stopband(self):
        """Test RRC filter stopband is zero."""
        # Well beyond rolloff
        frequencies = np.linspace(20e6, 30e6, 100)
        H = rrc_filter_freq(
            frequencies=frequencies,
            symbol_rate=10e6,
            rolloff=0.35
        )

        # In stopband, H should be 0.0
        assert np.allclose(H, 0.0)

    def test_rrc_filter_freq_invalid_rolloff(self):
        """Test that invalid rolloff raises error."""
        frequencies = np.linspace(-20e6, 20e6, 1000)
        with pytest.raises(ValueError, match="Rolloff must be between 0 and 1"):
            rrc_filter_freq(
                frequencies=frequencies,
                symbol_rate=10e6,
                rolloff=1.5
            )


class TestConstellations:
    """Test constellation generation."""

    def test_constellation_bpsk(self):
        """Test BPSK constellation generation."""
        constellation = generate_constellation(ModulationType.BPSK)
        assert len(constellation) == 2
        assert np.iscomplexobj(constellation)

    def test_constellation_qpsk(self):
        """Test QPSK constellation generation."""
        constellation = generate_constellation(ModulationType.QPSK)
        assert len(constellation) == 4
        assert np.iscomplexobj(constellation)

    def test_constellation_qam16(self):
        """Test 16-QAM constellation generation."""
        constellation = generate_constellation(ModulationType.QAM16)
        assert len(constellation) == 16
        assert np.iscomplexobj(constellation)

    def test_constellation_apsk16(self):
        """Test 16-APSK constellation generation."""
        constellation = generate_constellation(ModulationType.APSK16)
        assert len(constellation) == 16
        assert np.iscomplexobj(constellation)

    def test_constellation_apsk32(self):
        """Test 32-APSK constellation generation."""
        constellation = generate_constellation(ModulationType.APSK32)
        assert len(constellation) == 32
        assert np.iscomplexobj(constellation)

    def test_constellation_normalized(self):
        """Test that constellations are normalized to unit average power."""
        for mod_type in [ModulationType.BPSK, ModulationType.QPSK, ModulationType.QAM16]:
            constellation = generate_constellation(mod_type)
            avg_power = np.mean(np.abs(constellation) ** 2)
            assert np.isclose(avg_power, 1.0, rtol=0.01)

    def test_constellation_invalid_modulation(self):
        """Test that STATIC_CW raises error (no constellation)."""
        with pytest.raises(ValueError, match="Unknown modulation type"):
            generate_constellation(ModulationType.STATIC_CW)


class TestPowerConversions:
    """Test power conversion functions."""

    def test_watts_to_dbm(self):
        """Test Watts to dBm conversion."""
        # 1 mW = 0 dBm
        assert np.isclose(watts_to_dbm(0.001), 0.0)

        # 1 W = 30 dBm
        assert np.isclose(watts_to_dbm(1.0), 30.0)

        # 1 µW = -30 dBm
        assert np.isclose(watts_to_dbm(1e-6), -30.0)

    def test_dbm_to_watts(self):
        """Test dBm to Watts conversion."""
        # 0 dBm = 1 mW = 0.001 W
        assert np.isclose(dbm_to_watts(0.0), 0.001)

        # 30 dBm = 1 W
        assert np.isclose(dbm_to_watts(30.0), 1.0)

        # -30 dBm = 1 µW
        assert np.isclose(dbm_to_watts(-30.0), 1e-6)

    def test_power_conversion_roundtrip(self):
        """Test that conversions are reversible."""
        power_watts = 0.123
        dbm = watts_to_dbm(power_watts)
        power_back = dbm_to_watts(dbm)
        assert np.isclose(power_watts, power_back)


class TestMeasurementNoise:
    """Test measurement noise addition."""

    def test_add_measurement_noise(self):
        """Test adding measurement noise to PSD."""
        psd_clean = np.ones(1000) * 1e-12
        psd_noisy = add_measurement_noise(psd_clean, noise_factor_db=0.5)

        assert len(psd_noisy) == len(psd_clean)
        assert np.all(np.isfinite(psd_noisy))
        # Noisy version should be different from clean
        assert not np.allclose(psd_noisy, psd_clean)

    def test_measurement_noise_increases_variance(self):
        """Test that noise increases variance."""
        psd_clean = np.ones(10000) * 1e-12

        # Small noise
        psd_low_noise = add_measurement_noise(psd_clean, noise_factor_db=0.1)
        var_low = np.var(10 * np.log10(psd_low_noise))

        # Large noise
        psd_high_noise = add_measurement_noise(psd_clean, noise_factor_db=1.0)
        var_high = np.var(10 * np.log10(psd_high_noise))

        # High noise should have higher variance
        assert var_high > var_low


class TestValidationFunctions:
    """Test validation utility functions."""

    def test_validate_positive_accepts_positive(self):
        """Test that validate_positive accepts positive values."""
        # Should not raise
        validate_positive(1.0, "test_value")
        validate_positive(0.001, "test_value")
        validate_positive(1e6, "test_value")

    def test_validate_positive_rejects_zero(self):
        """Test that validate_positive rejects zero."""
        with pytest.raises(ValueError, match="test_value must be positive"):
            validate_positive(0.0, "test_value")

    def test_validate_positive_rejects_negative(self):
        """Test that validate_positive rejects negative values."""
        with pytest.raises(ValueError, match="test_value must be positive"):
            validate_positive(-1.0, "test_value")

    def test_validate_range_accepts_in_range(self):
        """Test that validate_range accepts values in range."""
        # Should not raise
        validate_range(0.5, 0.0, 1.0, "test_value")
        validate_range(0.0, 0.0, 1.0, "test_value")
        validate_range(1.0, 0.0, 1.0, "test_value")

    def test_validate_range_rejects_below(self):
        """Test that validate_range rejects values below range."""
        with pytest.raises(ValueError, match="test_value must be between"):
            validate_range(-0.1, 0.0, 1.0, "test_value")

    def test_validate_range_rejects_above(self):
        """Test that validate_range rejects values above range."""
        with pytest.raises(ValueError, match="test_value must be between"):
            validate_range(1.1, 0.0, 1.0, "test_value")
