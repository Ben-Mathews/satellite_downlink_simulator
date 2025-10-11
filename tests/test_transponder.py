"""Smoke tests for Transponder class."""

import pytest
import numpy as np
from satellite_downlink_simulator.objects.transponder import Transponder
from satellite_downlink_simulator.objects.carrier import Carrier
from satellite_downlink_simulator.objects.enums import CarrierType, ModulationType


class TestTransponderInstantiation:
    """Test transponder instantiation."""

    def test_create_simple_transponder(self, simple_transponder):
        """Test creating a basic transponder."""
        assert simple_transponder is not None
        assert simple_transponder.center_frequency_hz == 12.5e9
        assert simple_transponder.bandwidth_hz == 36e6
        assert len(simple_transponder.carriers) == 0

    def test_create_transponder_with_carriers(self, transponder_with_carriers):
        """Test creating a transponder with carriers."""
        assert transponder_with_carriers is not None
        assert len(transponder_with_carriers.carriers) == 2


class TestTransponderValidation:
    """Test transponder parameter validation."""

    def test_positive_center_frequency(self):
        """Test that center_frequency_hz must be positive."""
        with pytest.raises(ValueError, match="center_frequency_hz must be positive"):
            Transponder(
                center_frequency_hz=0.0,
                bandwidth_hz=36e6,
                noise_power_density_watts_per_hz=1e-15
            )

    def test_positive_bandwidth(self):
        """Test that bandwidth_hz must be positive."""
        with pytest.raises(ValueError, match="bandwidth_hz must be positive"):
            Transponder(
                center_frequency_hz=12.5e9,
                bandwidth_hz=0.0,
                noise_power_density_watts_per_hz=1e-15
            )

    def test_positive_noise_density(self):
        """Test that noise_power_density_watts_per_hz must be positive."""
        with pytest.raises(ValueError, match="noise_power_density_watts_per_hz must be positive"):
            Transponder(
                center_frequency_hz=12.5e9,
                bandwidth_hz=36e6,
                noise_power_density_watts_per_hz=0.0
            )


class TestTransponderCarrierManagement:
    """Test adding and validating carriers."""

    def test_add_carrier(self, simple_transponder, simple_fdma_carrier):
        """Test adding a carrier to transponder."""
        simple_transponder.add_carrier(simple_fdma_carrier)
        assert len(simple_transponder.carriers) == 1
        assert simple_transponder.carriers[0] == simple_fdma_carrier

    def test_carrier_exceeds_bandwidth(self, simple_transponder):
        """Test that carrier exceeding transponder bandwidth is rejected."""
        # Create carrier wider than transponder
        wide_carrier = Carrier(
            frequency_offset_hz=0.0,
            cn_db=15.0,
            symbol_rate_sps=50e6,  # 67.5 MHz with rolloff > 36 MHz transponder
            modulation=ModulationType.QPSK,
            carrier_type=CarrierType.FDMA,
            rrc_rolloff=0.35
        )

        with pytest.raises(ValueError, match="Carrier.*extends beyond transponder bandwidth"):
            simple_transponder.add_carrier(wide_carrier)

    def test_carrier_overlaps_rejected(self, simple_transponder):
        """Test that overlapping carriers are rejected by default."""
        carrier1 = Carrier(
            frequency_offset_hz=0.0,
            cn_db=15.0,
            symbol_rate_sps=10e6,
            modulation=ModulationType.QPSK,
            carrier_type=CarrierType.FDMA,
            rrc_rolloff=0.35
        )
        simple_transponder.add_carrier(carrier1)

        # Create overlapping carrier
        carrier2 = Carrier(
            frequency_offset_hz=5e6,  # Overlaps with carrier1
            cn_db=15.0,
            symbol_rate_sps=10e6,
            modulation=ModulationType.QPSK,
            carrier_type=CarrierType.FDMA,
            rrc_rolloff=0.35
        )

        with pytest.raises(ValueError, match="Carriers.*overlap"):
            simple_transponder.add_carrier(carrier2)

    def test_carrier_overlaps_allowed(self):
        """Test that overlapping carriers can be allowed."""
        # Create transponder with allow_overlap=True
        transponder = Transponder(
            center_frequency_hz=12.5e9,
            bandwidth_hz=36e6,
            noise_power_density_watts_per_hz=1e-15,
            allow_overlap=True
        )

        carrier1 = Carrier(
            frequency_offset_hz=0.0,
            cn_db=15.0,
            symbol_rate_sps=10e6,
            modulation=ModulationType.QPSK,
            carrier_type=CarrierType.FDMA,
            rrc_rolloff=0.35
        )
        transponder.add_carrier(carrier1)

        # Create overlapping carrier
        carrier2 = Carrier(
            frequency_offset_hz=5e6,
            cn_db=15.0,
            symbol_rate_sps=10e6,
            modulation=ModulationType.QPSK,
            carrier_type=CarrierType.FDMA,
            rrc_rolloff=0.35
        )

        transponder.add_carrier(carrier2)
        assert len(transponder.carriers) == 2


class TestTransponderRandomCarrierPopulation:
    """Test random carrier population."""

    def test_populate_with_random_carriers(self, simple_transponder, random_seed):
        """Test populating transponder with random carriers."""
        num_requested = 5
        num_created = simple_transponder.populate_with_random_carriers(
            num_carriers=num_requested,
            seed=random_seed
        )

        assert num_created > 0
        assert num_created <= num_requested
        assert len(simple_transponder.carriers) == num_created

    def test_populate_validates_carriers_fit(self, simple_transponder, random_seed):
        """Test that populated carriers fit within transponder."""
        simple_transponder.populate_with_random_carriers(num_carriers=5, seed=random_seed)

        for carrier in simple_transponder.carriers:
            lower_edge = carrier.frequency_offset_hz - carrier.bandwidth_hz / 2
            upper_edge = carrier.frequency_offset_hz + carrier.bandwidth_hz / 2
            assert lower_edge >= -simple_transponder.bandwidth_hz / 2
            assert upper_edge <= simple_transponder.bandwidth_hz / 2

    def test_populate_carriers_do_not_overlap(self, simple_transponder, random_seed):
        """Test that populated carriers don't overlap."""
        simple_transponder.populate_with_random_carriers(num_carriers=5, seed=random_seed)

        # Check all pairs of carriers
        carriers = simple_transponder.carriers
        for i, carrier1 in enumerate(carriers):
            for carrier2 in carriers[i+1:]:
                lower1 = carrier1.frequency_offset_hz - carrier1.bandwidth_hz / 2
                upper1 = carrier1.frequency_offset_hz + carrier1.bandwidth_hz / 2
                lower2 = carrier2.frequency_offset_hz - carrier2.bandwidth_hz / 2
                upper2 = carrier2.frequency_offset_hz + carrier2.bandwidth_hz / 2

                # Carriers should not overlap
                assert upper1 <= lower2 or upper2 <= lower1


class TestTransponderProperties:
    """Test transponder property calculations."""

    def test_frequency_range(self, simple_transponder):
        """Test frequency range calculation."""
        lower = simple_transponder.lower_frequency_hz
        upper = simple_transponder.upper_frequency_hz
        expected_lower = simple_transponder.center_frequency_hz - simple_transponder.bandwidth_hz / 2
        expected_upper = simple_transponder.center_frequency_hz + simple_transponder.bandwidth_hz / 2
        assert lower == expected_lower
        assert upper == expected_upper

    def test_total_carrier_power(self, transponder_with_carriers):
        """Test total carrier power calculation."""
        total_power = transponder_with_carriers.total_carrier_power_watts
        assert total_power > 0

        # Verify it's the sum of individual carrier average powers
        expected_total = sum(
            c.calculate_average_power_watts(transponder_with_carriers.noise_power_density_watts_per_hz)
            for c in transponder_with_carriers.carriers
        )
        assert np.isclose(total_power, expected_total)


class TestTransponderStringRepresentation:
    """Test transponder string representation."""

    def test_str_transponder(self, simple_transponder):
        """Test string representation of transponder."""
        transponder_str = str(simple_transponder)
        assert "Transponder" in transponder_str
        assert "12.5" in transponder_str  # Frequency in GHz
        assert "36" in transponder_str    # Bandwidth in MHz

    def test_str_with_carriers(self, transponder_with_carriers):
        """Test string representation includes carrier count."""
        transponder_str = str(transponder_with_carriers)
        assert "2 carrier(s)" in transponder_str
