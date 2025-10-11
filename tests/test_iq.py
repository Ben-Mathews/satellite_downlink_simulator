"""Smoke tests for IQ generation."""

import pytest
import numpy as np
from satellite_downlink_simulator.simulation.iq import generate_iq


class TestIQGeneration:
    """Test IQ generation functionality."""

    def test_generate_iq_single_carrier(self, simple_transponder, simple_fdma_carrier):
        """Test generating IQ data for single carrier."""
        simple_transponder.add_carrier(simple_fdma_carrier)

        iq, metadata = generate_iq(
            simple_transponder,
            duration_s=0.001
        )

        assert iq is not None
        assert metadata is not None
        assert len(iq) > 0
        assert iq.dtype == np.complex128

    def test_generate_iq_multiple_carriers(self, transponder_with_carriers):
        """Test generating IQ data for multiple carriers."""
        iq, metadata = generate_iq(
            transponder_with_carriers,
            duration_s=0.001
        )

        assert iq is not None
        assert len(iq) > 0
        assert iq.dtype == np.complex128

    def test_generate_iq_beam(self, beam_with_transponders):
        """Test generating IQ data for beam."""
        iq, metadata = generate_iq(
            beam_with_transponders,
            duration_s=0.001
        )

        assert iq is not None
        assert len(iq) > 0

    def test_generate_iq_static_cw(self, simple_transponder, static_cw_carrier):
        """Test generating IQ data with STATIC_CW carrier."""
        simple_transponder.add_carrier(static_cw_carrier)

        iq, metadata = generate_iq(
            simple_transponder,
            duration_s=0.001
        )

        assert iq is not None
        assert len(iq) > 0

    def test_generate_iq_tdma(self, simple_transponder, simple_tdma_carrier):
        """Test generating IQ data with TDMA carrier."""
        simple_transponder.add_carrier(simple_tdma_carrier)

        iq, metadata = generate_iq(
            simple_transponder,
            duration_s=0.01  # Longer duration to capture multiple bursts
        )

        assert iq is not None
        assert len(iq) > 0


class TestIQMetadata:
    """Test IQ metadata generation."""

    def test_iq_metadata_fields(self, transponder_with_carriers):
        """Test that IQ metadata contains correct fields."""
        duration = 0.001
        iq, metadata = generate_iq(
            transponder_with_carriers,
            duration_s=duration
        )

        assert metadata.duration_s == duration
        assert metadata.num_samples == len(iq)
        assert metadata.sample_rate_hz > 0
        assert metadata.center_frequency_hz == transponder_with_carriers.center_frequency_hz

    def test_iq_metadata_sample_rate(self, transponder_with_carriers):
        """Test that sample rate is correctly calculated."""
        iq, metadata = generate_iq(
            transponder_with_carriers,
            duration_s=0.001
        )

        # Sample rate should be > bandwidth (1.25x by default)
        assert metadata.sample_rate_hz >= metadata.bandwidth_hz

    def test_iq_metadata_num_samples(self, transponder_with_carriers):
        """Test that number of samples matches duration and sample rate."""
        duration = 0.001
        iq, metadata = generate_iq(
            transponder_with_carriers,
            duration_s=duration
        )

        expected_samples = int(metadata.sample_rate_hz * duration)
        assert metadata.num_samples == expected_samples
        assert len(iq) == expected_samples


class TestIQOutputFormat:
    """Test IQ output format."""

    def test_iq_complex_values(self, transponder_with_carriers):
        """Test that IQ data contains complex values."""
        iq, metadata = generate_iq(
            transponder_with_carriers,
            duration_s=0.001
        )

        # IQ should be complex
        assert np.iscomplexobj(iq)

    def test_iq_array_finite(self, transponder_with_carriers):
        """Test that IQ array contains finite values."""
        iq, metadata = generate_iq(
            transponder_with_carriers,
            duration_s=0.001
        )

        # All IQ values should be finite (no NaN or Inf)
        assert np.all(np.isfinite(iq))

    def test_iq_power_reasonable(self, transponder_with_carriers):
        """Test that IQ data has reasonable power levels."""
        iq, metadata = generate_iq(
            transponder_with_carriers,
            duration_s=0.001
        )

        # Calculate power
        power = np.mean(np.abs(iq) ** 2)
        assert power > 0
        assert np.isfinite(power)

    def test_iq_length_matches_duration(self, transponder_with_carriers):
        """Test that IQ length matches requested duration."""
        duration = 0.001
        iq, metadata = generate_iq(
            transponder_with_carriers,
            duration_s=duration
        )

        # Number of samples should match duration * sample_rate
        expected_length = int(metadata.sample_rate_hz * duration)
        assert len(iq) == expected_length


class TestIQTDMABursting:
    """Test TDMA bursting in IQ data."""

    def test_tdma_bursting_visible(self, simple_transponder, simple_tdma_carrier):
        """Test that TDMA bursting creates power variations."""
        simple_transponder.add_carrier(simple_tdma_carrier)

        # Generate longer duration to capture multiple bursts
        duration = 0.01
        iq, metadata = generate_iq(
            simple_transponder,
            duration_s=duration
        )

        # Calculate instantaneous power
        power = np.abs(iq) ** 2

        # TDMA should have periods of high and low power
        max_power = np.max(power)
        min_power = np.min(power)

        # There should be significant power variation
        assert max_power > min_power * 10  # At least 10x variation

    def test_tdma_duty_cycle(self, simple_transponder, simple_tdma_carrier):
        """Test that TDMA duty cycle is approximately correct."""
        simple_transponder.add_carrier(simple_tdma_carrier)

        # Generate multiple frame periods
        duration = 0.05
        iq, metadata = generate_iq(
            simple_transponder,
            duration_s=duration
        )

        # Calculate power and threshold
        power = np.abs(iq) ** 2
        threshold = np.max(power) * 0.1

        # Count samples above threshold (burst periods)
        burst_samples = np.sum(power > threshold)
        duty_cycle_measured = burst_samples / len(power)

        # Should be within 20% of specified duty cycle
        expected_duty_cycle = simple_tdma_carrier.duty_cycle
        assert abs(duty_cycle_measured - expected_duty_cycle) < 0.2
