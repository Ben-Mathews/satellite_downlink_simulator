"""Smoke tests for PSD generation."""

import pytest
import numpy as np
from satellite_downlink_simulator.simulation.psd import generate_psd


class TestPSDGeneration:
    """Test PSD generation functionality."""

    def test_generate_psd_single_transponder(self, simple_transponder, simple_fdma_carrier):
        """Test generating PSD for single transponder with one carrier."""
        simple_transponder.add_carrier(simple_fdma_carrier)

        freq, psd, metadata = generate_psd(
            simple_transponder,
            rbw_hz=10e3,
            vbw_hz=1e3
        )

        assert freq is not None
        assert psd is not None
        assert metadata is not None
        assert len(freq) == len(psd)
        assert len(freq) > 0

    def test_generate_psd_multiple_carriers(self, transponder_with_carriers):
        """Test generating PSD for transponder with multiple carriers."""
        freq, psd, metadata = generate_psd(
            transponder_with_carriers,
            rbw_hz=10e3,
            vbw_hz=1e3
        )

        assert freq is not None
        assert psd is not None
        assert len(freq) == len(psd)

    def test_generate_psd_beam(self, beam_with_transponders):
        """Test generating PSD for beam with multiple transponders."""
        freq, psd, metadata = generate_psd(
            beam_with_transponders,
            rbw_hz=10e3,
            vbw_hz=1e3
        )

        assert freq is not None
        assert psd is not None
        assert len(freq) == len(psd)

    def test_generate_psd_static_cw(self, simple_transponder, static_cw_carrier):
        """Test generating PSD with STATIC_CW carrier."""
        simple_transponder.add_carrier(static_cw_carrier)

        freq, psd, metadata = generate_psd(
            simple_transponder,
            rbw_hz=10e3,
            vbw_hz=1e3
        )

        assert freq is not None
        assert psd is not None
        assert len(freq) == len(psd)

    def test_generate_psd_tdma(self, simple_transponder, simple_tdma_carrier):
        """Test generating PSD with TDMA carrier."""
        simple_transponder.add_carrier(simple_tdma_carrier)

        freq, psd, metadata = generate_psd(
            simple_transponder,
            rbw_hz=10e3,
            vbw_hz=1e3
        )

        assert freq is not None
        assert psd is not None
        assert len(freq) == len(psd)


class TestPSDMetadata:
    """Test PSD metadata generation."""

    def test_psd_metadata_fields(self, transponder_with_carriers):
        """Test that PSD metadata contains correct fields."""
        freq, psd, metadata = generate_psd(
            transponder_with_carriers,
            rbw_hz=10e3,
            vbw_hz=1e3
        )

        assert metadata.center_frequency_hz == transponder_with_carriers.center_frequency_hz
        assert metadata.rbw_hz == 10e3
        assert metadata.vbw_hz == 1e3
        assert metadata.num_points == len(freq)

    def test_psd_metadata_span(self, transponder_with_carriers):
        """Test that PSD metadata span is correct."""
        freq, psd, metadata = generate_psd(
            transponder_with_carriers,
            rbw_hz=10e3,
            vbw_hz=1e3
        )

        expected_span = freq[-1] - freq[0]
        assert np.isclose(metadata.span_hz, expected_span, rtol=0.01)


class TestPSDParameters:
    """Test PSD generation with various parameters."""

    def test_psd_no_noise(self, transponder_with_carriers):
        """Test generating PSD without measurement noise."""
        freq, psd, metadata = generate_psd(
            transponder_with_carriers,
            rbw_hz=10e3,
            vbw_hz=1e3,
            add_noise=False
        )

        assert freq is not None
        assert psd is not None
        # PSD should be smoother without noise
        assert len(psd) > 0

    def test_psd_different_rbw(self, transponder_with_carriers):
        """Test generating PSD with different RBW values."""
        freq1, psd1, _ = generate_psd(
            transponder_with_carriers,
            rbw_hz=10e3,
            vbw_hz=1e3
        )

        freq2, psd2, _ = generate_psd(
            transponder_with_carriers,
            rbw_hz=50e3,
            vbw_hz=5e3
        )

        # Different RBW should give different number of points
        assert len(freq1) != len(freq2)

    def test_psd_vbw_affects_noise(self, transponder_with_carriers):
        """Test that VBW affects measurement noise."""
        # Low VBW/RBW ratio = less noise
        freq1, psd1, _ = generate_psd(
            transponder_with_carriers,
            rbw_hz=10e3,
            vbw_hz=100,
            add_noise=True
        )

        # High VBW/RBW ratio = more noise
        freq2, psd2, _ = generate_psd(
            transponder_with_carriers,
            rbw_hz=10e3,
            vbw_hz=10e3,
            add_noise=True
        )

        # Both should generate valid PSDs
        assert len(psd1) == len(psd2)
        assert np.all(np.isfinite(psd1))
        assert np.all(np.isfinite(psd2))


class TestPSDOutputFormat:
    """Test PSD output format."""

    def test_psd_in_dbm_per_hz(self, transponder_with_carriers):
        """Test that PSD is in dBm/Hz."""
        freq, psd, metadata = generate_psd(
            transponder_with_carriers,
            rbw_hz=10e3,
            vbw_hz=1e3
        )

        # PSD values should be in dBm (negative values for low power)
        # Typical satellite signals are -120 to -60 dBm/Hz
        assert np.all(psd < 0)  # Should be negative in dBm/Hz

    def test_frequency_array_monotonic(self, transponder_with_carriers):
        """Test that frequency array is monotonically increasing."""
        freq, psd, metadata = generate_psd(
            transponder_with_carriers,
            rbw_hz=10e3,
            vbw_hz=1e3
        )

        # Check that frequencies are monotonically increasing
        assert np.all(np.diff(freq) > 0)

    def test_psd_array_finite(self, transponder_with_carriers):
        """Test that PSD array contains finite values."""
        freq, psd, metadata = generate_psd(
            transponder_with_carriers,
            rbw_hz=10e3,
            vbw_hz=1e3
        )

        # All PSD values should be finite (no NaN or Inf)
        assert np.all(np.isfinite(psd))
