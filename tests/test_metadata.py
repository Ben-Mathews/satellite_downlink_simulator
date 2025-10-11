"""Smoke tests for metadata classes."""

import pytest
from datetime import datetime
from satellite_downlink_simulator.objects.metadata import IQMetadata, PSDMetadata


class TestIQMetadata:
    """Test IQMetadata class."""

    def test_create_iq_metadata(self):
        """Test creating IQ metadata."""
        metadata = IQMetadata(
            sample_rate_hz=50e6,
            center_frequency_hz=12.5e9,
            duration_s=0.001,
            num_samples=50000,
            bandwidth_hz=36e6
        )
        assert metadata is not None
        assert metadata.sample_rate_hz == 50e6
        assert metadata.center_frequency_hz == 12.5e9
        assert metadata.duration_s == 0.001
        assert metadata.num_samples == 50000
        assert metadata.bandwidth_hz == 36e6

    def test_iq_metadata_timestamp(self):
        """Test that timestamp is automatically created."""
        metadata = IQMetadata(
            sample_rate_hz=50e6,
            center_frequency_hz=12.5e9,
            duration_s=0.001,
            num_samples=50000,
            bandwidth_hz=36e6
        )
        assert isinstance(metadata.timestamp, datetime)

    def test_iq_metadata_custom_timestamp(self):
        """Test creating IQ metadata with custom timestamp."""
        custom_time = datetime(2024, 1, 1, 12, 0, 0)
        metadata = IQMetadata(
            sample_rate_hz=50e6,
            center_frequency_hz=12.5e9,
            duration_s=0.001,
            num_samples=50000,
            bandwidth_hz=36e6,
            timestamp=custom_time
        )
        assert metadata.timestamp == custom_time

    def test_iq_metadata_str(self):
        """Test IQ metadata string representation."""
        metadata = IQMetadata(
            sample_rate_hz=50e6,
            center_frequency_hz=12.5e9,
            duration_s=0.001,
            num_samples=50000,
            bandwidth_hz=36e6
        )
        metadata_str = str(metadata)
        assert "IQ Metadata" in metadata_str
        assert "50.000 MHz" in metadata_str
        assert "12.500000 GHz" in metadata_str


class TestPSDMetadata:
    """Test PSDMetadata class."""

    def test_create_psd_metadata(self):
        """Test creating PSD metadata."""
        metadata = PSDMetadata(
            center_frequency_hz=12.5e9,
            span_hz=36e6,
            rbw_hz=10e3,
            vbw_hz=1e3,
            num_points=3601
        )
        assert metadata is not None
        assert metadata.center_frequency_hz == 12.5e9
        assert metadata.span_hz == 36e6
        assert metadata.rbw_hz == 10e3
        assert metadata.vbw_hz == 1e3
        assert metadata.num_points == 3601

    def test_psd_metadata_timestamp(self):
        """Test that timestamp is automatically created."""
        metadata = PSDMetadata(
            center_frequency_hz=12.5e9,
            span_hz=36e6,
            rbw_hz=10e3,
            vbw_hz=1e3,
            num_points=3601
        )
        assert isinstance(metadata.timestamp, datetime)

    def test_psd_metadata_custom_timestamp(self):
        """Test creating PSD metadata with custom timestamp."""
        custom_time = datetime(2024, 1, 1, 12, 0, 0)
        metadata = PSDMetadata(
            center_frequency_hz=12.5e9,
            span_hz=36e6,
            rbw_hz=10e3,
            vbw_hz=1e3,
            num_points=3601,
            timestamp=custom_time
        )
        assert metadata.timestamp == custom_time

    def test_psd_metadata_str(self):
        """Test PSD metadata string representation."""
        metadata = PSDMetadata(
            center_frequency_hz=12.5e9,
            span_hz=36e6,
            rbw_hz=10e3,
            vbw_hz=1e3,
            num_points=3601
        )
        metadata_str = str(metadata)
        assert "PSD Metadata" in metadata_str
        assert "12.500000 GHz" in metadata_str
        assert "36.000 MHz" in metadata_str
        assert "10.000 kHz" in metadata_str
