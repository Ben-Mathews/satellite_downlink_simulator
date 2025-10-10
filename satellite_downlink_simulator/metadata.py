"""Metadata classes for generated signals."""

import attrs
from typing import Optional
from datetime import datetime


@attrs.define
class IQMetadata:
    """
    Metadata for generated IQ data.

    Parameters
    ----------
    sample_rate_hz : float
        Sample rate in Hz
    center_frequency_hz : float
        Center frequency in Hz
    duration_s : float
        Duration of IQ data in seconds
    num_samples : int
        Number of IQ samples
    bandwidth_hz : float
        Signal bandwidth in Hz
    timestamp : datetime, optional
        Generation timestamp
    """

    sample_rate_hz: float = attrs.field()
    center_frequency_hz: float = attrs.field()
    duration_s: float = attrs.field()
    num_samples: int = attrs.field()
    bandwidth_hz: float = attrs.field()
    timestamp: datetime = attrs.field(factory=datetime.now)

    def __str__(self) -> str:
        """String representation of IQ metadata."""
        return (
            f"IQ Metadata:\n"
            f"  Sample Rate: {self.sample_rate_hz / 1e6:.3f} MHz\n"
            f"  Center Frequency: {self.center_frequency_hz / 1e9:.6f} GHz\n"
            f"  Duration: {self.duration_s:.6f} s\n"
            f"  Number of Samples: {self.num_samples:,}\n"
            f"  Bandwidth: {self.bandwidth_hz / 1e6:.3f} MHz\n"
            f"  Timestamp: {self.timestamp.isoformat()}"
        )


@attrs.define
class PSDMetadata:
    """
    Metadata for generated PSD data.

    Parameters
    ----------
    center_frequency_hz : float
        Center frequency in Hz
    span_hz : float
        Frequency span in Hz
    rbw_hz : float
        Resolution bandwidth in Hz
    vbw_hz : float
        Video bandwidth in Hz
    num_points : int
        Number of frequency points
    timestamp : datetime, optional
        Generation timestamp
    """

    center_frequency_hz: float = attrs.field()
    span_hz: float = attrs.field()
    rbw_hz: float = attrs.field()
    vbw_hz: float = attrs.field()
    num_points: int = attrs.field()
    timestamp: datetime = attrs.field(factory=datetime.now)

    def __str__(self) -> str:
        """String representation of PSD metadata."""
        return (
            f"PSD Metadata:\n"
            f"  Center Frequency: {self.center_frequency_hz / 1e9:.6f} GHz\n"
            f"  Span: {self.span_hz / 1e6:.3f} MHz\n"
            f"  RBW: {self.rbw_hz / 1e3:.3f} kHz\n"
            f"  VBW: {self.vbw_hz / 1e3:.3f} kHz\n"
            f"  Number of Points: {self.num_points:,}\n"
            f"  Timestamp: {self.timestamp.isoformat()}"
        )
