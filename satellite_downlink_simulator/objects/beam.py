"""Beam class definition."""

import attrs
from typing import List, Optional
from .transponder import Transponder
from .enums import Band, Polarization, BeamDirection


@attrs.define
class Beam:
    """
    Represents a satellite beam containing multiple transponders.

    Parameters
    ----------
    band : Band
        Frequency band (L, C, X, KA)
    polarization : Polarization
        Polarization type (LHCP, RHCP, HORIZONTAL, VERTICAL)
    direction : BeamDirection
        Beam direction (DOWNLINK, UPLINK)
    transponders : List[Transponder], optional
        List of transponders in this beam
    name : str, optional
        Optional name for the beam
    """

    band: Band = attrs.field()
    polarization: Polarization = attrs.field()
    direction: BeamDirection = attrs.field()
    transponders: List[Transponder] = attrs.field(factory=list)
    name: Optional[str] = attrs.field(default=None)

    def add_transponder(self, transponder: Transponder) -> None:
        """
        Add a transponder to the beam.

        Parameters
        ----------
        transponder : Transponder
            Transponder to add
        """
        self.transponders.append(transponder)

    def remove_transponder(self, transponder: Transponder) -> None:
        """
        Remove a transponder from the beam.

        Parameters
        ----------
        transponder : Transponder
            Transponder to remove
        """
        self.transponders.remove(transponder)

    @property
    def lower_frequency_hz(self) -> float:
        """
        Calculate the lower edge frequency of the beam.

        Returns
        -------
        float
            Lower edge frequency in Hz (minimum of all transponder lower edges)
        """
        if not self.transponders:
            return 0.0
        return min(t.lower_frequency_hz for t in self.transponders)

    @property
    def upper_frequency_hz(self) -> float:
        """
        Calculate the upper edge frequency of the beam.

        Returns
        -------
        float
            Upper edge frequency in Hz (maximum of all transponder upper edges)
        """
        if not self.transponders:
            return 0.0
        return max(t.upper_frequency_hz for t in self.transponders)

    @property
    def total_bandwidth_hz(self) -> float:
        """
        Calculate the total frequency span of the beam.

        Returns
        -------
        float
            Total bandwidth from lowest to highest frequency in Hz
        """
        if not self.transponders:
            return 0.0
        return self.upper_frequency_hz - self.lower_frequency_hz

    @property
    def center_frequency_hz(self) -> float:
        """
        Calculate the center frequency of the beam.

        Returns
        -------
        float
            Center frequency in Hz (midpoint between lowest and highest edges)
        """
        if not self.transponders:
            return 0.0
        return (self.lower_frequency_hz + self.upper_frequency_hz) / 2

    @property
    def total_carrier_power_watts(self) -> float:
        """
        Calculate total power from all carriers in all transponders.

        Returns
        -------
        float
            Total carrier power in Watts
        """
        return sum(t.total_carrier_power_watts for t in self.transponders)

    @property
    def total_noise_power_watts(self) -> float:
        """
        Calculate total noise power from all transponders.

        Returns
        -------
        float
            Total noise power in Watts
        """
        return sum(t.total_noise_power_watts for t in self.transponders)

    @property
    def total_carriers(self) -> int:
        """
        Count total number of carriers in all transponders.

        Returns
        -------
        int
            Total number of carriers
        """
        return sum(len(t.carriers) for t in self.transponders)

    def __str__(self) -> str:
        """String representation of the beam."""
        name = f"{self.name}: " if self.name else ""
        return (
            f"{name}Beam {self.band.value}-band {self.polarization.value} {self.direction.value}, "
            f"CF={self.center_frequency_hz / 1e9:.3f} GHz, "
            f"BW={self.total_bandwidth_hz / 1e6:.1f} MHz, "
            f"{len(self.transponders)} transponder(s), "
            f"{self.total_carriers} carrier(s)"
        )
