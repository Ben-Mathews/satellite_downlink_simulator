"""Transponder class definition."""

import attrs
import numpy as np
from typing import List, Optional, Tuple
from .carrier import Carrier
from .enums import ModulationType, CarrierType
from ..utils import validate_positive


@attrs.define
class Transponder:
    """
    Represents a satellite transponder containing multiple carriers.

    Parameters
    ----------
    center_frequency_hz : float
        Absolute downlink center frequency in Hz
    bandwidth_hz : float
        Transponder bandwidth in Hz
    noise_power_density_watts_per_hz : float
        Noise floor power spectral density in W/Hz
    carriers : List[Carrier], optional
        List of carriers in this transponder
    noise_rolloff : float, optional
        RRC rolloff factor for transponder edge shaping, default 0.25
    allow_overlap : bool, optional
        Allow carriers to overlap in frequency, default False
    name : str, optional
        Optional name for the transponder
    """

    center_frequency_hz: float = attrs.field()
    bandwidth_hz: float = attrs.field()
    noise_power_density_watts_per_hz: float = attrs.field()
    carriers: List[Carrier] = attrs.field(factory=list)
    noise_rolloff: float = attrs.field(default=0.25)
    allow_overlap: bool = attrs.field(default=False)
    name: Optional[str] = attrs.field(default=None)

    @center_frequency_hz.validator
    def _validate_center_frequency_hz(self, attribute, value):
        validate_positive(value, "center_frequency_hz")

    @bandwidth_hz.validator
    def _validate_bandwidth_hz(self, attribute, value):
        validate_positive(value, "bandwidth_hz")

    @noise_power_density_watts_per_hz.validator
    def _validate_noise_power_density_watts_per_hz(self, attribute, value):
        validate_positive(value, "noise_power_density_watts_per_hz")

    @noise_rolloff.validator
    def _validate_noise_rolloff(self, attribute, value):
        if not 0 <= value <= 1:
            raise ValueError(f"noise_rolloff must be between 0 and 1, got {value}")

    def __attrs_post_init__(self):
        """Post-initialization validation."""
        self.validate_carriers()

    def validate_carriers(self) -> None:
        """
        Validate that all carriers fit within the transponder bandwidth
        and check for overlaps if not allowed.

        Raises
        ------
        ValueError
            If any carrier extends beyond transponder bandwidth or overlaps occur
        """
        if not self.carriers:
            return

        # Check that all carriers fit within transponder bandwidth
        for carrier in self.carriers:
            carrier_lower = carrier.frequency_offset_hz - carrier.bandwidth_hz / 2
            carrier_upper = carrier.frequency_offset_hz + carrier.bandwidth_hz / 2
            transponder_lower = -self.bandwidth_hz / 2
            transponder_upper = self.bandwidth_hz / 2

            if carrier_lower < transponder_lower or carrier_upper > transponder_upper:
                raise ValueError(
                    f"Carrier '{carrier.name or 'unnamed'}' at offset "
                    f"{carrier.frequency_offset_hz / 1e6:.3f} MHz with bandwidth "
                    f"{carrier.bandwidth_hz / 1e6:.3f} MHz extends beyond transponder "
                    f"bandwidth {self.bandwidth_hz / 1e6:.3f} MHz. "
                    f"Carrier range: [{carrier_lower / 1e6:.3f}, {carrier_upper / 1e6:.3f}] MHz, "
                    f"Transponder range: [{transponder_lower / 1e6:.3f}, {transponder_upper / 1e6:.3f}] MHz"
                )

        # Check for carrier overlaps if not allowed
        if not self.allow_overlap and len(self.carriers) > 1:
            # Sort carriers by frequency offset
            sorted_carriers = sorted(self.carriers, key=lambda c: c.frequency_offset_hz)

            for i in range(len(sorted_carriers) - 1):
                carrier1 = sorted_carriers[i]
                carrier2 = sorted_carriers[i + 1]

                carrier1_upper = carrier1.frequency_offset_hz + carrier1.bandwidth_hz / 2
                carrier2_lower = carrier2.frequency_offset_hz - carrier2.bandwidth_hz / 2

                if carrier1_upper > carrier2_lower:
                    raise ValueError(
                        f"Carriers '{carrier1.name or f'#{i}'}' and "
                        f"'{carrier2.name or f'#{i+1}'}' overlap. "
                        f"Carrier 1 upper edge: {carrier1_upper / 1e6:.3f} MHz, "
                        f"Carrier 2 lower edge: {carrier2_lower / 1e6:.3f} MHz. "
                        f"Set allow_overlap=True to permit overlapping carriers."
                    )

    def add_carrier(self, carrier: Carrier) -> None:
        """
        Add a carrier to the transponder.

        Parameters
        ----------
        carrier : Carrier
            Carrier to add

        Raises
        ------
        ValueError
            If carrier validation fails
        """
        # Temporarily add carrier for validation
        self.carriers.append(carrier)
        try:
            self.validate_carriers()
        except ValueError:
            # Remove carrier if validation fails
            self.carriers.remove(carrier)
            raise

    def remove_carrier(self, carrier: Carrier) -> None:
        """
        Remove a carrier from the transponder.

        Parameters
        ----------
        carrier : Carrier
            Carrier to remove
        """
        self.carriers.remove(carrier)

    def populate_with_random_carriers(
        self,
        num_carriers: int,
        symbol_rate_range_sps: Tuple[float, float] = (100e3, 20e6),
        cn_range_db: Tuple[float, float] = (6.0, 30.0),
        modulation_types: Optional[List[ModulationType]] = None,
        rrc_rolloff_values: Optional[List[float]] = None,
        carrier_type: CarrierType = CarrierType.FDMA,
        max_iterations: int = 1000,
        seed: Optional[int] = None,
        name_prefix: str = "C"
    ) -> int:
        """
        Populate transponder with randomly generated non-overlapping carriers.

        This method iteratively attempts to place carriers within the transponder
        bandwidth until the target number is reached or max iterations exceeded.

        Parameters
        ----------
        num_carriers : int
            Target number of carriers to create
        symbol_rate_range_sps : Tuple[float, float], optional
            Min and max symbol rate in symbols/second, default (100e3, 20e6)
        cn_range_db : Tuple[float, float], optional
            Min and max C/N in dB, default (6.0, 30.0)
        modulation_types : List[ModulationType], optional
            List of allowed modulation types, defaults to all types
        rrc_rolloff_values : List[float], optional
            List of allowed RRC rolloff values, default [0.20, 0.25, 0.35]
        carrier_type : CarrierType, optional
            Carrier type (FDMA or TDMA), default FDMA
        max_iterations : int, optional
            Maximum iterations without successful placement, default 1000
        seed : int, optional
            Random seed for reproducibility
        name_prefix : str, optional
            Prefix for carrier names, default "C"

        Returns
        -------
        int
            Number of carriers successfully created

        Notes
        -----
        - Carrier center frequencies are rounded to nearest 100 kHz
        - Symbol rates are in increments of 100 kHz
        - Carrier centers must be at least 10% of transponder BW from edges
        - All carrier bandwidth must fit within transponder bandwidth
        """
        if seed is not None:
            np.random.seed(seed)

        # Set defaults
        if modulation_types is None:
            modulation_types = [
                ModulationType.BPSK,
                ModulationType.QPSK,
                ModulationType.QAM16,
                ModulationType.APSK16,
                ModulationType.APSK32
            ]

        if rrc_rolloff_values is None:
            rrc_rolloff_values = [0.20, 0.25, 0.35]

        # Validate inputs
        if symbol_rate_range_sps[0] > symbol_rate_range_sps[1]:
            raise ValueError("symbol_rate_range_sps min must be <= max")
        if cn_range_db[0] > cn_range_db[1]:
            raise ValueError("cn_range_db min must be <= max")

        freq_increment = 100e3  # 100 kHz
        symbol_rate_increment = 100e3  # 100 kHz

        # Define usable frequency range (10% margin from edges)
        margin = 0.20 * self.bandwidth_hz
        usable_lower = -self.bandwidth_hz / 2 + margin
        usable_upper = self.bandwidth_hz / 2 - margin

        carriers_added = 0
        iterations_without_success = 0

        while carriers_added < num_carriers and iterations_without_success < max_iterations:
            # Step 1: Pick random center frequency (rounded to 100 kHz)
            center_offset_hz = np.random.uniform(usable_lower, usable_upper)
            center_offset_hz = round(center_offset_hz / freq_increment) * freq_increment

            # Step 3: Pick random C/N and modulation
            cn_db = np.random.uniform(cn_range_db[0], cn_range_db[1])
            modulation = np.random.choice(modulation_types)
            rrc_rolloff = np.random.choice(rrc_rolloff_values)

            # Step 2: Pick random symbol rate that fits
            # Calculate max symbol rate based on distance to transponder edges
            distance_to_lower_edge = center_offset_hz - (-self.bandwidth_hz / 2)
            distance_to_upper_edge = (self.bandwidth_hz / 2) - center_offset_hz
            max_carrier_half_bw = min(distance_to_lower_edge, distance_to_upper_edge)

            # Bandwidth = symbol_rate * (1 + rrc_rolloff)
            max_symbol_rate = (2 * max_carrier_half_bw) / (1 + rrc_rolloff)

            # Apply user constraints
            effective_min_sr = symbol_rate_range_sps[0]
            effective_max_sr = min(symbol_rate_range_sps[1], max_symbol_rate)

            if effective_min_sr > effective_max_sr:
                # Can't fit a carrier here
                iterations_without_success += 1
                continue

            # Pick symbol rate in increments of 100 kHz
            num_steps = int((effective_max_sr - effective_min_sr) / symbol_rate_increment)
            if num_steps < 1:
                iterations_without_success += 1
                continue

            step = np.random.randint(0, num_steps + 1)
            symbol_rate_sps = effective_min_sr + step * symbol_rate_increment

            # Create carrier
            try:
                carrier = Carrier(
                    frequency_offset_hz=center_offset_hz,
                    cn_db=cn_db,
                    symbol_rate_sps=symbol_rate_sps,
                    modulation=modulation,
                    carrier_type=carrier_type,
                    rrc_rolloff=rrc_rolloff,
                    name=f"{name_prefix}{carriers_added + 1}"
                )

                # Try to add carrier (will validate for overlaps)
                self.add_carrier(carrier)
                carriers_added += 1
                iterations_without_success = 0  # Reset counter on success

            except ValueError:
                # Carrier overlaps or doesn't fit
                iterations_without_success += 1
                continue

        return carriers_added

    @property
    def total_carrier_power_watts(self) -> float:
        """
        Calculate total power from all carriers (average power for TDMA).

        Returns
        -------
        float
            Total carrier power in Watts
        """
        return sum(
            carrier.calculate_average_power_watts(self.noise_power_density_watts_per_hz)
            for carrier in self.carriers
        )

    @property
    def total_noise_power_watts(self) -> float:
        """
        Calculate total noise power across the transponder bandwidth.

        Returns
        -------
        float
            Total noise power in Watts
        """
        return self.noise_power_density_watts_per_hz * self.bandwidth_hz

    @property
    def lower_frequency_hz(self) -> float:
        """
        Calculate the lower edge frequency of the transponder.

        Returns
        -------
        float
            Lower edge frequency in Hz
        """
        return self.center_frequency_hz - self.bandwidth_hz / 2

    @property
    def upper_frequency_hz(self) -> float:
        """
        Calculate the upper edge frequency of the transponder.

        Returns
        -------
        float
            Upper edge frequency in Hz
        """
        return self.center_frequency_hz + self.bandwidth_hz / 2

    def __str__(self) -> str:
        """String representation of the transponder."""
        name = f"{self.name}: " if self.name else ""
        return (
            f"{name}Transponder @ {self.center_frequency_hz / 1e9:.3f} GHz, "
            f"BW={self.bandwidth_hz / 1e6:.1f} MHz, "
            f"NPD={self.noise_power_density_watts_per_hz:.3e} W/Hz, "
            f"{len(self.carriers)} carrier(s)"
        )
