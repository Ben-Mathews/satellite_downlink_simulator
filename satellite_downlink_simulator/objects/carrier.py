"""Carrier class definition."""

import attrs
from typing import Optional
from .enums import CarrierType, ModulationType, CarrierStandard
from ..utils import validate_positive, validate_range


@attrs.define
class Carrier:
    """
    Represents a satellite carrier signal.

    Parameters
    ----------
    frequency_offset_hz : float
        Carrier center frequency offset from transponder center in Hz
    cn_db : float
        Carrier-to-noise ratio (C/N) in dB relative to transponder noise floor
    symbol_rate_sps : float, optional
        Symbol rate in symbols/second (Baud). Required for modulated carriers,
        ignored for STATIC_CW.
    modulation : ModulationType
        Modulation type (BPSK, QPSK, etc., or STATIC_CW for unmodulated carrier)
    carrier_type : CarrierType
        Carrier type (FDMA or TDMA)
    rrc_rolloff : float, optional
        Root-raised-cosine rolloff factor (0 to 1), default 0.35.
        Ignored for STATIC_CW.
    standard : CarrierStandard, optional
        Communication standard, default NONE
    burst_time_s : float, optional
        For TDMA: duration of a single burst in seconds (required for TDMA)
    duty_cycle : float, optional
        For TDMA: fraction of time transmitting (0 to 1) (required for TDMA)
    name : str, optional
        Optional name for the carrier
    """

    frequency_offset_hz: float = attrs.field()
    cn_db: float = attrs.field()
    modulation: ModulationType = attrs.field()
    carrier_type: CarrierType = attrs.field()
    symbol_rate_sps: Optional[float] = attrs.field(default=None)
    rrc_rolloff: float = attrs.field(default=0.35)
    standard: CarrierStandard = attrs.field(default=CarrierStandard.NONE)
    burst_time_s: Optional[float] = attrs.field(default=None)
    duty_cycle: Optional[float] = attrs.field(default=None)
    name: Optional[str] = attrs.field(default=None)

    @frequency_offset_hz.validator
    def _validate_frequency_offset_hz(self, attribute, value):
        """Frequency offset can be negative (below transponder center)."""
        pass  # No validation needed, can be any value

    @cn_db.validator
    def _validate_cn_db(self, attribute, value):
        validate_positive(value, "cn_db")

    @symbol_rate_sps.validator
    def _validate_symbol_rate_sps(self, attribute, value):
        if value is not None:
            validate_positive(value, "symbol_rate_sps")

    @rrc_rolloff.validator
    def _validate_rrc_rolloff(self, attribute, value):
        validate_range(value, 0.0, 1.0, "rrc_rolloff")

    @burst_time_s.validator
    def _validate_burst_time_s(self, attribute, value):
        if value is not None:
            validate_positive(value, "burst_time_s")

    @duty_cycle.validator
    def _validate_duty_cycle(self, attribute, value):
        if value is not None:
            validate_range(value, 0.0, 1.0, "duty_cycle")

    def __attrs_post_init__(self):
        """Post-initialization validation."""
        # STATIC_CW carriers must NOT have symbol_rate_sps
        if self.modulation == ModulationType.STATIC_CW:
            if self.symbol_rate_sps is not None:
                raise ValueError(
                    "STATIC_CW carriers should not specify symbol_rate_sps (it is ignored)"
                )
        else:
            # Modulated carriers MUST have symbol_rate_sps
            if self.symbol_rate_sps is None:
                raise ValueError(
                    f"Modulated carriers ({self.modulation.value}) must specify symbol_rate_sps"
                )

        # TDMA carriers must have burst_time_s and duty_cycle
        if self.carrier_type == CarrierType.TDMA:
            if self.burst_time_s is None or self.duty_cycle is None:
                raise ValueError(
                    "TDMA carriers must specify both burst_time_s and duty_cycle"
                )

        # FDMA carriers should not have burst_time_s or duty_cycle
        if self.carrier_type == CarrierType.FDMA:
            if self.burst_time_s is not None or self.duty_cycle is not None:
                raise ValueError(
                    "FDMA carriers should not specify burst_time_s or duty_cycle"
                )

    @property
    def bandwidth_hz(self) -> float:
        """
        Calculate occupied bandwidth in Hz.

        For STATIC_CW carriers, returns a small fixed bandwidth (100 Hz).
        For modulated carriers, returns bandwidth based on symbol rate and RRC rolloff.

        Returns
        -------
        float
            Occupied bandwidth in Hz
        """
        if self.modulation == ModulationType.STATIC_CW:
            return 100.0  # Fixed small bandwidth for unmodulated CW
        return self.symbol_rate_sps * (1 + self.rrc_rolloff)

    @property
    def frame_period_s(self) -> Optional[float]:
        """
        Calculate TDMA frame period in seconds.

        Returns
        -------
        float or None
            Frame period for TDMA carriers, None for FDMA
        """
        if self.carrier_type == CarrierType.TDMA and self.burst_time_s and self.duty_cycle:
            return self.burst_time_s / self.duty_cycle
        return None

    @property
    def guard_time_s(self) -> Optional[float]:
        """
        Calculate TDMA guard time in seconds.

        Returns
        -------
        float or None
            Guard time for TDMA carriers, None for FDMA
        """
        if self.frame_period_s:
            return self.frame_period_s - self.burst_time_s
        return None

    def calculate_power_watts(self, noise_power_density_watts_per_hz: float) -> float:
        """
        Calculate carrier power in Watts from C/N and transponder noise density.

        Power is calculated as: P = N0 * BW * 10^(C/N_dB / 10)
        where N0 is noise power density and BW is carrier bandwidth.

        Parameters
        ----------
        noise_power_density_watts_per_hz : float
            Transponder noise power density in W/Hz

        Returns
        -------
        float
            Carrier power in Watts
        """
        # Noise power in carrier bandwidth
        noise_power_watts = noise_power_density_watts_per_hz * self.bandwidth_hz

        # Convert C/N from dB to linear ratio
        cn_linear = 10 ** (self.cn_db / 10)

        # Calculate carrier power
        carrier_power_watts = cn_linear * noise_power_watts

        return carrier_power_watts

    def calculate_average_power_watts(self, noise_power_density_watts_per_hz: float) -> float:
        """
        Calculate average carrier power accounting for TDMA duty cycle.

        Parameters
        ----------
        noise_power_density_watts_per_hz : float
            Transponder noise power density in W/Hz

        Returns
        -------
        float
            Average power in Watts (peak power × duty_cycle for TDMA, peak power for FDMA)
        """
        power_watts = self.calculate_power_watts(noise_power_density_watts_per_hz)

        if self.carrier_type == CarrierType.TDMA and self.duty_cycle:
            return power_watts * self.duty_cycle
        return power_watts

    def __str__(self) -> str:
        """String representation of the carrier."""
        name = f"{self.name}: " if self.name else ""
        type_str = f"{self.carrier_type.value}"
        if self.carrier_type == CarrierType.TDMA:
            type_str += f" (DC={self.duty_cycle:.2f})"

        # For STATIC_CW, don't show symbol rate
        if self.modulation == ModulationType.STATIC_CW:
            return (
                f"{name}{type_str} {self.modulation.value} @ "
                f"{self.frequency_offset_hz / 1e6:+.3f} MHz, "
                f"C/N={self.cn_db:.1f} dB"
            )

        return (
            f"{name}{type_str} {self.modulation.value} @ "
            f"{self.frequency_offset_hz / 1e6:+.3f} MHz, "
            f"SR={self.symbol_rate_sps / 1e6:.3f} Msps, "
            f"BW={self.bandwidth_hz / 1e6:.3f} MHz, "
            f"C/N={self.cn_db:.1f} dB"
        )
