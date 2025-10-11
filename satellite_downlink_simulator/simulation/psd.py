"""Power Spectral Density (PSD) generation for satellite signals."""

import numpy as np
from typing import Union, Tuple
from ..objects.beam import Beam
from ..objects.transponder import Transponder
from ..objects.metadata import PSDMetadata
from ..objects.enums import ModulationType
from ..utils import (
    rrc_filter_freq,
    add_measurement_noise,
)


def generate_psd(
    obj: Union[Beam, Transponder],
    rbw_hz: float,
    vbw_hz: float = None,
    add_noise: bool = True,
    noise_factor_db: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, PSDMetadata]:
    """
    Generate a simulated Power Spectral Density (PSD) for a beam or transponder.

    This function generates a fast PSD simulation without creating full IQ data.
    It models carrier shapes using RRC filters and adds transponder noise floors.

    Parameters
    ----------
    obj : Beam or Transponder
        The beam or transponder to generate PSD for
    rbw_hz : float
        Resolution bandwidth in Hz (determines frequency bin spacing)
    vbw_hz : float, optional
        Video bandwidth in Hz (for smoothing), defaults to rbw_hz if not specified
    add_noise : bool, optional
        Add measurement noise to the PSD trace, default True
    noise_factor_db : float, optional
        Standard deviation of measurement noise in dB, default 0.5

    Returns
    -------
    frequencies : np.ndarray
        Frequency array in Hz (absolute frequencies)
    psd : np.ndarray
        PSD values in dBm/Hz
    metadata : PSDMetadata
        Metadata about the generated PSD

    Raises
    ------
    ValueError
        If obj is not a Beam or Transponder, or if parameters are invalid
    """
    if vbw_hz is None:
        vbw_hz = rbw_hz

    if rbw_hz <= 0:
        raise ValueError(f"rbw_hz must be positive, got {rbw_hz}")
    if vbw_hz <= 0:
        raise ValueError(f"vbw_hz must be positive, got {vbw_hz}")

    # Determine frequency span and center
    if isinstance(obj, Beam):
        if not obj.transponders:
            raise ValueError("Beam has no transponders")
        center_freq_hz = obj.center_frequency_hz
        span_hz = obj.total_bandwidth_hz
        transponders = obj.transponders
    elif isinstance(obj, Transponder):
        center_freq_hz = obj.center_frequency_hz
        span_hz = obj.bandwidth_hz
        transponders = [obj]
    else:
        raise ValueError(f"obj must be a Beam or Transponder, got {type(obj)}")

    # Create frequency array based on RBW
    num_points = int(span_hz / rbw_hz) + 1
    frequencies = np.linspace(
        center_freq_hz - span_hz / 2,
        center_freq_hz + span_hz / 2,
        num_points
    )

    # Initialize PSD array (linear scale, W/Hz)
    psd_linear = np.zeros(num_points)

    # Process each transponder
    for transponder in transponders:
        # Get frequency range for this transponder
        t_lower = transponder.lower_frequency_hz
        t_upper = transponder.upper_frequency_hz
        t_center = transponder.center_frequency_hz

        # Find indices within transponder range
        t_mask = (frequencies >= t_lower) & (frequencies <= t_upper)
        t_freqs = frequencies[t_mask]

        if len(t_freqs) == 0:
            continue

        # Add transponder noise floor with RRC shaping
        freq_rel = t_freqs - t_center  # Relative to transponder center
        noise_shape = rrc_filter_freq(freq_rel, transponder.bandwidth_hz, transponder.noise_rolloff)
        noise_psd = transponder.noise_power_density_watts_per_hz * (noise_shape ** 2)
        psd_linear[t_mask] += noise_psd

        # Add each carrier
        for carrier in transponder.carriers:
            # Carrier frequency relative to frequency array
            carrier_abs_freq = t_center + carrier.frequency_offset_hz
            freq_rel_to_carrier = t_freqs - carrier_abs_freq

            # Get carrier power
            power_watts = carrier.calculate_average_power_watts(transponder.noise_power_density_watts_per_hz)

            if carrier.modulation == ModulationType.STATIC_CW:
                # STATIC_CW: Render as impulse (delta function) concentrated in nearest bin
                # Find the nearest frequency bin to the carrier
                nearest_idx = np.argmin(np.abs(freq_rel_to_carrier))

                # Create impulse: concentrate all power as PSD in one bin
                # The power_watts is already in Watts, convert to W/Hz for this bin
                # Since this is an impulse, the PSD value represents all the power
                carrier_psd = np.zeros_like(freq_rel_to_carrier)
                # Use the actual carrier bandwidth (100 Hz) to calculate PSD
                carrier_psd[nearest_idx] = power_watts / carrier.bandwidth_hz
            else:
                # Modulated carrier: Use RRC filter response
                carrier_shape = rrc_filter_freq(
                    freq_rel_to_carrier,
                    carrier.symbol_rate_sps,
                    carrier.rrc_rolloff
                )

                # Normalize: integrate |H(f)|^2 to get total power scaling
                # For RRC, the integral of |H(f)|^2 over all frequencies equals 1/T (symbol rate)
                # So power spectral density = power / symbol_rate * |H(f)|^2
                carrier_psd = power_watts / carrier.symbol_rate_sps * (carrier_shape ** 2)

            # Add to total PSD
            psd_linear[t_mask] += carrier_psd

    # Add measurement noise based on VBW
    # VBW sets the capture/integration time which affects noise variance
    # Capture time = RBW / VBW (approximately)
    # Noise std dev scales as 1/sqrt(capture_time) = sqrt(VBW/RBW)
    if add_noise:
        # Calculate noise scaling based on VBW/RBW ratio
        vbw_rbw_ratio = vbw_hz / rbw_hz
        # Scale the noise standard deviation
        # Base noise is noise_factor_db, scaled by sqrt(VBW/RBW)
        effective_noise_std_db = noise_factor_db * np.sqrt(vbw_rbw_ratio)
        psd_linear = add_measurement_noise(psd_linear, effective_noise_std_db)

    # Convert to dBm/Hz (convert Watts to milliwatts then to dB)
    psd_dbm_hz = 10 * np.log10(psd_linear * 1000 + 1e-30)  # Avoid log(0)

    # Create metadata
    metadata = PSDMetadata(
        center_frequency_hz=center_freq_hz,
        span_hz=span_hz,
        rbw_hz=rbw_hz,
        vbw_hz=vbw_hz,
        num_points=num_points,
    )

    return frequencies, psd_dbm_hz, metadata
