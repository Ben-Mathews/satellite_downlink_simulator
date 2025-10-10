"""Signal generation functions for PSD and IQ data."""

import numpy as np
from typing import Union, Tuple
from .beam import Beam
from .transponder import Transponder
from .carrier import Carrier
from .metadata import IQMetadata, PSDMetadata
from .utils import (
    rrc_filter_freq,
    rrc_filter_time,
    generate_constellation,
    apply_vbw_smoothing,
    add_measurement_noise,
)
from .enums import CarrierType


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

            # RRC filter response for carrier
            carrier_shape = rrc_filter_freq(
                freq_rel_to_carrier,
                carrier.symbol_rate_sps,
                carrier.rrc_rolloff
            )

            # Distribute carrier power across its spectrum
            # Use average power for TDMA carriers (already in Watts)
            power_watts = carrier.calculate_average_power_watts(transponder.noise_power_density_watts_per_hz)

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


def generate_iq(
    obj: Union[Beam, Transponder],
    duration_s: float,
    seed: int = None,
) -> Tuple[np.ndarray, IQMetadata]:
    """
    Generate simulated IQ data for a beam or transponder.

    This function generates actual time-domain IQ samples with modulated carriers,
    RRC pulse shaping, and TDMA bursting.

    Parameters
    ----------
    obj : Beam or Transponder
        The beam or transponder to generate IQ data for
    duration_s : float
        Duration of IQ data to generate in seconds
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    iq_data : np.ndarray
        Complex IQ samples (complex64)
    metadata : IQMetadata
        Metadata about the generated IQ data

    Raises
    ------
    ValueError
        If obj is not a Beam or Transponder, or if parameters are invalid
    """
    if duration_s <= 0:
        raise ValueError(f"duration_s must be positive, got {duration_s}")

    if seed is not None:
        np.random.seed(seed)

    # Determine frequency span and sample rate
    if isinstance(obj, Beam):
        if not obj.transponders:
            raise ValueError("Beam has no transponders")
        center_freq_hz = obj.center_frequency_hz
        bandwidth_hz = obj.total_bandwidth_hz
        transponders = obj.transponders
    elif isinstance(obj, Transponder):
        center_freq_hz = obj.center_frequency_hz
        bandwidth_hz = obj.bandwidth_hz
        transponders = [obj]
    else:
        raise ValueError(f"obj must be a Beam or Transponder, got {type(obj)}")

    # Calculate sample rate: 1.25x bandwidth
    sample_rate_hz = 1.25 * bandwidth_hz
    num_samples = int(duration_s * sample_rate_hz)
    time = np.arange(num_samples) / sample_rate_hz

    # Initialize IQ array
    iq_data = np.zeros(num_samples, dtype=np.complex64)

    # Process each transponder
    for transponder in transponders:
        # Add transponder noise with RRC shaping
        # Generate white noise
        noise_i = np.random.normal(0, 1, num_samples)
        noise_q = np.random.normal(0, 1, num_samples)
        noise = noise_i + 1j * noise_q

        # Shape noise with RRC filter to match transponder bandwidth
        # Calculate samples per symbol for noise shaping
        # Use a representative symbol rate (e.g., 1/10 of bandwidth)
        noise_symbol_rate_sps = transponder.bandwidth_hz / 10
        noise_sps = int(sample_rate_hz / noise_symbol_rate_sps)
        if noise_sps < 2:
            noise_sps = 2

        rrc_taps = rrc_filter_time(
            noise_symbol_rate_sps,
            transponder.noise_rolloff,
            span=10,
            samples_per_symbol=noise_sps
        )

        # Filter noise
        noise_shaped = np.convolve(noise, rrc_taps, mode='same')

        # Scale to match transponder noise power density
        # Power = N0 * BW, where N0 is noise power density (W/Hz)
        target_noise_power_watts = transponder.noise_power_density_watts_per_hz * transponder.bandwidth_hz
        current_noise_power = np.mean(np.abs(noise_shaped) ** 2)
        if current_noise_power > 0:
            noise_shaped *= np.sqrt(target_noise_power_watts / current_noise_power)

        # Frequency shift to transponder center
        freq_offset_hz = transponder.center_frequency_hz - center_freq_hz
        phase_shift = np.exp(2j * np.pi * freq_offset_hz * time)
        iq_data += noise_shaped * phase_shift

        # Generate each carrier
        for carrier in transponder.carriers:
            iq_carrier = _generate_carrier_iq(
                carrier,
                transponder,
                sample_rate_hz,
                num_samples,
                time,
            )

            # Frequency shift to absolute carrier frequency
            carrier_abs_freq_hz = transponder.center_frequency_hz + carrier.frequency_offset_hz
            carrier_offset_hz = carrier_abs_freq_hz - center_freq_hz
            phase_shift = np.exp(2j * np.pi * carrier_offset_hz * time)
            iq_data += iq_carrier * phase_shift

    # Create metadata
    metadata = IQMetadata(
        sample_rate_hz=sample_rate_hz,
        center_frequency_hz=center_freq_hz,
        duration_s=duration_s,
        num_samples=num_samples,
        bandwidth_hz=bandwidth_hz,
    )

    return iq_data, metadata


def _generate_carrier_iq(
    carrier: Carrier,
    transponder: Transponder,
    sample_rate_hz: float,
    num_samples: int,
    time: np.ndarray,
) -> np.ndarray:
    """
    Generate IQ data for a single carrier.

    Parameters
    ----------
    carrier : Carrier
        Carrier to generate
    transponder : Transponder
        Transponder containing the carrier (for noise density)
    sample_rate_hz : float
        Sample rate in Hz
    num_samples : int
        Number of samples to generate
    time : np.ndarray
        Time array in seconds

    Returns
    -------
    np.ndarray
        Complex IQ samples for the carrier (at baseband)
    """
    # Calculate samples per symbol
    samples_per_symbol = int(sample_rate_hz / carrier.symbol_rate_sps)
    if samples_per_symbol < 2:
        samples_per_symbol = 2

    # Number of symbols needed
    num_symbols = int(np.ceil(num_samples / samples_per_symbol)) + 20  # Extra for filter delay

    # Generate random symbols
    constellation = generate_constellation(carrier.modulation)
    symbol_indices = np.random.randint(0, len(constellation), num_symbols)
    symbols = constellation[symbol_indices]

    # Upsample symbols
    upsampled = np.zeros(len(symbols) * samples_per_symbol, dtype=complex)
    upsampled[::samples_per_symbol] = symbols

    # Apply RRC pulse shaping filter
    rrc_taps = rrc_filter_time(
        carrier.symbol_rate_sps,
        carrier.rrc_rolloff,
        span=10,
        samples_per_symbol=samples_per_symbol
    )

    # Filter
    modulated = np.convolve(upsampled, rrc_taps, mode='same')

    # Trim to correct length (account for filter delay)
    delay_samples = len(rrc_taps) // 2
    modulated = modulated[delay_samples:delay_samples + num_samples]

    # Pad if necessary
    if len(modulated) < num_samples:
        modulated = np.pad(modulated, (0, num_samples - len(modulated)))

    # Scale to carrier power (in Watts - linear scale)
    current_power = np.mean(np.abs(modulated) ** 2)
    if current_power > 0:
        carrier_power_watts = carrier.calculate_power_watts(transponder.noise_power_density_watts_per_hz)
        modulated *= np.sqrt(carrier_power_watts / current_power)

    # Apply TDMA bursting if applicable
    if carrier.carrier_type == CarrierType.TDMA:
        modulated = _apply_tdma_bursting(
            modulated,
            carrier.burst_time_s,
            carrier.duty_cycle,
            sample_rate_hz,
        )

    return modulated


def _apply_tdma_bursting(
    signal: np.ndarray,
    burst_time_s: float,
    duty_cycle: float,
    sample_rate_hz: float,
) -> np.ndarray:
    """
    Apply TDMA bursting pattern to a signal.

    Parameters
    ----------
    signal : np.ndarray
        Continuous signal to burst
    burst_time_s : float
        Burst duration in seconds
    duty_cycle : float
        Duty cycle (0 to 1)
    sample_rate_hz : float
        Sample rate in Hz

    Returns
    -------
    np.ndarray
        Bursted signal
    """
    frame_period_s = burst_time_s / duty_cycle
    burst_samples = int(burst_time_s * sample_rate_hz)
    frame_samples = int(frame_period_s * sample_rate_hz)

    # Create burst mask
    burst_mask = np.zeros(len(signal))

    for i in range(0, len(signal), frame_samples):
        end_idx = min(i + burst_samples, len(signal))
        burst_mask[i:end_idx] = 1.0

    return signal * burst_mask
