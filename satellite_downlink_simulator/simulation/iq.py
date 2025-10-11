"""IQ time-domain signal generation for satellite signals."""

import numpy as np
from typing import Union, Tuple
from ..objects.beam import Beam
from ..objects.transponder import Transponder
from ..objects.carrier import Carrier
from ..objects.metadata import IQMetadata
from ..objects.enums import CarrierType
from ..utils import (
    rrc_filter_time,
    generate_constellation,
)


def generate_iq(
    obj: Union[Beam, Transponder],
    duration_s: float,
    seed: int = None,
    sample_rate_hz: float = None,
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
    sample_rate_hz : float, optional
        Sample rate in Hz. If not specified, defaults to 1.25 × bandwidth.
        Must be >= bandwidth to satisfy Nyquist criterion.

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

    # Calculate sample rate: 1.25x bandwidth (default) or use user-provided value
    if sample_rate_hz is None:
        sample_rate_hz = 1.25 * bandwidth_hz

    # Validate sample rate meets Nyquist criterion
    if sample_rate_hz < bandwidth_hz:
        raise ValueError(
            f"sample_rate_hz must be >= bandwidth to satisfy Nyquist criterion. "
            f"Minimum required: {bandwidth_hz:.2f} Hz, got: {sample_rate_hz:.2f} Hz"
        )

    num_samples = int(duration_s * sample_rate_hz)
    time = np.arange(num_samples) / sample_rate_hz

    # Initialize IQ array
    iq_data = np.zeros(num_samples, dtype=np.complex64)

    # Process each transponder
    for transponder in transponders:
        # Add transponder noise with RRC shaping
        # Goal: Match the noise floor from generate_psd() where PSD = N0 * |H_RRC(f)|²
        #
        # Solution: Apply RRC frequency response directly in frequency domain
        # This ensures the spectral shape matches exactly
        #
        # Step 1: Generate white noise with PSD = N0 across the sample rate
        noise_power = transponder.noise_power_density_watts_per_hz * sample_rate_hz
        noise_std = np.sqrt(noise_power / 2)
        noise_i = np.random.normal(0, noise_std, num_samples)
        noise_q = np.random.normal(0, noise_std, num_samples)
        noise = noise_i + 1j * noise_q

        # Step 2: Apply RRC shaping in frequency domain
        # Take FFT of noise
        noise_fft = np.fft.fft(noise)
        freqs_fft = np.fft.fftfreq(num_samples, 1/sample_rate_hz)

        # Get RRC frequency response
        # Use transponder bandwidth as symbol rate to match generate_psd()
        from ..utils import rrc_filter_freq
        rrc_response = rrc_filter_freq(freqs_fft, transponder.bandwidth_hz, transponder.noise_rolloff)

        # Apply RRC shape (multiply in frequency domain)
        noise_fft_shaped = noise_fft * rrc_response

        # Step 3: Convert back to time domain
        noise_shaped = np.fft.ifft(noise_fft_shaped)

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
    from ..objects.enums import ModulationType

    # Handle STATIC_CW carriers (unmodulated continuous wave)
    if carrier.modulation == ModulationType.STATIC_CW:
        # For STATIC_CW, generate a constant complex tone with carrier power
        carrier_power_watts = carrier.calculate_power_watts(transponder.noise_power_density_watts_per_hz)
        amplitude = np.sqrt(carrier_power_watts)
        # Return a constant amplitude complex signal (at baseband, no frequency offset yet)
        return np.full(num_samples, amplitude + 0j, dtype=np.complex64)

    # Calculate samples per symbol for modulated carriers
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
