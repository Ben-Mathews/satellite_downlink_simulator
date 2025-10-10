"""Utility functions for signal processing and validation."""

import numpy as np
from scipy import signal
from typing import Tuple
from .enums import ModulationType


def rrc_filter_time(symbol_rate: float, rolloff: float, span: int, samples_per_symbol: int) -> np.ndarray:
    """
    Generate a root-raised-cosine (RRC) filter in time domain.

    Parameters
    ----------
    symbol_rate : float
        Symbol rate in symbols/second
    rolloff : float
        Rolloff factor (beta), typically 0.2-0.5
    span : int
        Filter span in symbols
    samples_per_symbol : int
        Number of samples per symbol

    Returns
    -------
    np.ndarray
        RRC filter coefficients (normalized to unit energy)
    """
    if not 0 <= rolloff <= 1:
        raise ValueError(f"Rolloff must be between 0 and 1, got {rolloff}")

    num_taps = span * samples_per_symbol + 1
    t = np.arange(num_taps, dtype=float) / samples_per_symbol
    t = t - (num_taps - 1) / 2 / samples_per_symbol  # Center at 0

    # Handle special cases to avoid divide by zero
    h = np.zeros(num_taps)

    for i, time in enumerate(t):
        if time == 0:
            h[i] = (1 + rolloff * (4 / np.pi - 1))
        elif abs(abs(time) - 1 / (4 * rolloff)) < 1e-10:
            # Special case at t = +/- 1/(4*rolloff)
            h[i] = (rolloff / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff)) +
                (1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff))
            )
        else:
            numerator = np.sin(np.pi * time * (1 - rolloff)) + \
                       4 * rolloff * time * np.cos(np.pi * time * (1 + rolloff))
            denominator = np.pi * time * (1 - (4 * rolloff * time) ** 2)
            h[i] = numerator / denominator

    # Normalize to unit energy
    h = h / np.sqrt(np.sum(h ** 2))

    return h


def rrc_filter_freq(frequencies: np.ndarray, symbol_rate: float, rolloff: float) -> np.ndarray:
    """
    Generate the frequency response of a root-raised-cosine filter.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency array in Hz
    symbol_rate : float
        Symbol rate in symbols/second
    rolloff : float
        Rolloff factor (beta)

    Returns
    -------
    np.ndarray
        Magnitude of RRC frequency response (not squared)
    """
    if not 0 <= rolloff <= 1:
        raise ValueError(f"Rolloff must be between 0 and 1, got {rolloff}")

    # Normalize frequencies by symbol rate
    f_norm = np.abs(frequencies) / symbol_rate

    H = np.zeros_like(f_norm)

    # Passband: |f| <= (1-rolloff)/(2T)
    passband = f_norm <= (1 - rolloff) / 2
    H[passband] = 1.0

    # Transition band: (1-rolloff)/(2T) < |f| <= (1+rolloff)/(2T)
    transition = (f_norm > (1 - rolloff) / 2) & (f_norm <= (1 + rolloff) / 2)
    if np.any(transition):
        f_t = f_norm[transition]
        H[transition] = np.sqrt(0.5 * (1 + np.cos(np.pi / rolloff * (f_t - (1 - rolloff) / 2))))

    # Stopband: |f| > (1+rolloff)/(2T)
    # H is already zero

    return H


def generate_constellation(modulation: ModulationType) -> np.ndarray:
    """
    Generate constellation points for a given modulation type.

    Parameters
    ----------
    modulation : ModulationType
        The modulation type

    Returns
    -------
    np.ndarray
        Complex constellation points (normalized to unit average power)
    """
    if modulation == ModulationType.BPSK:
        constellation = np.array([-1, 1], dtype=complex)

    elif modulation == ModulationType.QPSK:
        # Gray coded QPSK
        constellation = np.array([
            1 + 1j,  # 00
            -1 + 1j, # 01
            1 - 1j,  # 10
            -1 - 1j  # 11
        ], dtype=complex) / np.sqrt(2)

    elif modulation == ModulationType.QAM16:
        # 16-QAM constellation (square)
        levels = np.array([-3, -1, 1, 3])
        I, Q = np.meshgrid(levels, levels)
        constellation = (I + 1j * Q).flatten() / np.sqrt(10)

    elif modulation == ModulationType.APSK16:
        # 16-APSK: 4 inner + 12 outer ring (DVB-S2 style)
        # Radius ratio optimized for AWGN
        r1 = 1.0  # Inner ring
        r2 = 2.85  # Outer ring (typical value)

        # Inner ring: 4 points
        inner = r1 * np.exp(1j * np.pi / 4 * (2 * np.arange(4) + 1))

        # Outer ring: 12 points
        outer = r2 * np.exp(1j * np.pi / 6 * (2 * np.arange(12) + 1))

        constellation = np.concatenate([inner, outer])
        # Normalize to unit average power
        constellation = constellation / np.sqrt(np.mean(np.abs(constellation) ** 2))

    elif modulation == ModulationType.APSK32:
        # 32-APSK: 4+12+16 rings (DVB-S2 style)
        r1 = 1.0
        r2 = 2.84
        r3 = 5.27

        # Inner ring: 4 points
        inner = r1 * np.exp(1j * np.pi / 4 * (2 * np.arange(4) + 1))

        # Middle ring: 12 points
        middle = r2 * np.exp(1j * np.pi / 6 * (2 * np.arange(12) + 1))

        # Outer ring: 16 points
        outer = r3 * np.exp(1j * np.pi / 8 * (2 * np.arange(16) + 1))

        constellation = np.concatenate([inner, middle, outer])
        # Normalize to unit average power
        constellation = constellation / np.sqrt(np.mean(np.abs(constellation) ** 2))

    else:
        raise ValueError(f"Unknown modulation type: {modulation}")

    return constellation


def apply_vbw_smoothing(psd: np.ndarray, rbw: float, vbw: float) -> np.ndarray:
    """
    Apply VBW (video bandwidth) smoothing to a PSD trace.

    Parameters
    ----------
    psd : np.ndarray
        PSD values (linear scale)
    rbw : float
        Resolution bandwidth in Hz
    vbw : float
        Video bandwidth in Hz

    Returns
    -------
    np.ndarray
        Smoothed PSD values
    """
    if vbw >= rbw:
        # No smoothing needed
        return psd

    # Number of points to average (approximate)
    # VBW acts like a moving average filter
    num_avg = max(1, int(rbw / vbw))

    # Apply moving average
    kernel = np.ones(num_avg) / num_avg
    smoothed = np.convolve(psd, kernel, mode='same')

    return smoothed


def watts_to_dbm(watts: float) -> float:
    """Convert power from Watts to dBm."""
    return 10 * np.log10(watts * 1000)


def dbm_to_watts(dbm: float) -> float:
    """Convert power from dBm to Watts."""
    return 10 ** (dbm / 10) / 1000


def add_measurement_noise(psd_linear: np.ndarray, noise_factor_db: float = 0.5) -> np.ndarray:
    """
    Add realistic measurement noise to a PSD trace.

    Parameters
    ----------
    psd_linear : np.ndarray
        PSD values in linear scale (W/Hz)
    noise_factor_db : float
        Standard deviation of noise in dB

    Returns
    -------
    np.ndarray
        PSD with added measurement noise (still in linear scale)
    """
    # Convert to dB, add Gaussian noise, convert back
    psd_db = 10 * np.log10(psd_linear + 1e-20)  # Avoid log(0)
    noise = np.random.normal(0, noise_factor_db, size=psd_db.shape)
    psd_noisy_db = psd_db + noise
    psd_noisy_linear = 10 ** (psd_noisy_db / 10)

    return psd_noisy_linear


def validate_positive(value: float, name: str) -> None:
    """Validate that a value is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_range(value: float, min_val: float, max_val: float, name: str) -> None:
    """Validate that a value is within a range."""
    if not min_val <= value <= max_val:
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")
