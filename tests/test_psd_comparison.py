"""
Test comparing direct PSD generation with IQ->FFT PSD computation.

This test validates that the frequency-domain PSD generation matches
the time-domain IQ generation followed by FFT analysis.
"""

import pytest
import numpy as np
from scipy import signal
from satellite_downlink_simulator.objects.transponder import Transponder
from satellite_downlink_simulator.objects.carrier import Carrier
from satellite_downlink_simulator.objects.enums import CarrierType, ModulationType
from satellite_downlink_simulator.simulation.psd import generate_psd
from satellite_downlink_simulator.simulation.iq import generate_iq


def compute_psd_from_iq(iq_data, sample_rate_hz, nperseg=1024):
    """
    Compute PSD from IQ data using Welch's method.

    Parameters
    ----------
    iq_data : np.ndarray
        Complex IQ samples
    sample_rate_hz : float
        Sample rate in Hz
    nperseg : int
        Length of each segment for Welch's method

    Returns
    -------
    frequencies : np.ndarray
        Frequency array in Hz (relative to center)
    psd_dbm_per_hz : np.ndarray
        PSD in dBm/Hz
    """
    # Use Welch's method to estimate PSD
    frequencies, psd_watts_per_hz = signal.welch(
        iq_data,
        fs=sample_rate_hz,
        nperseg=nperseg,
        scaling='density',
        return_onesided=False
    )

    # Shift frequencies to be centered at 0
    frequencies = np.fft.fftshift(frequencies)
    psd_watts_per_hz = np.fft.fftshift(psd_watts_per_hz)

    # Center frequencies around 0 (remove DC offset)
    frequencies = frequencies - sample_rate_hz / 2

    # Convert to dBm/Hz
    psd_dbm_per_hz = 10 * np.log10(psd_watts_per_hz * 1000)

    return frequencies, psd_dbm_per_hz


@pytest.fixture
def transponder_10_carriers(random_seed):
    """Create a 36 MHz transponder with 10 carriers for testing."""
    transponder = Transponder(
        center_frequency_hz=12.5e9,
        bandwidth_hz=36e6,
        noise_power_density_watts_per_hz=1e-15,
        name="Test Transponder with 10 Carriers"
    )

    # Populate with 10 carriers
    num_created = transponder.populate_with_random_carriers(
        num_carriers=10,
        seed=random_seed,
        cn_range_db=(10.0, 20.0),
        symbol_rate_range_sps=(1e6, 10e6),
        modulation_types=[ModulationType.QPSK, ModulationType.BPSK],
        carrier_type=CarrierType.FDMA
    )

    assert num_created == 10, f"Expected 10 carriers, got {num_created}"

    return transponder


class TestPSDComparison:
    """Test PSD comparison between direct generation and IQ->FFT."""

    def test_psd_comparison_basic(self, transponder_10_carriers):
        """
        Compare direct PSD generation with IQ->FFT PSD for 10 carriers.

        This is the main test requested in Task 3.
        """
        # Generate direct PSD
        rbw_hz = 50e3  # 50 kHz RBW
        vbw_hz = 5e3   # 5 kHz VBW

        freq_direct, psd_direct, metadata_direct = generate_psd(
            transponder_10_carriers,
            rbw_hz=rbw_hz,
            vbw_hz=vbw_hz,
            add_noise=False  # Disable noise for cleaner comparison
        )

        # Generate IQ data
        duration_s = 0.01  # 10 ms should give good frequency resolution
        iq_data, metadata_iq = generate_iq(
            transponder_10_carriers,
            duration_s=duration_s
        )

        # Compute PSD from IQ using Welch's method
        # nperseg affects frequency resolution: freq_res ≈ sample_rate / nperseg
        # Target similar resolution to direct PSD
        nperseg = int(metadata_iq.sample_rate_hz / rbw_hz)
        freq_iq, psd_iq = compute_psd_from_iq(
            iq_data,
            metadata_iq.sample_rate_hz,
            nperseg=nperseg
        )

        # Interpolate IQ-derived PSD to match direct PSD frequencies
        psd_iq_interp = np.interp(freq_direct, freq_iq, psd_iq)

        # Compare PSDs
        # Calculate RMS error in dB
        diff_db = psd_direct - psd_iq_interp
        rms_error = np.sqrt(np.mean(diff_db ** 2))

        print(f"\n=== PSD Comparison Results ===")
        print(f"Direct PSD points: {len(freq_direct)}")
        print(f"IQ->FFT PSD points: {len(freq_iq)}")
        print(f"RMS error: {rms_error:.3f} dB")
        print(f"Mean error: {np.mean(diff_db):.3f} dB")
        print(f"Max error: {np.max(np.abs(diff_db)):.3f} dB")
        print(f"Min direct PSD: {np.min(psd_direct):.1f} dBm/Hz")
        print(f"Max direct PSD: {np.max(psd_direct):.1f} dBm/Hz")

        # Assert that RMS error is within acceptable tolerance
        # Based on user guidance: "a few dB of difference is reasonable"
        assert rms_error < 5.0, f"RMS error {rms_error:.3f} dB exceeds 5 dB threshold"

        # Also check that most points are within tolerance
        within_3db = np.sum(np.abs(diff_db) < 3.0) / len(diff_db) * 100
        print(f"Points within 3 dB: {within_3db:.1f}%")
        assert within_3db > 80, f"Only {within_3db:.1f}% of points within 3 dB"

    def test_psd_comparison_single_carrier(self, simple_transponder):
        """Test PSD comparison with a single carrier for better accuracy."""
        # Add single carrier
        carrier = Carrier(
            frequency_offset_hz=0.0,
            cn_db=15.0,
            symbol_rate_sps=10e6,
            modulation=ModulationType.QPSK,
            carrier_type=CarrierType.FDMA,
            rrc_rolloff=0.35,
            name="Single Test Carrier"
        )
        simple_transponder.add_carrier(carrier)

        # Generate direct PSD
        rbw_hz = 50e3
        vbw_hz = 5e3
        freq_direct, psd_direct, _ = generate_psd(
            simple_transponder,
            rbw_hz=rbw_hz,
            vbw_hz=vbw_hz,
            add_noise=False
        )

        # Generate IQ data
        duration_s = 0.02  # Longer duration for better frequency resolution
        iq_data, metadata_iq = generate_iq(
            simple_transponder,
            duration_s=duration_s
        )

        # Compute PSD from IQ
        nperseg = int(metadata_iq.sample_rate_hz / rbw_hz)
        freq_iq, psd_iq = compute_psd_from_iq(
            iq_data,
            metadata_iq.sample_rate_hz,
            nperseg=nperseg
        )

        # Interpolate to match frequencies
        psd_iq_interp = np.interp(freq_direct, freq_iq, psd_iq)

        # Calculate error
        diff_db = psd_direct - psd_iq_interp
        rms_error = np.sqrt(np.mean(diff_db ** 2))

        print(f"\n=== Single Carrier PSD Comparison ===")
        print(f"RMS error: {rms_error:.3f} dB")
        print(f"Mean error: {np.mean(diff_db):.3f} dB")

        # Single carrier should have better agreement
        assert rms_error < 3.0, f"Single carrier RMS error {rms_error:.3f} dB exceeds 3 dB"

    def test_psd_comparison_noise_floor(self, simple_transponder):
        """Test that noise floor matches between methods."""
        # Transponder with no carriers - just noise
        rbw_hz = 50e3
        vbw_hz = 5e3

        # Generate direct PSD
        freq_direct, psd_direct, _ = generate_psd(
            simple_transponder,
            rbw_hz=rbw_hz,
            vbw_hz=vbw_hz,
            add_noise=False
        )

        # Generate IQ data
        duration_s = 0.01
        iq_data, metadata_iq = generate_iq(
            simple_transponder,
            duration_s=duration_s
        )

        # Compute PSD from IQ
        nperseg = int(metadata_iq.sample_rate_hz / rbw_hz)
        freq_iq, psd_iq = compute_psd_from_iq(
            iq_data,
            metadata_iq.sample_rate_hz,
            nperseg=nperseg
        )

        # Average noise floor values
        noise_direct = np.median(psd_direct)
        noise_iq = np.median(psd_iq)

        print(f"\n=== Noise Floor Comparison ===")
        print(f"Direct PSD noise floor: {noise_direct:.1f} dBm/Hz")
        print(f"IQ->FFT noise floor: {noise_iq:.1f} dBm/Hz")
        print(f"Difference: {abs(noise_direct - noise_iq):.1f} dB")

        # Noise floors should be similar
        assert abs(noise_direct - noise_iq) < 3.0, \
            f"Noise floor difference {abs(noise_direct - noise_iq):.1f} dB exceeds 3 dB"

    def test_psd_comparison_carrier_peaks(self, transponder_10_carriers):
        """Test that carrier peak levels match between methods."""
        rbw_hz = 50e3
        vbw_hz = 5e3

        # Generate direct PSD
        freq_direct, psd_direct, _ = generate_psd(
            transponder_10_carriers,
            rbw_hz=rbw_hz,
            vbw_hz=vbw_hz,
            add_noise=False
        )

        # Generate IQ data
        duration_s = 0.01
        iq_data, metadata_iq = generate_iq(
            transponder_10_carriers,
            duration_s=duration_s
        )

        # Compute PSD from IQ
        nperseg = int(metadata_iq.sample_rate_hz / rbw_hz)
        freq_iq, psd_iq = compute_psd_from_iq(
            iq_data,
            metadata_iq.sample_rate_hz,
            nperseg=nperseg
        )

        # Find peak levels
        peak_direct = np.max(psd_direct)
        peak_iq = np.max(psd_iq)

        print(f"\n=== Carrier Peak Comparison ===")
        print(f"Direct PSD peak: {peak_direct:.1f} dBm/Hz")
        print(f"IQ->FFT peak: {peak_iq:.1f} dBm/Hz")
        print(f"Difference: {abs(peak_direct - peak_iq):.1f} dB")

        # Peak levels should be similar
        assert abs(peak_direct - peak_iq) < 5.0, \
            f"Peak difference {abs(peak_direct - peak_iq):.1f} dB exceeds 5 dB"

    def test_psd_comparison_integrated_power(self, transponder_10_carriers):
        """Test that integrated power matches between methods."""
        rbw_hz = 50e3
        vbw_hz = 5e3

        # Generate direct PSD
        freq_direct, psd_direct, _ = generate_psd(
            transponder_10_carriers,
            rbw_hz=rbw_hz,
            vbw_hz=vbw_hz,
            add_noise=False
        )

        # Generate IQ data
        duration_s = 0.01
        iq_data, metadata_iq = generate_iq(
            transponder_10_carriers,
            duration_s=duration_s
        )

        # Compute PSD from IQ
        nperseg = int(metadata_iq.sample_rate_hz / rbw_hz)
        freq_iq, psd_iq = compute_psd_from_iq(
            iq_data,
            metadata_iq.sample_rate_hz,
            nperseg=nperseg
        )

        # Convert to linear scale and integrate
        psd_direct_linear = 10 ** (psd_direct / 10) / 1000  # dBm to W/Hz
        psd_iq_linear = 10 ** (psd_iq / 10) / 1000

        power_direct = np.trapz(psd_direct_linear, freq_direct)
        power_iq = np.trapz(psd_iq_linear, freq_iq)

        print(f"\n=== Integrated Power Comparison ===")
        print(f"Direct PSD integrated power: {power_direct:.6e} W")
        print(f"IQ->FFT integrated power: {power_iq:.6e} W")
        print(f"Ratio: {power_iq / power_direct:.3f}")

        # Integrated power should be within 20%
        ratio = power_iq / power_direct
        assert 0.8 < ratio < 1.2, \
            f"Integrated power ratio {ratio:.3f} outside acceptable range [0.8, 1.2]"
