"""
Examples and test cases for the Satellite Spectrum Emulator.

This script demonstrates various capabilities of the simulator including:
1. Single transponder with multiple FDMA carriers
2. TDMA carrier bursting
3. Multi-transponder beam
4. Overlapping vs non-overlapping carriers
5. Different modulation types
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from satellite_downlink_simulator import (
    Carrier, Transponder, Beam,
    CarrierType, ModulationType, CarrierStandard,
    Band, Polarization, BeamDirection,
    generate_psd, generate_iq
)


def example1_single_transponder_fdma():
    """Example 1: Single transponder with multiple FDMA carriers."""
    print("\n" + "="*80)
    print("Example 1: Single Transponder with Multiple FDMA Carriers")
    print("="*80)

    # Create carriers with different modulations
    carrier1 = Carrier(
        frequency_offset_hz=-12e6,  # -12 MHz
        cn_db=15.0,  # 15 dB C/N
        symbol_rate_sps=5e6,  # 5 Msps
        modulation=ModulationType.QPSK,
        carrier_type=CarrierType.FDMA,
        rrc_rolloff=0.35,
        standard=CarrierStandard.DVB_S2,
        name="QPSK Carrier"
    )

    carrier2 = Carrier(
        frequency_offset_hz=0.0,  # Center
        cn_db=18.0,  # 18 dB C/N
        symbol_rate_sps=10e6,  # 10 Msps
        modulation=ModulationType.APSK16,
        carrier_type=CarrierType.FDMA,
        standard=CarrierStandard.DVB_S2,
        name="16APSK Carrier"
    )

    carrier3 = Carrier(
        frequency_offset_hz=11e6,  # +11 MHz
        cn_db=14.0,  # 14 dB C/N
        symbol_rate_sps=3e6,  # 3 Msps
        modulation=ModulationType.QPSK,
        carrier_type=CarrierType.FDMA,
        standard=CarrierStandard.DVB_S,
        name="DVB-S Carrier"
    )

    # Create transponder
    transponder = Transponder(
        center_frequency_hz=12.5e9,  # 12.5 GHz
        bandwidth_hz=36e6,  # 36 MHz
        noise_power_density_watts_per_hz=1e-15,  # W/Hz
        carriers=[carrier1, carrier2, carrier3],
        noise_rolloff=0.25,
        name="Ku-band Transponder"
    )

    print(f"\n{transponder}")
    for i, carrier in enumerate(transponder.carriers, 1):
        print(f"  {i}. {carrier}")

    # Generate PSD
    print("\nGenerating PSD...")
    frequencies, psd, psd_meta = generate_psd(
        transponder,
        rbw_hz=10e3,  # 10 kHz
        vbw_hz=10e3,
    )

    # Generate IQ
    print("Generating IQ data...")
    iq_data, iq_meta = generate_iq(
        transponder,
        duration_s=0.001,  # 1 ms
        seed=42
    )

    print(f"\n{psd_meta}")
    print(f"\n{iq_meta}")

    # Plot
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 2, figure=fig)

    # PSD
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot((frequencies - transponder.center_frequency_hz) / 1e6, psd, linewidth=0.5)
    ax1.set_xlabel('Frequency Offset from Transponder Center (MHz)')
    ax1.set_ylabel('PSD (dBm/Hz)')
    ax1.set_title('Power Spectral Density - FDMA Carriers')
    ax1.grid(True, alpha=0.3)

    # IQ time domain - I channel
    ax2 = fig.add_subplot(gs[1, 0])
    time_us = np.arange(min(10000, len(iq_data))) / iq_meta.sample_rate_hz * 1e6
    ax2.plot(time_us, np.real(iq_data[:len(time_us)]), linewidth=0.5)
    ax2.set_xlabel('Time (us)')
    ax2.set_ylabel('Amplitude (I)')
    ax2.set_title('IQ Time Domain - In-Phase')
    ax2.grid(True, alpha=0.3)

    # IQ time domain - Q channel
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(time_us, np.imag(iq_data[:len(time_us)]), linewidth=0.5)
    ax3.set_xlabel('Time (us)')
    ax3.set_ylabel('Amplitude (Q)')
    ax3.set_title('IQ Time Domain - Quadrature')
    ax3.grid(True, alpha=0.3)

    # Constellation
    ax4 = fig.add_subplot(gs[2, 0])
    # Downsample for constellation plot
    downsample = int(iq_meta.sample_rate_hz / 5e6)
    constellation_samples = iq_data[::downsample][:5000]
    ax4.scatter(np.real(constellation_samples), np.imag(constellation_samples),
                alpha=0.1, s=1)
    ax4.set_xlabel('In-Phase')
    ax4.set_ylabel('Quadrature')
    ax4.set_title('Constellation Diagram')
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')

    # Spectrogram
    ax5 = fig.add_subplot(gs[2, 1])
    # Use shorter segment for faster computation
    spec_samples = min(50000, len(iq_data))
    from matplotlib import mlab
    Sxx, f, t = mlab.specgram(
        iq_data[:spec_samples],
        NFFT=256,
        Fs=iq_meta.sample_rate_hz,
        noverlap=128
    )
    f_shifted = (f - iq_meta.sample_rate_hz/2) / 1e6
    ax5.pcolormesh(t * 1e6, f_shifted,
                   10 * np.log10(Sxx + 1e-20), shading='auto', cmap='viridis')
    ax5.set_xlabel('Time (us)')
    ax5.set_ylabel('Frequency Offset (MHz)')
    ax5.set_title('Spectrogram')

    plt.tight_layout()
    plt.savefig('example1_fdma_transponder.png', dpi=150)
    print("\nPlot saved as 'example1_fdma_transponder.png'")


def example2_tdma_carrier():
    """Example 2: TDMA carrier with bursting."""
    print("\n" + "="*80)
    print("Example 2: TDMA Carrier Bursting")
    print("="*80)

    # Create TDMA carrier
    tdma_carrier = Carrier(
        frequency_offset_hz=0.0,
        cn_db=20.0,  # 20 dB C/N
        symbol_rate_sps=10e6,  # 10 Msps
        modulation=ModulationType.QPSK,
        carrier_type=CarrierType.TDMA,
        burst_time_s=0.0001,  # 100 us burst
        duty_cycle=0.3,  # 30% duty cycle
        name="TDMA Burst"
    )

    transponder = Transponder(
        center_frequency_hz=14.25e9,  # 14.25 GHz
        bandwidth_hz=20e6,  # 20 MHz
        noise_power_density_watts_per_hz=5e-16,
        carriers=[tdma_carrier],
        name="TDMA Transponder"
    )

    print(f"\n{transponder}")
    print(f"  {tdma_carrier}")
    print(f"  Frame period: {tdma_carrier.frame_period_s*1e6:.1f} us")
    print(f"  Guard time: {tdma_carrier.guard_time_s*1e6:.1f} us")

    # Generate IQ
    print("\nGenerating IQ data...")
    iq_data, iq_meta = generate_iq(
        transponder,
        duration_s=0.002,  # 2 ms to see multiple bursts
        seed=42
    )

    # Generate PSD
    print("Generating PSD...")
    frequencies, psd, psd_meta = generate_psd(
        transponder,
        rbw_hz=5e3,  # 5 kHz
        vbw_hz=5e3,
    )

    # Plot
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 2, figure=fig)

    # Time domain - amplitude
    ax1 = fig.add_subplot(gs[0, :])
    time_us = np.arange(len(iq_data)) / iq_meta.sample_rate_hz * 1e6
    amplitude = np.abs(iq_data)
    ax1.plot(time_us, amplitude, linewidth=0.5)
    ax1.set_xlabel('Time (us)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('TDMA Burst Pattern - Amplitude')
    ax1.grid(True, alpha=0.3)

    # Zoom on single burst
    ax2 = fig.add_subplot(gs[1, 0])
    burst_samples = int(tdma_carrier.burst_time_s * iq_meta.sample_rate_hz * 1.5)
    time_burst = np.arange(burst_samples) / iq_meta.sample_rate_hz * 1e6
    ax2.plot(time_burst, np.abs(iq_data[:burst_samples]), linewidth=0.5)
    ax2.set_xlabel('Time (us)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Single Burst - Zoomed')
    ax2.grid(True, alpha=0.3)

    # I/Q scatter
    ax3 = fig.add_subplot(gs[1, 1])
    # Only plot samples during burst (non-zero amplitude)
    burst_threshold = np.max(amplitude) * 0.1
    burst_samples_idx = amplitude > burst_threshold
    downsample_factor = max(1, int(np.sum(burst_samples_idx) / 5000))
    burst_iq = iq_data[burst_samples_idx][::downsample_factor]
    ax3.scatter(np.real(burst_iq), np.imag(burst_iq), alpha=0.3, s=1)
    ax3.set_xlabel('In-Phase')
    ax3.set_ylabel('Quadrature')
    ax3.set_title('Constellation (During Burst)')
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')

    # Spectrogram
    ax4 = fig.add_subplot(gs[2, :])
    from matplotlib import mlab
    Sxx, f, t = mlab.specgram(
        iq_data,
        NFFT=256,
        Fs=iq_meta.sample_rate_hz,
        noverlap=200
    )
    f_shifted = (f - iq_meta.sample_rate_hz/2) / 1e6
    pcm = ax4.pcolormesh(t * 1e6, f_shifted,
                         10 * np.log10(Sxx + 1e-20), shading='auto', cmap='viridis')
    ax4.set_xlabel('Time (us)')
    ax4.set_ylabel('Frequency Offset (MHz)')
    ax4.set_title('Spectrogram - TDMA Bursts')
    plt.colorbar(pcm, ax=ax4, label='Power (dB)')

    plt.tight_layout()
    plt.savefig('example2_tdma_bursting.png', dpi=150)
    print("\nPlot saved as 'example2_tdma_bursting.png'")


def example3_multi_transponder_beam():
    """Example 3: Multi-transponder beam with 10 transponders."""
    print("\n" + "="*80)
    print("Example 3: Multi-Transponder Beam (10 Transponders)")
    print("="*80)

    np.random.seed(42)  # For reproducible carrier generation

    transponder_bandwidth_hz = 36e6  # 36 MHz per transponder
    base_frequency_hz = 12.5e9  # 12.5 GHz center
    noise_density = 1e-16

    transponders = []

    for t_idx in range(6):
        # Position transponder (adjacent spacing)
        transponder_offset = (t_idx - 4.5) * transponder_bandwidth_hz
        transponder_center = base_frequency_hz + transponder_offset

        # Random number of carriers (2-20)
        num_carriers = np.random.randint(2, 21)

        # Create transponder
        transponder = Transponder(
            center_frequency_hz=transponder_center,
            bandwidth_hz=transponder_bandwidth_hz,
            noise_power_density_watts_per_hz=noise_density,
            name=f"Transponder {t_idx+1}"
        )

        # Populate with random carriers
        carriers_created = transponder.populate_with_random_carriers(
            num_carriers=num_carriers,
            symbol_rate_range_sps=(100e3, 20e6),
            cn_range_db=(6.0, 30.0),
            rrc_rolloff_values=[0.20, 0.25, 0.35],
            seed=42 + t_idx,  # Different seed per transponder
            name_prefix=f"T{t_idx+1}-C"
        )

        print(f"Transponder {t_idx+1}: Created {carriers_created}/{num_carriers} carriers")
        transponders.append(transponder)

    # Create beam
    beam = Beam(
        band=Band.KA,
        polarization=Polarization.RHCP,
        direction=BeamDirection.DOWNLINK,
        transponders=transponders,
        name="Asia-Pacific Beam"
    )

    print(f"\n{beam}")
    for transponder in beam.transponders:
        print(f"\n  {transponder}")
        for carrier in transponder.carriers:
            print(f"    - {carrier}")

    # Generate PSD
    print("\nGenerating beam PSD...")
    frequencies, psd, psd_meta = generate_psd(
        beam,
        rbw_hz=10e3,  # 10 kHz
        vbw_hz=1000,  # VBW = 1% of RBW for smoothing
    )

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 6))

    # PSD
    ax.plot((frequencies - beam.center_frequency_hz) / 1e6, psd, linewidth=0.5)
    ax.set_xlabel('Frequency Offset from Beam Center (MHz)')
    ax.set_ylabel('PSD (dBm/Hz)')
    ax.set_title('Multi-Transponder Beam Spectrum (10 Transponders)')
    ax.grid(True, alpha=0.3)

    # Mark transponder boundaries
    for idx, tp in enumerate(beam.transponders):
        offset_lower = (tp.lower_frequency_hz - beam.center_frequency_hz) / 1e6
        offset_upper = (tp.upper_frequency_hz - beam.center_frequency_hz) / 1e6
        ax.axvline(offset_lower, color='r', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(offset_upper, color='r', linestyle='--', alpha=0.5, linewidth=1)
        ax.text((offset_lower + offset_upper) / 2, ax.get_ylim()[1] - 5,
                f"T{idx+1}", ha='center', fontsize=6, rotation=90, va='top')

    plt.tight_layout()
    plt.savefig('example3_multi_transponder_beam.png', dpi=150)
    print("\nPlot saved as 'example3_multi_transponder_beam.png'")


def example4_validation_tests():
    """Example 4: Validation - overlapping and out-of-bounds carriers."""
    print("\n" + "="*80)
    print("Example 4: Validation Tests")
    print("="*80)

    # Test 1: Carrier extends beyond transponder
    print("\nTest 1: Carrier extends beyond transponder bandwidth")
    try:
        carrier_oob = Carrier(
            frequency_offset_hz=15e6,  # Too far from center
            cn_db=15.0,  # 15 dB C/N
            symbol_rate_sps=10e6,  # Large bandwidth
            modulation=ModulationType.QPSK,
            carrier_type=CarrierType.FDMA,
        )

        transponder_oob = Transponder(
            center_frequency_hz=12.5e9,
            bandwidth_hz=20e6,  # Only 20 MHz
            noise_power_density_watts_per_hz=1e-15,
            carriers=[carrier_oob],
        )
        print("  ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"  OK: Correctly caught error: {str(e)[:100]}...")

    # Test 2: Overlapping carriers (not allowed)
    print("\nTest 2: Overlapping carriers without permission")
    try:
        carrier_a = Carrier(
            frequency_offset_hz=0.0,
            cn_db=15.0,  # 15 dB C/N
            symbol_rate_sps=10e6,
            modulation=ModulationType.QPSK,
            carrier_type=CarrierType.FDMA,
        )
        carrier_b = Carrier(
            frequency_offset_hz=5e6,  # Overlaps with carrier_a
            cn_db=15.0,  # 15 dB C/N
            symbol_rate_sps=10e6,
            modulation=ModulationType.QPSK,
            carrier_type=CarrierType.FDMA,
        )

        transponder_overlap = Transponder(
            center_frequency_hz=12.5e9,
            bandwidth_hz=50e6,
            noise_power_density_watts_per_hz=1e-15,
            carriers=[carrier_a, carrier_b],
            allow_overlap=False,
        )
        print("  ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"  OK: Correctly caught error: {str(e)[:100]}...")

    # Test 3: Overlapping carriers (allowed)
    print("\nTest 3: Overlapping carriers with permission")
    try:
        carrier_a = Carrier(
            frequency_offset_hz=0.0,
            cn_db=15.0,  # 15 dB C/N
            symbol_rate_sps=10e6,
            modulation=ModulationType.QPSK,
            carrier_type=CarrierType.FDMA,
            name="Carrier A"
        )
        carrier_b = Carrier(
            frequency_offset_hz=5e6,
            cn_db=15.0,  # 15 dB C/N
            symbol_rate_sps=10e6,
            modulation=ModulationType.QPSK,
            carrier_type=CarrierType.FDMA,
            name="Carrier B"
        )

        transponder_ok = Transponder(
            center_frequency_hz=12.5e9,
            bandwidth_hz=50e6,
            noise_power_density_watts_per_hz=1e-15,
            carriers=[carrier_a, carrier_b],
            allow_overlap=True,  # Allow overlap
        )
        print(f"  OK: Successfully created transponder with overlapping carriers")
        print(f"    {transponder_ok}")

        # Generate and plot PSD
        frequencies, psd, _ = generate_psd(transponder_ok, rbw_hz=10e3, vbw_hz=10e3)

        plt.figure(figsize=(12, 6))
        plt.plot((frequencies - transponder_ok.center_frequency_hz) / 1e6, psd)
        plt.xlabel('Frequency Offset (MHz)')
        plt.ylabel('PSD (dBm/Hz)')
        plt.title('Overlapping Carriers (Allowed)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('example4_overlapping_carriers.png', dpi=150)
        print("    Plot saved as 'example4_overlapping_carriers.png'")

    except ValueError as e:
        print(f"  ERROR: Unexpected error: {e}")

    # Test 4: TDMA without burst parameters
    print("\nTest 4: TDMA carrier without burst parameters")
    try:
        carrier_tdma_bad = Carrier(
            frequency_offset_hz=0.0,
            cn_db=15.0,  # 15 dB C/N
            symbol_rate_sps=10e6,
            modulation=ModulationType.QPSK,
            carrier_type=CarrierType.TDMA,
            # Missing burst_time and duty_cycle
        )
        print("  ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"  OK: Correctly caught error: {str(e)[:80]}...")

    # Test 5: FDMA with burst parameters
    print("\nTest 5: FDMA carrier with burst parameters")
    try:
        carrier_fdma_bad = Carrier(
            frequency_offset_hz=0.0,
            cn_db=15.0,  # 15 dB C/N
            symbol_rate_sps=10e6,
            modulation=ModulationType.QPSK,
            carrier_type=CarrierType.FDMA,
            burst_time_s=0.001,  # Should not have this
            duty_cycle=0.5,
        )
        print("  ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"  OK: Correctly caught error: {str(e)[:80]}...")


def example5_modulation_comparison():
    """Example 5: Compare different modulation types."""
    print("\n" + "="*80)
    print("Example 5: Modulation Type Comparison")
    print("="*80)

    modulations = [
        ModulationType.BPSK,
        ModulationType.QPSK,
        ModulationType.QAM16,
        ModulationType.APSK16,
        ModulationType.APSK32,
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, mod_type in enumerate(modulations):
        print(f"\nGenerating {mod_type.value} carrier...")

        # Create carrier
        carrier = Carrier(
            frequency_offset_hz=0.0,
            cn_db=18.0,  # 18 dB C/N
            symbol_rate_sps=5e6,
            modulation=mod_type,
            carrier_type=CarrierType.FDMA,
            name=f"{mod_type.value} Carrier"
        )

        transponder = Transponder(
            center_frequency_hz=12.5e9,
            bandwidth_hz=20e6,
            noise_power_density_watts_per_hz=5e-16,
            carriers=[carrier],
        )

        # Generate IQ
        iq_data, iq_meta = generate_iq(transponder, duration_s=0.0005, seed=42)

        # Downsample for constellation
        downsample = int(iq_meta.sample_rate_hz / carrier.symbol_rate_sps / 2)
        constellation = iq_data[::downsample][:2000]

        # Plot constellation
        axes[idx].scatter(np.real(constellation), np.imag(constellation),
                         alpha=0.3, s=2)
        axes[idx].set_xlabel('In-Phase')
        axes[idx].set_ylabel('Quadrature')
        axes[idx].set_title(f'{mod_type.value}')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].axis('equal')
        max_val = np.max(np.abs(constellation)) * 1.2
        axes[idx].set_xlim([-max_val, max_val])
        axes[idx].set_ylim([-max_val, max_val])

    # Hide last subplot
    axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig('example5_modulation_comparison.png', dpi=150)
    print("\nPlot saved as 'example5_modulation_comparison.png'")


def main():
    """Run all examples."""
    print("\n" + "#"*80)
    print("# Satellite Spectrum Emulator - Examples and Test Cases")
    print("#"*80)

    # Run examples
    example1_single_transponder_fdma()
    example2_tdma_carrier()
    example3_multi_transponder_beam()
    example4_validation_tests()
    example5_modulation_comparison()

    print("\n" + "#"*80)
    print("# All examples completed successfully!")
    print("#"*80)
    print("\nGenerated plots:")
    print("  - example1_fdma_transponder.png")
    print("  - example2_tdma_bursting.png")
    print("  - example3_multi_transponder_beam.png")
    print("  - example4_overlapping_carriers.png")
    print("  - example5_modulation_comparison.png")
    print()


if __name__ == "__main__":
    main()
