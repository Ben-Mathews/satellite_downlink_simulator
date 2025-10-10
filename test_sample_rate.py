"""Test script for configurable sample rate feature."""

import numpy as np
from satellite_downlink_simulator import Transponder, Carrier, generate_iq, ModulationType, CarrierType

# Create a simple transponder with one carrier
transponder = Transponder(
    center_frequency_hz=12e9,
    bandwidth_hz=36e6,
    noise_power_density_watts_per_hz=1e-15,
    noise_rolloff=0.25,
)

# Add a single FDMA carrier
carrier = Carrier(
    name="Test Carrier",
    frequency_offset_hz=0,
    cn_db=15.0,
    symbol_rate_sps=10e6,
    modulation=ModulationType.QPSK,
    rrc_rolloff=0.35,
    carrier_type=CarrierType.FDMA,
)
transponder.add_carrier(carrier)

print("Testing configurable sample rate feature...")
print(f"Transponder bandwidth: {transponder.bandwidth_hz / 1e6:.2f} MHz")
print()

# Test 1: Default sample rate (should use 1.25x bandwidth)
print("Test 1: Default sample rate (1.25× bandwidth)")
iq_data, metadata = generate_iq(transponder, duration_s=0.001, seed=42)
expected_rate = 1.25 * transponder.bandwidth_hz
print(f"  Expected sample rate: {expected_rate / 1e6:.2f} MHz")
print(f"  Actual sample rate:   {metadata.sample_rate_hz / 1e6:.2f} MHz")
print(f"  [PASS]" if abs(metadata.sample_rate_hz - expected_rate) < 1 else "  [FAIL]")
print()

# Test 2: Custom sample rate (2x bandwidth)
print("Test 2: Custom sample rate (2× bandwidth)")
custom_rate = 2.0 * transponder.bandwidth_hz
iq_data, metadata = generate_iq(transponder, duration_s=0.001, seed=42, sample_rate_hz=custom_rate)
print(f"  Requested sample rate: {custom_rate / 1e6:.2f} MHz")
print(f"  Actual sample rate:    {metadata.sample_rate_hz / 1e6:.2f} MHz")
print(f"  [PASS]" if abs(metadata.sample_rate_hz - custom_rate) < 1 else "  [FAIL]")
print()

# Test 3: Minimum valid sample rate (exactly equal to bandwidth)
print("Test 3: Minimum valid sample rate (exactly equal to bandwidth)")
min_rate = transponder.bandwidth_hz
iq_data, metadata = generate_iq(transponder, duration_s=0.001, seed=42, sample_rate_hz=min_rate)
print(f"  Requested sample rate: {min_rate / 1e6:.2f} MHz")
print(f"  Actual sample rate:    {metadata.sample_rate_hz / 1e6:.2f} MHz")
print(f"  [PASS]" if abs(metadata.sample_rate_hz - min_rate) < 1 else "  [FAIL]")
print()

# Test 4: Invalid sample rate (should raise ValueError)
print("Test 4: Invalid sample rate (below Nyquist criterion)")
invalid_rate = 0.5 * transponder.bandwidth_hz
try:
    iq_data, metadata = generate_iq(transponder, duration_s=0.001, seed=42, sample_rate_hz=invalid_rate)
    print(f"  [FAIL] - Should have raised ValueError")
except ValueError as e:
    print(f"  [PASS] - Correctly raised ValueError:")
    print(f"    {str(e)}")
print()

print("All tests completed!")
