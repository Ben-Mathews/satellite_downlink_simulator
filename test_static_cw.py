"""Test script for STATIC_CW carrier type."""

from satellite_downlink_simulator import Transponder, Carrier, generate_psd, ModulationType, CarrierType

# Create a transponder
transponder = Transponder(
    center_frequency_hz=12e9,
    bandwidth_hz=36e6,
    noise_power_density_watts_per_hz=1e-15,
    noise_rolloff=0.25,
)

print("Testing STATIC_CW carrier implementation...")
print("=" * 60)

# Test 1: Create a STATIC_CW carrier
print("\nTest 1: Create STATIC_CW carrier")
try:
    cw_carrier = Carrier(
        name="Test CW",
        frequency_offset_hz=5e6,  # 5 MHz offset
        cn_db=20.0,
        modulation=ModulationType.STATIC_CW,
        carrier_type=CarrierType.FDMA,
    )
    print(f"  [PASS] Created STATIC_CW carrier")
    print(f"  {cw_carrier}")
    print(f"  Bandwidth: {cw_carrier.bandwidth_hz} Hz")
except Exception as e:
    print(f"  [FAIL] {e}")

# Test 2: Verify symbol_rate_sps is not allowed for STATIC_CW
print("\nTest 2: Verify symbol_rate_sps rejected for STATIC_CW")
try:
    bad_carrier = Carrier(
        name="Bad CW",
        frequency_offset_hz=0,
        cn_db=20.0,
        modulation=ModulationType.STATIC_CW,
        carrier_type=CarrierType.FDMA,
        symbol_rate_sps=10e6,  # Should be rejected
    )
    print(f"  [FAIL] Should have raised ValueError")
except ValueError as e:
    print(f"  [PASS] Correctly rejected: {e}")

# Test 3: Verify modulated carriers require symbol_rate_sps
print("\nTest 3: Verify modulated carriers require symbol_rate_sps")
try:
    bad_carrier = Carrier(
        name="Bad QPSK",
        frequency_offset_hz=0,
        cn_db=20.0,
        modulation=ModulationType.QPSK,
        carrier_type=CarrierType.FDMA,
        # Missing symbol_rate_sps
    )
    print(f"  [FAIL] Should have raised ValueError")
except ValueError as e:
    print(f"  [PASS] Correctly rejected: {e}")

# Test 4: Add STATIC_CW to transponder and generate PSD
print("\nTest 4: Generate PSD with STATIC_CW carrier")
try:
    transponder.add_carrier(cw_carrier)

    # Also add a modulated carrier for comparison
    qpsk_carrier = Carrier(
        name="QPSK Carrier",
        frequency_offset_hz=-5e6,
        cn_db=18.0,
        symbol_rate_sps=10e6,
        modulation=ModulationType.QPSK,
        carrier_type=CarrierType.FDMA,
        rrc_rolloff=0.35,
    )
    transponder.add_carrier(qpsk_carrier)

    # Generate PSD
    freq, psd, metadata = generate_psd(transponder, rbw_hz=10e3, vbw_hz=1e3)

    print(f"  [PASS] Generated PSD with {len(transponder.carriers)} carriers")
    print(f"  Frequency points: {metadata.num_points}")
    print(f"  Carriers in transponder:")
    for carrier in transponder.carriers:
        print(f"    - {carrier}")

except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("All STATIC_CW tests completed!")
