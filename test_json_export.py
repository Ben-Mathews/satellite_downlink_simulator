"""Test script for JSON export functionality."""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add the apps directory to the path
apps_dir = Path(__file__).parent / "apps" / "rf_pattern_of_life_example"
sys.path.insert(0, str(apps_dir))

from carrier_generator import CarrierGenerator
from interferer_generator import InterfererGenerator
from psd_simulator import PSDSimulator

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from satellite_downlink_simulator.simulation import SpectrumRecord

def test_json_export():
    """Test JSON export with a short simulation."""
    print("=" * 70)
    print("TESTING JSON EXPORT FEATURE")
    print("=" * 70)
    print()

    # Generate carriers
    print("Step 1: Generating carriers...")
    carrier_gen = CarrierGenerator(seed=42)
    carrier_config = carrier_gen.generate_carriers(
        num_static_per_xpdr=(3, 5),  # Fewer carriers for quick test
        num_dynamic=5
    )
    print(f"  Created {len(carrier_config.carriers)} carriers")
    print()

    # Generate interferers
    print("Step 2: Generating interferers...")
    interferer_gen = InterfererGenerator(seed=43)
    interferers = interferer_gen.generate_interferers(
        carrier_configs=carrier_config.carriers,
        num_long_duration=1,
        num_short_duration=2
    )
    print(f"  Created {len(interferers)} interferers")
    print()

    # Create simulator
    print("Step 3: Creating simulator...")
    simulator = PSDSimulator(
        carrier_config=carrier_config,
        interferer_configs=interferers,
        rbw_hz=100e3,  # 100 kHz RBW for faster generation
        vbw_hz=1e3
    )
    print()

    # Run short simulation (15 minutes, 3 snapshots)
    print("Step 4: Running test simulation (15 min, 3 snapshots)...")
    start_time = datetime(2025, 1, 15, 0, 0, 0)  # Fixed time for testing
    snapshots, metadata = simulator.run_simulation(
        duration_min=10,  # 10 minutes
        snapshot_interval_min=5,  # 5 minute intervals = 3 snapshots
        start_datetime=start_time,
        export_json=True
    )
    print()

    print("Step 5: Verifying JSON file...")
    json_filename = f"spectrum_records_{start_time.strftime('%Y%m%d-%H%M%S')}.json"
    json_filepath = apps_dir / "output" / json_filename

    if not json_filepath.exists():
        print(f"  ERROR: JSON file not found at {json_filepath}")
        return False

    print(f"  JSON file found: {json_filepath}")
    file_size_mb = os.path.getsize(json_filepath) / 1e6
    print(f"  File size: {file_size_mb:.2f} MB")
    print()

    # Load and verify
    print("Step 6: Loading and verifying JSON...")
    try:
        records = SpectrumRecord.from_file(str(json_filepath))
        print(f"  Loaded {len(records)} SpectrumRecord objects")

        if len(records) != len(snapshots):
            print(f"  ERROR: Expected {len(snapshots)} records, got {len(records)}")
            return False

        # Verify first record
        record = records[0]
        print(f"\n  First record details:")
        print(f"    Timestamp: {record.timestamp.isoformat()}")
        print(f"    Center freq: {record.cf_hz/1e9:.3f} GHz")
        print(f"    Bandwidth: {record.bw_hz/1e6:.1f} MHz")
        print(f"    RBW: {record.rbw_hz/1e3:.1f} kHz")
        print(f"    VBW: {record.vbw_hz/1e3:.1f} kHz")
        print(f"    Beams: {len(record.beams)}")
        if record.beams:
            beam = record.beams[0]
            print(f"      Beam 0: {len(beam.transponders)} transponders")
            total_carriers = sum(len(t.carriers) for t in beam.transponders)
            print(f"              {total_carriers} carriers")
        print(f"    Interferers: {len(record.interferers)}")
        for interferer in record.interferers:
            print(f"      {interferer.carrier.name}: {interferer.current_frequency_hz/1e9:.6f} GHz")
            print(f"        Sweeping: {interferer.is_sweeping}")
            print(f"        Overlapping: {interferer.overlapping_transponders}")

        # Decompress and verify PSD
        print(f"\n  Decompressing PSD data...")
        psd_decompressed = record.get_psd()
        print(f"    PSD shape: {psd_decompressed.shape}")
        print(f"    PSD range: {psd_decompressed.min():.1f} to {psd_decompressed.max():.1f} dBm/Hz")

        print()
        print("=" * 70)
        print("TEST PASSED!")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"  ERROR during verification: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_json_export()
    sys.exit(0 if success else 1)
