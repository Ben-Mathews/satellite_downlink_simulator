"""Utility script to regenerate visualizations from spectrum_records JSON file.

This script reads a spectrum_records_*.json file and regenerates all the plots
that were originally created during main.py execution.
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
from typing import List
import attrs

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from satellite_downlink_simulator.simulation import SpectrumRecord
from visualization import Visualizer


@attrs.define
class SimulationMetadata:
    """Metadata for complete simulation (reconstructed from SpectrumRecords)."""
    duration_min: float
    num_snapshots: int
    snapshot_interval_min: float
    rbw_hz: float
    vbw_hz: float
    sample_rate_hz: float
    fft_size: int
    num_transponders: int
    transponder_bandwidth_hz: float
    total_bandwidth_hz: float
    center_frequency_hz: float
    total_carriers: int
    static_carriers: int
    dynamic_carriers: int
    total_interferers: int


def load_and_process_records(json_file: str) -> tuple:
    """Load spectrum records and extract data for visualization.

    Args:
        json_file: Path to spectrum_records JSON file

    Returns:
        Tuple of (time_array, frequency_array, psd_array, metadata, activity_log)
    """
    print(f"Loading spectrum records from: {json_file}")

    # Load records
    records = SpectrumRecord.from_file(json_file)

    if len(records) == 0:
        raise ValueError("No spectrum records found in file")

    print(f"  Loaded {len(records)} spectrum records")

    # Sort records by timestamp
    records = sorted(records, key=lambda r: r.timestamp)

    # Extract timestamps and convert to time_min
    start_time = records[0].timestamp
    time_array = np.array([
        (record.timestamp - start_time).total_seconds() / 60.0
        for record in records
    ])

    print(f"  Time range: {time_array[0]:.1f} - {time_array[-1]:.1f} minutes")

    # Extract frequency array from first record
    # We need to reconstruct it from the PSD shape and bandwidth/center frequency
    first_record = records[0]
    psd_shape = first_record.psd_shape
    num_freq_points = psd_shape[0]

    # Calculate frequency array
    # Center frequency ± bandwidth/2
    freq_min = first_record.cf_hz - first_record.bw_hz / 2
    freq_max = first_record.cf_hz + first_record.bw_hz / 2
    frequency_array = np.linspace(freq_min, freq_max, num_freq_points)

    print(f"  Frequency range: {frequency_array[0]/1e9:.3f} - {frequency_array[-1]/1e9:.3f} GHz")
    print(f"  Frequency points: {num_freq_points}")

    # Extract PSD arrays
    print("  Decompressing PSD data...")
    psd_list = []
    for i, record in enumerate(records):
        if i % 50 == 0 and i > 0:
            print(f"    Processing record {i}/{len(records)}")
        psd = record.get_psd()
        psd_list.append(psd)

    psd_array = np.array(psd_list)
    print(f"  PSD array shape: {psd_array.shape}")

    # Create activity log
    print("  Building activity log...")
    activity_log = []
    for record in records:
        # Count carriers across all beams/transponders
        num_carriers = 0
        for beam in record.beams:
            for transponder in beam.transponders:
                num_carriers += len(transponder.carriers)

        num_interferers = len(record.interferers)

        activity_log.append({
            'time_min': (record.timestamp - start_time).total_seconds() / 60.0,
            'num_carriers': num_carriers,
            'num_interferers': num_interferers
        })

    print(f"    Max carriers: {max(a['num_carriers'] for a in activity_log)}")
    print(f"    Max interferers: {max(a['num_interferers'] for a in activity_log)}")

    # Reconstruct metadata
    print("  Reconstructing metadata...")

    # Calculate duration and interval
    duration_min = time_array[-1] - time_array[0]
    if len(records) > 1:
        # Calculate average interval
        intervals = np.diff(time_array)
        snapshot_interval_min = np.mean(intervals)
    else:
        snapshot_interval_min = 0.0

    # Get rbw/vbw from first record
    rbw_hz = first_record.rbw_hz
    vbw_hz = first_record.vbw_hz

    # Calculate sample rate and FFT size
    sample_rate_hz = first_record.bw_hz  # Approximate
    fft_size = num_freq_points

    # Count transponders and carriers
    num_transponders = 0
    total_carriers = 0
    all_carrier_names = set()

    for record in records:
        for beam in record.beams:
            num_transponders = max(num_transponders, len(beam.transponders))
            for transponder in beam.transponders:
                total_carriers = max(total_carriers, len(transponder.carriers))
                for carrier in transponder.carriers:
                    all_carrier_names.add(carrier.name)

    # Count unique carriers across all records
    total_unique_carriers = len(all_carrier_names)

    # Count unique interferers
    all_interferer_names = set()
    for record in records:
        for interferer in record.interferers:
            all_interferer_names.add(interferer.carrier.name)
    total_interferers = len(all_interferer_names)

    # We don't have static/dynamic distinction in SpectrumRecords, so approximate
    static_carriers = total_unique_carriers  # Conservative estimate
    dynamic_carriers = 0  # Unknown from records alone

    # Transponder bandwidth (use first transponder as reference)
    transponder_bandwidth_hz = 36e6  # Default, can't determine from records
    if records[0].beams:
        if records[0].beams[0].transponders:
            transponder_bandwidth_hz = records[0].beams[0].transponders[0].bandwidth_hz

    metadata = SimulationMetadata(
        duration_min=duration_min,
        num_snapshots=len(records),
        snapshot_interval_min=snapshot_interval_min,
        rbw_hz=rbw_hz,
        vbw_hz=vbw_hz,
        sample_rate_hz=sample_rate_hz,
        fft_size=fft_size,
        num_transponders=num_transponders,
        transponder_bandwidth_hz=transponder_bandwidth_hz,
        total_bandwidth_hz=first_record.bw_hz,
        center_frequency_hz=first_record.cf_hz,
        total_carriers=total_unique_carriers,
        static_carriers=static_carriers,
        dynamic_carriers=dynamic_carriers,
        total_interferers=total_interferers
    )

    print(f"  Metadata summary:")
    print(f"    Duration: {metadata.duration_min:.1f} minutes")
    print(f"    Snapshots: {metadata.num_snapshots}")
    print(f"    Interval: {metadata.snapshot_interval_min:.1f} minutes")
    print(f"    Total carriers: {metadata.total_carriers}")
    print(f"    Total interferers: {metadata.total_interferers}")

    return time_array, frequency_array, psd_array, metadata, activity_log


def main():
    """Main entry point for spectrum records utility."""

    parser = argparse.ArgumentParser(
        description='Regenerate visualizations from spectrum_records JSON file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Regenerate plots from a spectrum_records file
  python spectrum_records_utility.py spectrum_records_20250115-000000.json

  # Specify custom output directory
  python spectrum_records_utility.py spectrum_records_20250115-000000.json --output-dir custom_plots

  # Skip animated spectrogram (faster)
  python spectrum_records_utility.py spectrum_records_20250115-000000.json --no-animation

  # Generate MP4 instead of GIF
  python spectrum_records_utility.py spectrum_records_20250115-000000.json --format mp4
        """
    )

    # Input file
    parser.add_argument('json_file', type=str,
                       help='Path to spectrum_records JSON file')

    # Output options
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots (default: plots/ in same directory as JSON file)')

    # Animation options
    parser.add_argument('--no-animation', action='store_true',
                       help='Skip animated spectrogram generation (faster)')

    parser.add_argument('--format', type=str, choices=['gif', 'mp4'], default='gif',
                       help='Animation format (default: gif)')

    parser.add_argument('--fps', type=int, default=15,
                       help='Animation frames per second (default: 15)')

    parser.add_argument('--frame-decimation', type=int, default=1,
                       help='Frame decimation factor (default: 1, no decimation)')

    args = parser.parse_args()

    # Validate input file
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"ERROR: File not found: {json_path}")
        sys.exit(1)

    if not json_path.suffix == '.json':
        print(f"WARNING: File does not have .json extension: {json_path}")

    # Determine output directory
    if args.output_dir is None:
        # Default: plots/ in same directory as JSON file
        output_dir = json_path.parent / "plots"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(exist_ok=True)

    # Print banner
    print("=" * 70)
    print("SPECTRUM RECORDS VISUALIZATION UTILITY")
    print("=" * 70)
    print()
    print(f"Input file: {json_path}")
    print(f"Output directory: {output_dir}")
    print()

    # Load and process records
    try:
        time_array, frequency_array, psd_array, metadata, activity_log = load_and_process_records(str(json_path))
    except Exception as e:
        print(f"\nERROR: Failed to load spectrum records: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print()

    # Create visualizer
    print("=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    print()

    viz = Visualizer(
        time_array=time_array,
        frequency_array=frequency_array,
        psd_array=psd_array,
        metadata=metadata,
        output_dir=str(output_dir)
    )

    # Generate plots
    print()
    viz.create_waterfall_plot()
    viz.create_average_spectrum()
    viz.create_snapshot_comparison()
    viz.create_activity_timeline(activity_log=activity_log)

    if not args.no_animation:
        print()
        viz.create_animated_spectrogram(
            fps=args.fps,
            frame_decimation=args.frame_decimation,
            output_format=args.format
        )
    else:
        print("\nSkipping animated spectrogram (--no-animation specified)")

    # Summary
    print()
    print("=" * 70)
    print("VISUALIZATION COMPLETE!")
    print("=" * 70)
    print()
    print(f"Output directory: {output_dir}")
    print()
    print("Generated files:")
    print("  - waterfall_plot.png")
    print("  - average_spectrum.png")
    print("  - snapshot_comparison.png")
    print("  - activity_timeline.png")
    if not args.no_animation:
        print(f"  - animated_spectrogram.{args.format}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
