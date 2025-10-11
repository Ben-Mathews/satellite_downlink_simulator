"""Main CLI for RF Pattern of Life simulation.

Orchestrates carrier generation, interferer generation, PSD simulation, and visualization.
"""

import argparse
import os
import json
from pathlib import Path
from datetime import datetime

from carrier_generator import CarrierGenerator
from interferer_generator import InterfererGenerator
from psd_simulator import PSDSimulator
from visualization import Visualizer
import numpy as np


def main():
    """Main entry point for pattern-of-life simulation."""

    parser = argparse.ArgumentParser(
        description='RF Pattern of Life - 24-Hour Satellite Spectrum Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full 24-hour simulation with default settings
  python main.py

  # Regenerate carriers and run simulation
  python main.py --regenerate

  # Custom simulation parameters
  python main.py --duration-min 1440 --interval-min 5 --rbw-hz 100000 --vbw-hz 1000

  # Custom carrier/interferer counts
  python main.py --static-min 5 --static-max 15 --dynamic 20 --interferers-long 3 --interferers-short 10

  # Use custom seed for reproducibility
  python main.py --seed-carriers 123 --seed-interferers 456
        """
    )

    # Configuration options
    parser.add_argument('--regenerate', action='store_true',
                       help='Regenerate carrier configuration even if config.json exists')

    # Carrier generation parameters
    parser.add_argument('--static-min', type=int, default=5,
                       help='Minimum static carriers per transponder (default: 5)')
    parser.add_argument('--static-max', type=int, default=15,
                       help='Maximum static carriers per transponder (default: 15)')
    parser.add_argument('--dynamic', type=int, default=20,
                       help='Number of dynamic carriers (default: 20)')

    # Interferer generation parameters
    parser.add_argument('--interferers-long', type=int, default=2,
                       help='Number of long-duration interferers (default: 2)')
    parser.add_argument('--interferers-short', type=int, default=8,
                       help='Number of short-duration interferers (default: 8)')

    # Simulation parameters
    parser.add_argument('--duration-min', type=float, default=1440,
                       help='Simulation duration in minutes (default: 1440 = 24 hours)')
    parser.add_argument('--interval-min', type=float, default=5,
                       help='Snapshot interval in minutes (default: 5)')
    parser.add_argument('--rbw-hz', type=float, default=100e3,
                       help='Resolution bandwidth in Hz (default: 100000)')
    parser.add_argument('--vbw-hz', type=float, default=1e3,
                       help='Video bandwidth in Hz (default: 1000)')

    # Random seeds
    parser.add_argument('--seed-carriers', type=int, default=42,
                       help='Random seed for carrier generation (default: 42)')
    parser.add_argument('--seed-interferers', type=int, default=43,
                       help='Random seed for interferer generation (default: 43)')

    # Output options
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for results (default: output)')
    parser.add_argument('--plot-dir', type=str, default='plots',
                       help='Plot directory (relative to output dir) (default: plots)')
    parser.add_argument('--export-json', action='store_true',
                       help='Export SpectrumRecord objects to JSON file')
    parser.add_argument('--start-datetime', type=str, default=None,
                       help='Simulation start datetime (ISO format YYYY-MM-DDTHH:MM:SS, default: current time)')

    args = parser.parse_args()

    # Print banner
    print("=" * 70)
    print("RF PATTERN OF LIFE SIMULATOR")
    print("24-Hour Satellite Spectrum Simulation with Dynamic Carriers")
    print("=" * 70)
    print()

    # Setup paths
    script_dir = Path(__file__).parent
    config_file = script_dir / "config.json"
    output_dir = script_dir / args.output_dir
    plot_dir = output_dir / args.plot_dir

    # Create output directories
    output_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    print(f"Configuration:")
    print(f"  Output directory: {output_dir}")
    print(f"  Plot directory: {plot_dir}")
    print(f"  Config file: {config_file}")
    print()

    # Step 1: Generate or load carrier configuration
    print("=" * 70)
    print("STEP 1: CARRIER CONFIGURATION")
    print("=" * 70)
    print()

    if config_file.exists() and not args.regenerate:
        print(f"Loading existing configuration from {config_file}...")
        carrier_config = CarrierGenerator.load_config(str(config_file))
    else:
        print(f"Generating new carrier configuration...")
        print(f"  Static carriers per transponder: {args.static_min}-{args.static_max}")
        print(f"  Dynamic carriers: {args.dynamic}")
        print(f"  Random seed: {args.seed_carriers}")
        print()

        carrier_gen = CarrierGenerator(seed=args.seed_carriers)
        carrier_config = carrier_gen.generate_carriers(
            num_static_per_xpdr=(args.static_min, args.static_max),
            num_dynamic=args.dynamic
        )

        # Save configuration
        CarrierGenerator.save_config(carrier_config, str(config_file))

    print()
    print(f"Carrier Configuration Summary:")
    print(f"  Total carriers: {len(carrier_config.carriers)}")
    print(f"  Static carriers: {len([c for c in carrier_config.carriers if c.is_static])}")
    print(f"  Dynamic carriers: {len([c for c in carrier_config.carriers if not c.is_static])}")
    print()

    # Step 2: Generate interferers
    print("=" * 70)
    print("STEP 2: INTERFERER GENERATION")
    print("=" * 70)
    print()

    print(f"Generating CW interferers...")
    print(f"  Long-duration interferers: {args.interferers_long}")
    print(f"  Short-duration interferers: {args.interferers_short}")
    print(f"  Random seed: {args.seed_interferers}")
    print()

    interferer_gen = InterfererGenerator(seed=args.seed_interferers)
    interferers = interferer_gen.generate_interferers(
        carrier_configs=carrier_config.carriers,
        num_long_duration=args.interferers_long,
        num_short_duration=args.interferers_short
    )

    print()
    print(f"Interferer Summary:")
    print(f"  Total interferers: {len(interferers)}")
    sweeping = [i for i in interferers if i.sweep_type != 'none']
    print(f"  Sweeping interferers: {len(sweeping)}")
    print()

    # Step 3: Run PSD simulation
    print("=" * 70)
    print("STEP 3: PSD SIMULATION")
    print("=" * 70)
    print()

    # Parse start datetime
    if args.start_datetime:
        try:
            start_datetime = datetime.fromisoformat(args.start_datetime)
        except ValueError:
            print(f"Error: Invalid datetime format '{args.start_datetime}'")
            print(f"       Expected ISO format: YYYY-MM-DDTHH:MM:SS (e.g., 2025-01-15T00:00:00)")
            return
    else:
        start_datetime = datetime.now()

    print(f"Simulation Parameters:")
    print(f"  Start datetime: {start_datetime.isoformat()}")
    print(f"  Duration: {args.duration_min} minutes ({args.duration_min/60:.1f} hours)")
    print(f"  Snapshot interval: {args.interval_min} minutes")
    print(f"  Resolution bandwidth (RBW): {args.rbw_hz/1e3:.1f} kHz")
    print(f"  Video bandwidth (VBW): {args.vbw_hz/1e3:.1f} kHz")
    if args.export_json:
        print(f"  JSON export: ENABLED")
    print()

    simulator = PSDSimulator(
        carrier_config=carrier_config,
        interferer_configs=interferers,
        rbw_hz=args.rbw_hz,
        vbw_hz=args.vbw_hz
    )

    snapshots, metadata = simulator.run_simulation(
        duration_min=args.duration_min,
        snapshot_interval_min=args.interval_min,
        start_datetime=start_datetime,
        export_json=args.export_json
    )

    print()
    print(f"Simulation Complete!")
    print(f"  Generated {len(snapshots)} PSD snapshots")
    print(f"  Total bandwidth: {metadata.total_bandwidth_hz/1e6:.1f} MHz")
    print(f"  Frequency range: {snapshots[0].frequency_hz[0]/1e9:.3f} - {snapshots[0].frequency_hz[-1]/1e9:.3f} GHz")
    print()

    # Step 4: Save results
    print("=" * 70)
    print("STEP 4: SAVING RESULTS")
    print("=" * 70)
    print()

    simulator.save_results(snapshots, metadata, str(output_dir))
    print()

    # Step 5: Generate visualizations
    print("=" * 70)
    print("STEP 5: VISUALIZATION")
    print("=" * 70)
    print()

    # Extract data for visualization
    time_arr = np.array([s.time_min for s in snapshots])
    freq_arr = snapshots[0].frequency_hz
    psd_arr = np.array([s.psd_dbm_hz for s in snapshots])

    # Create activity log
    activity_log = []
    for s in snapshots:
        activity_log.append({
            'time_min': s.time_min,
            'num_carriers': s.num_carriers,
            'num_interferers': s.num_interferers
        })

    # Create visualizer and generate plots
    viz = Visualizer(
        time_array=time_arr,
        frequency_array=freq_arr,
        psd_array=psd_arr,
        metadata=metadata,
        output_dir=str(plot_dir)
    )

    viz.create_all_plots(activity_log=activity_log)

    print()

    # Final summary
    print("=" * 70)
    print("SIMULATION COMPLETE!")
    print("=" * 70)
    print()
    print(f"Results saved to: {output_dir}")
    print(f"Plots saved to: {plot_dir}")
    print()
    print(f"Generated files:")
    print(f"  - config.json                  (carrier configuration)")
    print(f"  - simulation_metadata.json     (simulation parameters)")
    print(f"  - psd_snapshots.npz            (PSD time-series data)")
    print(f"  - activity_log.json            (carrier/interferer activity)")
    if args.export_json:
        json_filename = f"spectrum_records_{start_datetime.strftime('%Y%m%d-%H%M%S')}.json"
        print(f"  - {json_filename:<30} (SpectrumRecord JSON export)")
    print(f"  - waterfall_plot.png           (spectrogram)")
    print(f"  - average_spectrum.png         (24-hour average)")
    print(f"  - snapshot_comparison.png      (selected time snapshots)")
    print(f"  - activity_timeline.png        (activity over time)")
    print()
    print(f"To regenerate with new carriers, use: python main.py --regenerate")
    print("=" * 70)


if __name__ == "__main__":
    main()
