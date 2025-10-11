"""24-hour PSD simulation for pattern-of-life analysis.

Generates time-series PSD data with dynamic carriers and interferers.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict
import json
from datetime import datetime, timedelta

from satellite_downlink_simulator import Transponder, Carrier, generate_psd, ModulationType, CarrierType
from satellite_downlink_simulator.simulation import SpectrumRecord, InterfererRecord, CarrierRecord
from satellite_downlink_simulator.objects import Beam, Band, Polarization, BeamDirection


@dataclass
class PSDSnapshot:
    """Single PSD snapshot at a specific time."""
    time_min: float  # Minutes since start
    frequency_hz: np.ndarray  # Frequency array
    psd_dbm_hz: np.ndarray  # PSD in dBm/Hz
    active_carriers: List[str]  # Names of active carriers
    active_interferers: List[str]  # Names of active interferers
    num_carriers: int  # Total active carriers
    num_interferers: int  # Total active interferers


@dataclass
class SimulationMetadata:
    """Metadata for complete simulation."""
    duration_min: float  # Total duration in minutes
    num_snapshots: int  # Number of PSD snapshots
    snapshot_interval_min: float  # Time between snapshots
    rbw_hz: float  # Resolution bandwidth
    vbw_hz: float  # Video bandwidth
    sample_rate_hz: float  # FFT sample rate
    fft_size: int  # FFT size

    # Beam configuration
    num_transponders: int
    transponder_bandwidth_hz: float
    total_bandwidth_hz: float
    center_frequency_hz: float

    # Carrier statistics
    total_carriers: int
    static_carriers: int
    dynamic_carriers: int
    total_interferers: int

    # Reproducibility
    seed_carriers: int
    seed_interferers: int


class PSDSimulator:
    """Simulates 24-hour pattern-of-life PSD data."""

    def __init__(self, carrier_config, interferer_configs: List,
                 rbw_hz: float = 10e3, vbw_hz: float = 1e3,
                 sample_rate_hz: float = None, fft_size: int = None):
        """Initialize simulator with configuration.

        Args:
            carrier_config: SimulationConfig from CarrierGenerator
            interferer_configs: List of InterfererConfig from InterfererGenerator
            rbw_hz: Resolution bandwidth (Hz)
            vbw_hz: Video bandwidth (Hz)
            sample_rate_hz: Sample rate for FFT (auto if None)
            fft_size: FFT size (auto if None)
        """
        self.carrier_config = carrier_config
        self.interferer_configs = interferer_configs
        self.rbw_hz = rbw_hz
        self.vbw_hz = vbw_hz
        self.sample_rate_hz = sample_rate_hz
        self.fft_size = fft_size

        # Create transponder objects
        self.transponders = []
        for xpdr_cfg in carrier_config.transponders:
            xpdr = Transponder(
                center_frequency_hz=xpdr_cfg['center_frequency_hz'],
                bandwidth_hz=xpdr_cfg['bandwidth_hz'],
                noise_power_density_watts_per_hz=xpdr_cfg['noise_power_density_watts_per_hz'],
                noise_rolloff=xpdr_cfg['noise_rolloff'],
            )
            self.transponders.append(xpdr)

        print(f"Initialized simulator with {len(self.transponders)} transponders")
        print(f"  RBW: {rbw_hz/1e3:.1f} kHz, VBW: {vbw_hz/1e3:.1f} kHz")

    def _populate_transponder_at_time(self, time_min: float) -> None:
        """Populate transponders with active carriers and interferers at given time.

        Args:
            time_min: Time in minutes since start
        """
        # Clear all carriers from transponders
        for xpdr in self.transponders:
            xpdr.carriers = []

        # Add active carriers
        for carrier_cfg in self.carrier_config.carriers:
            if carrier_cfg.is_active(time_min):
                xpdr_idx = carrier_cfg.transponder_idx
                carrier = Carrier(
                    name=carrier_cfg.name,
                    frequency_offset_hz=carrier_cfg.frequency_offset_hz,
                    cn_db=carrier_cfg.cn_db,
                    symbol_rate_sps=carrier_cfg.symbol_rate_sps,
                    modulation=ModulationType[carrier_cfg.modulation],
                    rrc_rolloff=carrier_cfg.rrc_rolloff,
                    carrier_type=CarrierType[carrier_cfg.carrier_type],
                )
                self.transponders[xpdr_idx].add_carrier(carrier)

        # Add active interferers as STATIC_CW carriers
        # Interferers can overlap with carriers, so temporarily enable overlap
        for interferer_cfg in self.interferer_configs:
            freq_offset = interferer_cfg.get_frequency_offset(time_min)
            if freq_offset is not None:
                xpdr_idx = interferer_cfg.transponder_idx
                cw_carrier = Carrier(
                    name=interferer_cfg.name,
                    frequency_offset_hz=freq_offset,
                    cn_db=interferer_cfg.cn_db,
                    modulation=ModulationType.STATIC_CW,
                    carrier_type=CarrierType.FDMA,
                )
                # Temporarily allow overlap for interferers
                original_overlap_setting = self.transponders[xpdr_idx].allow_overlap
                self.transponders[xpdr_idx].allow_overlap = True
                try:
                    self.transponders[xpdr_idx].add_carrier(cw_carrier)
                finally:
                    # Restore original setting
                    self.transponders[xpdr_idx].allow_overlap = original_overlap_setting

    def generate_snapshot(self, time_min: float) -> PSDSnapshot:
        """Generate PSD snapshot at specific time.

        Args:
            time_min: Time in minutes since start

        Returns:
            PSDSnapshot object
        """
        # Populate transponders with active carriers/interferers
        self._populate_transponder_at_time(time_min)

        # Count active carriers and interferers
        active_carriers = [c.name for xpdr in self.transponders
                          for c in xpdr.carriers
                          if not c.name.startswith('CW_')]
        active_interferers = [c.name for xpdr in self.transponders
                             for c in xpdr.carriers
                             if c.name.startswith('CW_')]

        # Generate PSD for each transponder
        all_freq = []
        all_psd = []

        for xpdr in self.transponders:
            freq, psd, metadata = generate_psd(
                xpdr,
                rbw_hz=self.rbw_hz,
                vbw_hz=self.vbw_hz,
                add_noise=True
            )
            all_freq.append(freq)
            all_psd.append(psd)

        # Concatenate all transponder PSDs
        # Note: There will be small gaps between transponders, but this is realistic
        frequency_hz = np.concatenate(all_freq)
        psd_dbm_hz = np.concatenate(all_psd)

        # Sort by frequency (should already be sorted, but just in case)
        sort_idx = np.argsort(frequency_hz)
        frequency_hz = frequency_hz[sort_idx]
        psd_dbm_hz = psd_dbm_hz[sort_idx]

        snapshot = PSDSnapshot(
            time_min=time_min,
            frequency_hz=frequency_hz,
            psd_dbm_hz=psd_dbm_hz,
            active_carriers=active_carriers,
            active_interferers=active_interferers,
            num_carriers=len(active_carriers),
            num_interferers=len(active_interferers)
        )

        return snapshot

    def run_simulation(self, duration_min: float = 1440,
                      snapshot_interval_min: float = 5,
                      start_datetime: Optional[datetime] = None,
                      export_json: bool = False) -> Tuple[List[PSDSnapshot], SimulationMetadata]:
        """Run complete 24-hour simulation.

        Args:
            duration_min: Total duration in minutes (default 1440 = 24 hours)
            snapshot_interval_min: Time between snapshots in minutes
            start_datetime: Simulation start datetime (default: current time)
            export_json: Whether to export SpectrumRecord objects to JSON

        Returns:
            Tuple of (list of PSDSnapshots, SimulationMetadata)
        """
        if start_datetime is None:
            start_datetime = datetime.now()
        print(f"\nRunning {duration_min/60:.1f}-hour simulation...")
        print(f"  Snapshot interval: {snapshot_interval_min:.1f} min")

        # Generate time points (aligned to snapshot interval)
        time_points = np.arange(0, duration_min + snapshot_interval_min/2, snapshot_interval_min)
        num_snapshots = len(time_points)

        print(f"  Total snapshots: {num_snapshots}")

        # Generate snapshots
        snapshots = []
        for i, time_min in enumerate(time_points):
            if i % 12 == 0:  # Progress update every hour
                print(f"  Processing t={time_min/60:.1f} hrs "
                      f"({i+1}/{num_snapshots} snapshots, {100*(i+1)/num_snapshots:.0f}%)")

            snapshot = self.generate_snapshot(time_min)
            snapshots.append(snapshot)

        print(f"  Completed {len(snapshots)} snapshots")

        # Calculate total bandwidth and center frequency
        total_bw_hz = len(self.transponders) * self.transponders[0].bandwidth_hz
        center_freq_hz = (self.transponders[0].center_frequency_hz +
                         self.transponders[-1].center_frequency_hz) / 2

        # Create metadata
        metadata = SimulationMetadata(
            duration_min=duration_min,
            num_snapshots=num_snapshots,
            snapshot_interval_min=snapshot_interval_min,
            rbw_hz=self.rbw_hz,
            vbw_hz=self.vbw_hz,
            sample_rate_hz=self.sample_rate_hz or 0,  # Not used in current implementation
            fft_size=self.fft_size or 0,  # Not used in current implementation
            num_transponders=len(self.transponders),
            transponder_bandwidth_hz=self.transponders[0].bandwidth_hz,
            total_bandwidth_hz=total_bw_hz,
            center_frequency_hz=center_freq_hz,
            total_carriers=len(self.carrier_config.carriers),
            static_carriers=len([c for c in self.carrier_config.carriers if c.is_static]),
            dynamic_carriers=len([c for c in self.carrier_config.carriers if not c.is_static]),
            total_interferers=len(self.interferer_configs),
            seed_carriers=self.carrier_config.seed,
            seed_interferers=42,  # From interferer generator
        )

        # Export to JSON if requested
        if export_json:
            self._export_spectrum_records(snapshots, start_datetime, snapshot_interval_min)

        return snapshots, metadata

    def _export_spectrum_records(self, snapshots: List[PSDSnapshot],
                                 start_datetime: datetime,
                                 snapshot_interval_min: float) -> None:
        """Export snapshots as SpectrumRecord objects to JSON.

        Args:
            snapshots: List of PSDSnapshot objects
            start_datetime: Simulation start datetime
            snapshot_interval_min: Time between snapshots in minutes
        """
        print(f"\n  Exporting SpectrumRecord objects to JSON...")
        import os
        from pathlib import Path

        spectrum_records = []

        # Build carrier time window lookup (all time windows for each carrier)
        carrier_time_windows = {}
        for carrier_cfg in self.carrier_config.carriers:
            if carrier_cfg.is_static:
                # Static carriers have no time windows (always on)
                carrier_time_windows[carrier_cfg.name] = []
            else:
                # Convert time windows from relative minutes to absolute datetimes
                windows = [
                    (start_datetime + timedelta(minutes=tw.start_min),
                     start_datetime + timedelta(minutes=tw.end_min))
                    for tw in carrier_cfg.time_windows
                ]
                carrier_time_windows[carrier_cfg.name] = windows

        # Build interferer time window lookup
        interferer_time_windows = {}
        for interferer_cfg in self.interferer_configs:
            windows = [(
                start_datetime + timedelta(minutes=interferer_cfg.start_time_min),
                start_datetime + timedelta(minutes=interferer_cfg.end_time_min)
            )]
            interferer_time_windows[interferer_cfg.name] = windows

        for i, snapshot in enumerate(snapshots):
            if i % 12 == 0:  # Progress every hour
                print(f"    Processing snapshot {i+1}/{len(snapshots)}...")

            snapshot_time = start_datetime + timedelta(minutes=snapshot.time_min)

            # Populate transponders with active carriers at this time
            self._populate_transponder_at_time(snapshot.time_min)

            # Build beam with transponders containing only active carriers
            beam = Beam(
                band=Band.KA,  # Ku band (using KA as closest match)
                polarization=Polarization.LHCP,
                direction=BeamDirection.DOWNLINK,
                name="Simulated Beam"
            )

            # Add transponders with their active carriers
            for xpdr_idx, xpdr in enumerate(self.transponders):
                if len(xpdr.carriers) > 0:  # Only include transponders with active carriers
                    # Create a new transponder with only non-interferer carriers
                    transponder_copy = Transponder(
                        center_frequency_hz=xpdr.center_frequency_hz,
                        bandwidth_hz=xpdr.bandwidth_hz,
                        noise_power_density_watts_per_hz=xpdr.noise_power_density_watts_per_hz,
                        noise_rolloff=xpdr.noise_rolloff,
                        name=f"Transponder_{xpdr_idx}",
                        allow_overlap=True
                    )

                    # Add only non-interferer carriers
                    for carrier in xpdr.carriers:
                        if not carrier.name.startswith('CW_'):
                            transponder_copy.add_carrier(carrier)

                    # Only add to beam if it has carriers
                    if len(transponder_copy.carriers) > 0:
                        beam.add_transponder(transponder_copy)

            # Build interferer records for active interferers
            interferer_records = []
            for interferer_cfg in self.interferer_configs:
                freq_offset = interferer_cfg.get_frequency_offset(snapshot.time_min)
                if freq_offset is not None:  # Interferer is active
                    # Calculate absolute frequency
                    xpdr = self.transponders[interferer_cfg.transponder_idx]
                    current_freq_hz = xpdr.center_frequency_hz + freq_offset

                    # Create carrier object for this interferer
                    interferer_carrier = Carrier(
                        name=interferer_cfg.name,
                        frequency_offset_hz=freq_offset,
                        cn_db=interferer_cfg.cn_db,
                        modulation=ModulationType.STATIC_CW,
                        carrier_type=CarrierType.FDMA
                    )

                    # Calculate sweep percentage
                    elapsed_min = snapshot.time_min - interferer_cfg.start_time_min
                    duration_min = interferer_cfg.end_time_min - interferer_cfg.start_time_min
                    sweep_percentage = elapsed_min / duration_min if duration_min > 0 else 0.0

                    # Determine which transponders this interferer overlaps
                    overlapping_transponders = []
                    for other_xpdr_idx, other_xpdr in enumerate(self.transponders):
                        lower_edge = other_xpdr.center_frequency_hz - other_xpdr.bandwidth_hz / 2
                        upper_edge = other_xpdr.center_frequency_hz + other_xpdr.bandwidth_hz / 2
                        if lower_edge <= current_freq_hz <= upper_edge:
                            overlapping_transponders.append(f"Transponder_{other_xpdr_idx}")

                    # Create interferer record
                    interferer_record = InterfererRecord(
                        carrier=interferer_carrier,
                        start_time=start_datetime + timedelta(minutes=interferer_cfg.start_time_min),
                        end_time=start_datetime + timedelta(minutes=interferer_cfg.end_time_min),
                        is_sweeping=(interferer_cfg.sweep_type != 'none'),
                        sweep_rate_hz_per_s=interferer_cfg.sweep_rate_hz_per_min / 60.0 if interferer_cfg.sweep_rate_hz_per_min else None,
                        sweep_type=interferer_cfg.sweep_type if interferer_cfg.sweep_type != 'none' else None,
                        sweep_start_freq_hz=xpdr.center_frequency_hz + interferer_cfg.target_frequency_offset_hz if interferer_cfg.target_frequency_offset_hz else None,
                        sweep_end_freq_hz=(xpdr.center_frequency_hz + interferer_cfg.target_frequency_offset_hz +
                                          interferer_cfg.sweep_range_hz) if interferer_cfg.sweep_range_hz else None,
                        current_frequency_hz=current_freq_hz,
                        sweep_percentage=sweep_percentage,
                        overlapping_transponders=overlapping_transponders,
                        time_windows=interferer_time_windows[interferer_cfg.name]
                    )
                    interferer_records.append(interferer_record)

            # Compress PSD data
            psd_compressed = SpectrumRecord.compress_psd(snapshot.psd_dbm_hz)

            # Calculate total bandwidth and center frequency
            # Use actual frequency range from snapshot, not calculated bandwidth
            freq_min = snapshot.frequency_hz[0]
            freq_max = snapshot.frequency_hz[-1]
            actual_bw_hz = freq_max - freq_min
            center_freq_hz = (freq_min + freq_max) / 2

            # Create SpectrumRecord
            record = SpectrumRecord(
                timestamp=snapshot_time,
                cf_hz=center_freq_hz,
                bw_hz=actual_bw_hz,
                rbw_hz=self.rbw_hz,
                vbw_hz=self.vbw_hz,
                psd_compressed=psd_compressed,
                psd_shape=snapshot.psd_dbm_hz.shape,  # Store actual shape
                beams=[beam] if len(beam.transponders) > 0 else [],
                interferers=interferer_records
            )
            spectrum_records.append(record)

        # Save to JSON file
        script_dir = Path(__file__).parent
        output_dir = script_dir / "output"
        output_dir.mkdir(exist_ok=True)

        json_filename = f"spectrum_records_{start_datetime.strftime('%Y%m%d-%H%M%S')}.json"
        json_filepath = output_dir / json_filename

        print(f"    Writing JSON to {json_filepath}...")
        SpectrumRecord.to_file(spectrum_records, str(json_filepath))

        # Calculate file size
        file_size_mb = os.path.getsize(json_filepath) / 1e6
        print(f"    JSON export complete: {len(spectrum_records)} records, {file_size_mb:.1f} MB")

    @staticmethod
    def save_results(snapshots: List[PSDSnapshot], metadata: SimulationMetadata,
                    output_dir: str):
        """Save simulation results to files.

        Args:
            snapshots: List of PSDSnapshot objects
            metadata: SimulationMetadata object
            output_dir: Directory to save results
        """
        import os

        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nSaving results to {output_dir}...")

        # Save metadata as JSON
        metadata_file = os.path.join(output_dir, "simulation_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        print(f"  Saved metadata to {metadata_file}")

        # Save snapshots as compressed numpy arrays
        # Store time, frequency (first snapshot only), and all PSDs
        snapshot_file = os.path.join(output_dir, "psd_snapshots.npz")

        time_array = np.array([s.time_min for s in snapshots])
        frequency_array = snapshots[0].frequency_hz  # Same for all snapshots
        psd_array = np.array([s.psd_dbm_hz for s in snapshots])

        np.savez_compressed(
            snapshot_file,
            time_min=time_array,
            frequency_hz=frequency_array,
            psd_dbm_hz=psd_array
        )
        print(f"  Saved PSD data to {snapshot_file}")
        print(f"    Shape: {psd_array.shape} (time × frequency)")
        print(f"    Size: {psd_array.nbytes / 1e6:.1f} MB")

        # Save activity summary (which carriers/interferers active at each time)
        activity_file = os.path.join(output_dir, "activity_log.json")
        activity_log = []
        for snapshot in snapshots:
            activity_log.append({
                'time_min': snapshot.time_min,
                'num_carriers': snapshot.num_carriers,
                'num_interferers': snapshot.num_interferers,
                'carriers': snapshot.active_carriers,
                'interferers': snapshot.active_interferers
            })

        with open(activity_file, 'w') as f:
            json.dump(activity_log, f, indent=2)
        print(f"  Saved activity log to {activity_file}")

        print(f"Save complete!")

    @staticmethod
    def load_results(output_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, SimulationMetadata]:
        """Load simulation results from files.

        Args:
            output_dir: Directory containing saved results

        Returns:
            Tuple of (time_array, frequency_array, psd_array, metadata)
        """
        import os

        print(f"Loading results from {output_dir}...")

        # Load metadata
        metadata_file = os.path.join(output_dir, "simulation_metadata.json")
        with open(metadata_file, 'r') as f:
            metadata_dict = json.load(f)
        metadata = SimulationMetadata(**metadata_dict)

        # Load PSD snapshots
        snapshot_file = os.path.join(output_dir, "psd_snapshots.npz")
        data = np.load(snapshot_file)
        time_array = data['time_min']
        frequency_array = data['frequency_hz']
        psd_array = data['psd_dbm_hz']

        print(f"  Loaded {len(time_array)} snapshots")
        print(f"  Frequency range: {frequency_array[0]/1e9:.3f} - {frequency_array[-1]/1e9:.3f} GHz")
        print(f"  PSD array shape: {psd_array.shape}")

        return time_array, frequency_array, psd_array, metadata


if __name__ == "__main__":
    # Test PSD simulation
    print("Testing PSD simulation...\n")

    from carrier_generator import CarrierGenerator
    from interferer_generator import InterfererGenerator

    # Generate carriers
    print("Generating carriers...")
    carrier_gen = CarrierGenerator(seed=42)
    carrier_config = carrier_gen.generate_carriers(num_static_per_xpdr=(5, 10), num_dynamic=15)

    # Generate interferers
    print("\nGenerating interferers...")
    interferer_gen = InterfererGenerator(seed=43)
    interferers = interferer_gen.generate_interferers(
        carrier_configs=carrier_config.carriers,
        num_long_duration=2,
        num_short_duration=5  # Reduced for testing
    )

    # Create simulator
    print("\nCreating simulator...")
    simulator = PSDSimulator(
        carrier_config=carrier_config,
        interferer_configs=interferers,
        rbw_hz=10e3,
        vbw_hz=1e3
    )

    # Run short simulation for testing (just 1 hour with 5-min intervals)
    print("\nRunning test simulation (1 hour)...")
    snapshots, metadata = simulator.run_simulation(
        duration_min=60,  # 1 hour
        snapshot_interval_min=5
    )

    # Display results
    print(f"\nSimulation complete!")
    print(f"  Duration: {metadata.duration_min} minutes")
    print(f"  Snapshots: {metadata.num_snapshots}")
    print(f"  Total bandwidth: {metadata.total_bandwidth_hz/1e6:.1f} MHz")
    print(f"  Carriers: {metadata.total_carriers} ({metadata.static_carriers} static, {metadata.dynamic_carriers} dynamic)")
    print(f"  Interferers: {metadata.total_interferers}")

    # Show activity at a few time points
    print(f"\nActivity summary:")
    for i in [0, 3, 6, 9, 12]:
        if i < len(snapshots):
            s = snapshots[i]
            print(f"  t={s.time_min:5.0f} min: {s.num_carriers} carriers, {s.num_interferers} interferers")

    # Show PSD statistics
    print(f"\nPSD statistics (first snapshot):")
    s = snapshots[0]
    print(f"  Frequency points: {len(s.frequency_hz)}")
    print(f"  PSD range: {np.min(s.psd_dbm_hz):.1f} to {np.max(s.psd_dbm_hz):.1f} dBm/Hz")
    print(f"  Frequency span: {(s.frequency_hz[-1] - s.frequency_hz[0])/1e6:.1f} MHz")

    # Test save/load
    print(f"\nTesting save/load...")
    test_dir = "test_output"
    simulator.save_results(snapshots, metadata, test_dir)

    print(f"\nLoading results back...")
    time_arr, freq_arr, psd_arr, loaded_meta = simulator.load_results(test_dir)
    print(f"  Verification: loaded {len(time_arr)} snapshots, {len(freq_arr)} frequency points")
    print(f"  Match: {np.allclose(psd_arr[0], snapshots[0].psd_dbm_hz)}")
