"""Carrier generation for pattern-of-life simulation.

Generates realistic static and dynamic carriers with time-varying activity.
"""

import json
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict


@dataclass
class TimeWindow:
    """Time window for carrier activity (minutes since start)."""
    start_min: float
    end_min: float

    def is_active(self, time_min: float) -> bool:
        """Check if carrier is active at given time."""
        return self.start_min <= time_min <= self.end_min


@dataclass
class CarrierConfig:
    """Configuration for a single carrier."""
    name: str
    transponder_idx: int  # Which transponder (0-5)
    frequency_offset_hz: float  # Offset from transponder center
    cn_db: float  # C/N ratio in dB
    symbol_rate_sps: float  # Symbol rate in symbols/sec
    modulation: str  # "QPSK", "QAM16", etc.
    rrc_rolloff: float  # RRC rolloff factor
    carrier_type: str  # "FDMA" or "TDMA"
    is_static: bool  # True if always on, False if dynamic
    time_windows: List[TimeWindow]  # Activity windows (empty if static)

    def is_active(self, time_min: float) -> bool:
        """Check if carrier is active at given time."""
        if self.is_static:
            return True
        return any(window.is_active(time_min) for window in self.time_windows)


@dataclass
class SimulationConfig:
    """Complete simulation configuration."""
    transponders: List[Dict]  # Transponder parameters
    carriers: List[CarrierConfig]  # All carriers
    seed: int  # Random seed for reproducibility


class CarrierGenerator:
    """Generates realistic carriers for pattern-of-life simulation."""

    def __init__(self, seed: int = 42):
        """Initialize generator with random seed."""
        self.rng = np.random.RandomState(seed)
        self.seed = seed

        # Transponder configuration: 6 x 36 MHz, contiguous, Ku-band
        self.num_transponders = 6
        self.transponder_bw_hz = 36e6
        self.ku_band_start_hz = 12.2e9  # Start of Ku downlink band

        # Generate transponder center frequencies
        self.transponders = []
        for i in range(self.num_transponders):
            center_freq = self.ku_band_start_hz + i * self.transponder_bw_hz + self.transponder_bw_hz / 2
            self.transponders.append({
                'index': i,
                'center_frequency_hz': center_freq,
                'bandwidth_hz': self.transponder_bw_hz,
                'noise_power_density_watts_per_hz': 1e-15,  # -120 dBm/Hz
                'noise_rolloff': 0.25
            })

    def _generate_carrier_params(self) -> Tuple[float, str, float, float]:
        """Generate random carrier parameters.

        Returns:
            Tuple of (symbol_rate_sps, modulation, rrc_rolloff, cn_db)
        """
        # Symbol rate: mix of low (1-5 Msps) and medium (5-15 Msps)
        if self.rng.rand() < 0.4:  # 40% low rate
            symbol_rate_sps = self.rng.uniform(1e6, 5e6)
        else:  # 60% medium rate
            symbol_rate_sps = self.rng.uniform(5e6, 15e6)

        # Round to 100 kHz increments
        symbol_rate_sps = round(symbol_rate_sps / 1e5) * 1e5

        # Modulation: mostly QPSK and 16-QAM
        mod_choices = ["QPSK", "QPSK", "QPSK", "QAM16", "QAM16", "APSK16"]
        modulation = self.rng.choice(mod_choices)

        # RRC rolloff: typical satellite values
        rrc_rolloff = self.rng.choice([0.20, 0.25, 0.35])

        # C/N: realistic operating range
        cn_db = self.rng.uniform(10.0, 25.0)

        return symbol_rate_sps, modulation, rrc_rolloff, cn_db

    def _select_transponder(self, already_allocated: List[int]) -> int:
        """Select transponder for new carrier (with uneven distribution).

        Args:
            already_allocated: List of transponder indices already assigned carriers

        Returns:
            Transponder index (0-5)
        """
        # Create weight distribution: some transponders more loaded
        # Transponders 1, 2, 4 are busier
        weights = np.array([0.8, 1.5, 1.5, 0.6, 1.5, 0.7])
        weights = weights / weights.sum()

        return self.rng.choice(self.num_transponders, p=weights)

    def _find_free_space(self, transponder_idx: int, occupied_ranges: List[Tuple[float, float]],
                         carrier_bw: float) -> float:
        """Find free frequency space in a transponder.

        Args:
            transponder_idx: Index of transponder
            occupied_ranges: List of (lower, upper) frequency ranges already occupied
            carrier_bw: Bandwidth of new carrier

        Returns:
            Frequency offset from transponder center, or None if no space
        """
        # Usable range: 80% of transponder bandwidth (20% edge margin)
        usable_span = 0.8 * self.transponder_bw_hz
        usable_lower = -usable_span / 2
        usable_upper = usable_span / 2

        # Sort occupied ranges
        occupied_ranges = sorted(occupied_ranges)

        # Try to find gap between existing carriers
        for i in range(len(occupied_ranges) + 1):
            if i == 0:
                gap_lower = usable_lower
            else:
                gap_lower = occupied_ranges[i-1][1]

            if i == len(occupied_ranges):
                gap_upper = usable_upper
            else:
                gap_upper = occupied_ranges[i][0]

            gap_width = gap_upper - gap_lower

            # Check if carrier fits in this gap
            if gap_width >= carrier_bw:
                # Place carrier randomly within gap
                max_offset = gap_width - carrier_bw
                offset_within_gap = self.rng.uniform(0, max_offset)
                center_offset = gap_lower + carrier_bw / 2 + offset_within_gap
                return center_offset

        return None

    def _generate_time_windows(self, num_windows: int) -> List[TimeWindow]:
        """Generate random time windows for dynamic carrier.

        Args:
            num_windows: Number of activity windows

        Returns:
            List of TimeWindow objects
        """
        windows = []

        for _ in range(num_windows):
            # Random duration: 30 min to 6 hours
            duration_min = self.rng.uniform(30, 360)

            # Random start time: anytime in first 18 hours (to allow window to fit)
            max_start = max(5, 1440 - duration_min - 5)  # Leave 5 min buffer
            start_min = self.rng.uniform(5, max_start)
            end_min = start_min + duration_min

            # Round to 5-minute boundaries
            start_min = round(start_min / 5) * 5
            end_min = round(end_min / 5) * 5

            windows.append(TimeWindow(start_min=start_min, end_min=end_min))

        # Sort windows by start time
        windows = sorted(windows, key=lambda w: w.start_min)

        return windows

    def generate_carriers(self, num_static_per_xpdr: Tuple[int, int] = (5, 15),
                          num_dynamic: int = 20) -> SimulationConfig:
        """Generate complete carrier configuration.

        Args:
            num_static_per_xpdr: Tuple of (min, max) static carriers per transponder
            num_dynamic: Number of dynamic carriers

        Returns:
            SimulationConfig with all carriers
        """
        carriers = []
        carrier_id = 0

        # Track occupied frequency ranges per transponder
        occupied_by_xpdr = [[] for _ in range(self.num_transponders)]

        # Generate static carriers (5-15 per transponder)
        print(f"Generating static carriers...")
        for xpdr_idx in range(self.num_transponders):
            num_static = self.rng.randint(num_static_per_xpdr[0], num_static_per_xpdr[1] + 1)

            for i in range(num_static):
                # Generate carrier parameters
                symbol_rate, modulation, rolloff, cn_db = self._generate_carrier_params()
                carrier_bw = symbol_rate * (1 + rolloff)

                # Find free space
                freq_offset = self._find_free_space(xpdr_idx, occupied_by_xpdr[xpdr_idx], carrier_bw)

                if freq_offset is None:
                    print(f"  Warning: Could not fit carrier {i+1}/{num_static} in transponder {xpdr_idx}")
                    continue

                # Add to occupied ranges
                lower = freq_offset - carrier_bw / 2
                upper = freq_offset + carrier_bw / 2
                occupied_by_xpdr[xpdr_idx].append((lower, upper))

                # Create carrier config
                carrier = CarrierConfig(
                    name=f"Static_{xpdr_idx}_{i}",
                    transponder_idx=xpdr_idx,
                    frequency_offset_hz=freq_offset,
                    cn_db=round(cn_db, 1),
                    symbol_rate_sps=symbol_rate,
                    modulation=modulation,
                    rrc_rolloff=rolloff,
                    carrier_type="FDMA",
                    is_static=True,
                    time_windows=[]
                )
                carriers.append(carrier)
                carrier_id += 1

        print(f"  Created {len(carriers)} static carriers")

        # Generate dynamic carriers
        print(f"Generating {num_dynamic} dynamic carriers...")
        for i in range(num_dynamic):
            # Generate carrier parameters
            symbol_rate, modulation, rolloff, cn_db = self._generate_carrier_params()
            carrier_bw = symbol_rate * (1 + rolloff)

            # Select transponder
            xpdr_idx = self._select_transponder([c.transponder_idx for c in carriers])

            # Find free space (note: we allow dynamic carriers to overlap with each other
            # if they're not active at the same time, but for simplicity we check against
            # all occupied space)
            freq_offset = self._find_free_space(xpdr_idx, occupied_by_xpdr[xpdr_idx], carrier_bw)

            if freq_offset is None:
                # Try other transponders
                found = False
                for alt_xpdr_idx in range(self.num_transponders):
                    if alt_xpdr_idx == xpdr_idx:
                        continue
                    freq_offset = self._find_free_space(alt_xpdr_idx, occupied_by_xpdr[alt_xpdr_idx], carrier_bw)
                    if freq_offset is not None:
                        xpdr_idx = alt_xpdr_idx
                        found = True
                        break

                if not found:
                    print(f"  Warning: Could not fit dynamic carrier {i+1}/{num_dynamic}")
                    continue

            # Don't add to occupied ranges (allow overlapping dynamic carriers)
            # But track them for placement purposes
            lower = freq_offset - carrier_bw / 2
            upper = freq_offset + carrier_bw / 2
            occupied_by_xpdr[xpdr_idx].append((lower, upper))

            # Generate time windows (1-3 windows per carrier)
            num_windows = self.rng.randint(1, 4)
            time_windows = self._generate_time_windows(num_windows)

            # Create carrier config
            carrier = CarrierConfig(
                name=f"Dynamic_{i}",
                transponder_idx=xpdr_idx,
                frequency_offset_hz=freq_offset,
                cn_db=round(cn_db, 1),
                symbol_rate_sps=symbol_rate,
                modulation=modulation,
                rrc_rolloff=rolloff,
                carrier_type="FDMA",
                is_static=False,
                time_windows=time_windows
            )
            carriers.append(carrier)
            carrier_id += 1

        print(f"  Created {len(carriers) - len([c for c in carriers if c.is_static])} dynamic carriers")
        print(f"Total carriers: {len(carriers)}")

        # Create simulation config
        config = SimulationConfig(
            transponders=self.transponders,
            carriers=carriers,
            seed=self.seed
        )

        return config

    @staticmethod
    def save_config(config: SimulationConfig, filepath: str):
        """Save configuration to JSON file.

        Args:
            config: SimulationConfig to save
            filepath: Path to JSON file
        """
        # Convert to dict with custom serialization for nested dataclasses
        config_dict = {
            'transponders': config.transponders,
            'carriers': [
                {
                    **asdict(carrier),
                    'time_windows': [asdict(w) for w in carrier.time_windows]
                }
                for carrier in config.carriers
            ],
            'seed': config.seed
        }

        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"Saved configuration to {filepath}")

    @staticmethod
    def load_config(filepath: str) -> SimulationConfig:
        """Load configuration from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            SimulationConfig object
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        # Reconstruct dataclasses
        carriers = []
        for c in config_dict['carriers']:
            time_windows = [TimeWindow(**w) for w in c['time_windows']]
            carrier = CarrierConfig(
                name=c['name'],
                transponder_idx=c['transponder_idx'],
                frequency_offset_hz=c['frequency_offset_hz'],
                cn_db=c['cn_db'],
                symbol_rate_sps=c['symbol_rate_sps'],
                modulation=c['modulation'],
                rrc_rolloff=c['rrc_rolloff'],
                carrier_type=c['carrier_type'],
                is_static=c['is_static'],
                time_windows=time_windows
            )
            carriers.append(carrier)

        config = SimulationConfig(
            transponders=config_dict['transponders'],
            carriers=carriers,
            seed=config_dict['seed']
        )

        print(f"Loaded configuration from {filepath}")
        return config


if __name__ == "__main__":
    # Test carrier generation
    print("Testing carrier generation...\n")

    generator = CarrierGenerator(seed=42)
    config = generator.generate_carriers(num_static_per_xpdr=(5, 15), num_dynamic=20)

    print(f"\nGenerated configuration:")
    print(f"  Transponders: {len(config.transponders)}")
    print(f"  Total carriers: {len(config.carriers)}")
    print(f"  Static carriers: {len([c for c in config.carriers if c.is_static])}")
    print(f"  Dynamic carriers: {len([c for c in config.carriers if not c.is_static])}")

    # Show carrier distribution
    print(f"\nCarrier distribution by transponder:")
    for i in range(6):
        xpdr_carriers = [c for c in config.carriers if c.transponder_idx == i]
        static_count = len([c for c in xpdr_carriers if c.is_static])
        dynamic_count = len([c for c in xpdr_carriers if not c.is_static])
        print(f"  Transponder {i}: {len(xpdr_carriers)} total ({static_count} static, {dynamic_count} dynamic)")

    # Show a few example carriers
    print(f"\nExample carriers:")
    for i, carrier in enumerate(config.carriers[:3]):
        print(f"\n  {carrier.name}:")
        print(f"    Transponder: {carrier.transponder_idx}")
        print(f"    Frequency offset: {carrier.frequency_offset_hz / 1e6:.3f} MHz")
        print(f"    Symbol rate: {carrier.symbol_rate_sps / 1e6:.3f} Msps")
        print(f"    Modulation: {carrier.modulation}")
        print(f"    C/N: {carrier.cn_db} dB")
        print(f"    Static: {carrier.is_static}")
        if not carrier.is_static:
            print(f"    Time windows: {len(carrier.time_windows)}")
            for w in carrier.time_windows:
                print(f"      {w.start_min:.0f} - {w.end_min:.0f} min")

    # Test save/load
    print(f"\nTesting save/load...")
    test_file = "test_config.json"
    generator.save_config(config, test_file)
    loaded_config = generator.load_config(test_file)
    print(f"  Verification: {len(loaded_config.carriers)} carriers loaded")
