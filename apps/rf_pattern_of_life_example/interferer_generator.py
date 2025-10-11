"""CW interferer generation for pattern-of-life simulation.

Generates realistic CW interferers with time-varying activity and frequency sweeping.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum


class SweepType(Enum):
    """Type of frequency sweep."""
    NONE = "none"  # Static frequency
    LINEAR = "linear"  # Linear sweep
    SAWTOOTH = "sawtooth"  # Sawtooth sweep
    RANDOM_WALK = "random_walk"  # Random walk


@dataclass
class InterfererConfig:
    """Configuration for a single CW interferer."""
    name: str
    transponder_idx: int  # Which transponder (0-5)
    start_time_min: float  # When interferer appears (minutes since start)
    end_time_min: float  # When interferer disappears
    cn_db: float  # C/N ratio in dB (relative to transponder noise)

    # Target carrier (optional - None means independent interferer)
    target_carrier_name: Optional[str]  # Name of carrier to interfere with
    target_frequency_offset_hz: Optional[float]  # Base frequency of target

    # Frequency sweep parameters
    sweep_type: str  # "none", "linear", "sawtooth", "random_walk"
    sweep_rate_hz_per_min: float  # Sweep rate in Hz/min (0 for static)
    sweep_range_hz: float  # Total sweep range (for sawtooth)

    def get_frequency_offset(self, time_min: float) -> float:
        """Calculate frequency offset at given time.

        Args:
            time_min: Time in minutes since start

        Returns:
            Frequency offset in Hz from transponder center
        """
        if time_min < self.start_time_min or time_min > self.end_time_min:
            return None  # Interferer not active

        elapsed_min = time_min - self.start_time_min

        if self.sweep_type == SweepType.NONE.value:
            return self.target_frequency_offset_hz

        elif self.sweep_type == SweepType.LINEAR.value:
            # Linear sweep from base frequency
            offset = self.sweep_rate_hz_per_min * elapsed_min
            return self.target_frequency_offset_hz + offset

        elif self.sweep_type == SweepType.SAWTOOTH.value:
            # Sawtooth: sweep up, reset, repeat
            sweep_period_min = self.sweep_range_hz / abs(self.sweep_rate_hz_per_min)
            phase = (elapsed_min % sweep_period_min) / sweep_period_min
            offset = phase * self.sweep_range_hz
            return self.target_frequency_offset_hz + offset

        elif self.sweep_type == SweepType.RANDOM_WALK.value:
            # Random walk is handled externally (needs continuous state)
            # For now, return base frequency
            return self.target_frequency_offset_hz

        return self.target_frequency_offset_hz

    def is_active(self, time_min: float) -> bool:
        """Check if interferer is active at given time."""
        return self.start_time_min <= time_min <= self.end_time_min


class InterfererGenerator:
    """Generates realistic CW interferers for pattern-of-life simulation."""

    def __init__(self, seed: int = 42):
        """Initialize generator with random seed."""
        self.rng = np.random.RandomState(seed)
        self.seed = seed

    def generate_interferers(self, carrier_configs: List, num_long_duration: int = 2,
                            num_short_duration: int = 8) -> List[InterfererConfig]:
        """Generate CW interferers targeting carriers.

        Args:
            carrier_configs: List of CarrierConfig objects to potentially target
            num_long_duration: Number of long-duration interferers (hours)
            num_short_duration: Number of short-duration interferers (minutes)

        Returns:
            List of InterfererConfig objects
        """
        interferers = []

        # Filter to get carriers that are good targets
        # Prefer static carriers and carriers active for long periods
        static_carriers = [c for c in carrier_configs if c.is_static]

        print(f"Generating {num_long_duration} long-duration interferers...")

        # Generate long-duration interferers (3-23 hours)
        # These start 1 hour in and can run until end
        for i in range(num_long_duration):
            # Start between 60-120 min
            start_time_min = self.rng.uniform(60, 120)
            # Duration: 3-23 hours
            duration_min = self.rng.uniform(180, 1380)
            end_time_min = min(start_time_min + duration_min, 1440)

            # Round to 5-minute boundaries
            start_time_min = round(start_time_min / 5) * 5
            end_time_min = round(end_time_min / 5) * 5

            # 80% chance to target a carrier, 20% chance independent
            if self.rng.rand() < 0.8 and len(static_carriers) > 0:
                target_carrier = self.rng.choice(static_carriers)
                target_name = target_carrier.name
                target_freq = target_carrier.frequency_offset_hz
                xpdr_idx = target_carrier.transponder_idx

                # Offset within carrier bandwidth
                carrier_bw = target_carrier.symbol_rate_sps * (1 + target_carrier.rrc_rolloff)
                freq_offset_within_carrier = self.rng.uniform(-carrier_bw/2, carrier_bw/2)
                target_freq += freq_offset_within_carrier

                # Clamp to transponder bounds (slightly inside ±18 MHz to avoid boundary issues)
                target_freq = np.clip(target_freq, -17.99e6, 17.99e6)

                # C/N: Set higher than target carrier to be visible (add 5-15 dB)
                cn_boost_db = self.rng.uniform(5.0, 15.0)
                cn_db = target_carrier.cn_db + cn_boost_db
            else:
                # Independent interferer
                target_name = None
                xpdr_idx = self.rng.randint(0, 6)
                # Random frequency within transponder (90% of bandwidth for safety)
                target_freq = self.rng.uniform(-16e6, 16e6)  # Within 90% of 36 MHz

                # C/N: 15-30 dB (strong interferers)
                cn_db = self.rng.uniform(15.0, 30.0)

            # 40% chance to sweep
            if self.rng.rand() < 0.4:
                sweep_type = self.rng.choice([
                    SweepType.LINEAR.value,
                    SweepType.SAWTOOTH.value,
                ])

                # Slow sweep: 10-200 MHz/hr = 166 kHz/min to 3.33 MHz/min
                sweep_rate_hz_per_hr = self.rng.uniform(10e6, 200e6)
                sweep_rate_hz_per_min = sweep_rate_hz_per_hr / 60

                if sweep_type == SweepType.SAWTOOTH.value:
                    # Sweep range: limited to stay within transponder
                    max_sweep_range = min(17.99e6 - abs(target_freq), 20e6)
                    sweep_range_hz = self.rng.uniform(2e6, max(5e6, max_sweep_range))
                    sweep_range_hz = min(sweep_range_hz, max_sweep_range)
                else:
                    # Linear sweep: reduce rate if would exceed bounds during active period
                    duration_hrs = (end_time_min - start_time_min) / 60
                    max_sweep_distance = 17.99e6 - abs(target_freq)
                    max_safe_rate_hz_per_hr = max_sweep_distance / max(duration_hrs, 1)
                    sweep_rate_hz_per_hr = min(sweep_rate_hz_per_hr, max_safe_rate_hz_per_hr)
                    sweep_rate_hz_per_min = sweep_rate_hz_per_hr / 60
                    sweep_range_hz = 0  # Not used for linear

            else:
                sweep_type = SweepType.NONE.value
                sweep_rate_hz_per_min = 0
                sweep_range_hz = 0

            interferer = InterfererConfig(
                name=f"CW_Long_{i}",
                transponder_idx=xpdr_idx,
                start_time_min=start_time_min,
                end_time_min=end_time_min,
                cn_db=round(cn_db, 1),
                target_carrier_name=target_name,
                target_frequency_offset_hz=target_freq,
                sweep_type=sweep_type,
                sweep_rate_hz_per_min=sweep_rate_hz_per_min,
                sweep_range_hz=sweep_range_hz,
            )
            interferers.append(interferer)

        print(f"  Created {num_long_duration} long-duration interferers")

        # Generate short-duration interferers (10 min - 3 hours)
        print(f"Generating {num_short_duration} short-duration interferers...")
        for i in range(num_short_duration):
            # Start anytime after 60 min
            start_time_min = self.rng.uniform(60, 1200)
            # Duration: 10 min - 3 hours
            duration_min = self.rng.uniform(10, 180)
            end_time_min = min(start_time_min + duration_min, 1440)

            # Round to 5-minute boundaries
            start_time_min = round(start_time_min / 5) * 5
            end_time_min = round(end_time_min / 5) * 5

            # 90% chance to target a carrier (short bursts usually target something)
            if self.rng.rand() < 0.9 and len(carrier_configs) > 0:
                # Can target any carrier (static or dynamic)
                target_carrier = self.rng.choice(carrier_configs)
                target_name = target_carrier.name
                target_freq = target_carrier.frequency_offset_hz
                xpdr_idx = target_carrier.transponder_idx

                # Offset within carrier bandwidth
                carrier_bw = target_carrier.symbol_rate_sps * (1 + target_carrier.rrc_rolloff)
                freq_offset_within_carrier = self.rng.uniform(-carrier_bw/2, carrier_bw/2)
                target_freq += freq_offset_within_carrier

                # Clamp to transponder bounds (slightly inside ±18 MHz to avoid boundary issues)
                target_freq = np.clip(target_freq, -17.99e6, 17.99e6)

                # C/N: Set higher than target carrier to be visible (add 5-12 dB)
                cn_boost_db = self.rng.uniform(5.0, 12.0)
                cn_db = target_carrier.cn_db + cn_boost_db
            else:
                # Independent interferer
                target_name = None
                xpdr_idx = self.rng.randint(0, 6)
                # Random frequency within transponder (90% of bandwidth for safety)
                target_freq = self.rng.uniform(-16e6, 16e6)  # Within 90% of 36 MHz

                # C/N: 10-25 dB (moderate to strong)
                cn_db = self.rng.uniform(10.0, 25.0)

            # 30% chance to sweep (less common for short bursts)
            if self.rng.rand() < 0.3:
                sweep_type = self.rng.choice([
                    SweepType.LINEAR.value,
                    SweepType.SAWTOOTH.value,
                ])

                # Faster sweep for short duration: 50-500 MHz/hr
                sweep_rate_hz_per_hr = self.rng.uniform(50e6, 500e6)
                sweep_rate_hz_per_min = sweep_rate_hz_per_hr / 60

                if sweep_type == SweepType.SAWTOOTH.value:
                    # Sweep range: limited to stay within transponder
                    max_sweep_range = min(18e6 - abs(target_freq), 10e6)
                    sweep_range_hz = self.rng.uniform(2e6, max(3e6, max_sweep_range))
                    sweep_range_hz = min(sweep_range_hz, max_sweep_range)
                else:
                    # Linear sweep: reduce rate if would exceed bounds during active period
                    duration_hrs = (end_time_min - start_time_min) / 60
                    max_sweep_distance = 17.99e6 - abs(target_freq)
                    max_safe_rate_hz_per_hr = max_sweep_distance / max(duration_hrs, 0.1)
                    sweep_rate_hz_per_hr = min(sweep_rate_hz_per_hr, max_safe_rate_hz_per_hr)
                    sweep_rate_hz_per_min = sweep_rate_hz_per_hr / 60
                    sweep_range_hz = 0

            else:
                sweep_type = SweepType.NONE.value
                sweep_rate_hz_per_min = 0
                sweep_range_hz = 0

            interferer = InterfererConfig(
                name=f"CW_Short_{i}",
                transponder_idx=xpdr_idx,
                start_time_min=start_time_min,
                end_time_min=end_time_min,
                cn_db=round(cn_db, 1),
                target_carrier_name=target_name,
                target_frequency_offset_hz=target_freq,
                sweep_type=sweep_type,
                sweep_rate_hz_per_min=sweep_rate_hz_per_min,
                sweep_range_hz=sweep_range_hz,
            )
            interferers.append(interferer)

        print(f"  Created {num_short_duration} short-duration interferers")

        # Sort by start time
        interferers = sorted(interferers, key=lambda x: x.start_time_min)

        print(f"Total interferers: {len(interferers)}")

        return interferers

    def count_active_interferers(self, interferers: List[InterfererConfig],
                                time_min: float) -> int:
        """Count how many interferers are active at given time.

        Args:
            interferers: List of InterfererConfig objects
            time_min: Time in minutes since start

        Returns:
            Number of active interferers
        """
        return sum(1 for i in interferers if i.is_active(time_min))

    def get_active_interferers(self, interferers: List[InterfererConfig],
                              time_min: float) -> List[Tuple[InterfererConfig, float]]:
        """Get all active interferers with their current frequencies.

        Args:
            interferers: List of InterfererConfig objects
            time_min: Time in minutes since start

        Returns:
            List of (InterfererConfig, frequency_offset_hz) tuples
        """
        active = []
        for interferer in interferers:
            freq = interferer.get_frequency_offset(time_min)
            if freq is not None:
                active.append((interferer, freq))
        return active


if __name__ == "__main__":
    # Test interferer generation
    print("Testing interferer generation...\n")

    # First need to create some mock carriers
    from carrier_generator import CarrierGenerator, CarrierConfig, TimeWindow

    # Generate carriers
    carrier_gen = CarrierGenerator(seed=42)
    carrier_config = carrier_gen.generate_carriers(num_static_per_xpdr=(5, 10), num_dynamic=15)

    print(f"\n{'='*60}\n")

    # Generate interferers
    interferer_gen = InterfererGenerator(seed=43)
    interferers = interferer_gen.generate_interferers(
        carrier_configs=carrier_config.carriers,
        num_long_duration=2,
        num_short_duration=8
    )

    print(f"\nGenerated {len(interferers)} interferers")

    # Show examples
    print(f"\nExample interferers:")
    for i, interferer in enumerate(interferers[:5]):
        print(f"\n  {interferer.name}:")
        print(f"    Transponder: {interferer.transponder_idx}")
        print(f"    Active: {interferer.start_time_min:.0f} - {interferer.end_time_min:.0f} min "
              f"({(interferer.end_time_min - interferer.start_time_min)/60:.1f} hrs)")
        print(f"    C/N: {interferer.cn_db} dB")
        print(f"    Target carrier: {interferer.target_carrier_name or 'Independent'}")
        if interferer.target_frequency_offset_hz is not None:
            print(f"    Base frequency: {interferer.target_frequency_offset_hz / 1e6:.3f} MHz")
        print(f"    Sweep type: {interferer.sweep_type}")
        if interferer.sweep_type != SweepType.NONE.value:
            sweep_rate_mhz_hr = interferer.sweep_rate_hz_per_min * 60 / 1e6
            print(f"    Sweep rate: {sweep_rate_mhz_hr:.1f} MHz/hr")
            if interferer.sweep_type == SweepType.SAWTOOTH.value:
                print(f"    Sweep range: {interferer.sweep_range_hz / 1e6:.1f} MHz")

    # Test activity count at different times
    print(f"\nInterferer activity over time:")
    test_times = [0, 60, 120, 300, 600, 900, 1200, 1440]
    for t in test_times:
        count = interferer_gen.count_active_interferers(interferers, t)
        print(f"  t={t:4.0f} min ({t/60:5.1f} hrs): {count} active interferers")

    # Show frequency sweep example
    print(f"\nFrequency sweep example (first sweeping interferer):")
    sweeping = [i for i in interferers if i.sweep_type != SweepType.NONE.value]
    if sweeping:
        interferer = sweeping[0]
        print(f"  {interferer.name} ({interferer.sweep_type}):")
        # Sample frequencies every 30 minutes during active period
        t = interferer.start_time_min
        while t <= interferer.end_time_min:
            freq = interferer.get_frequency_offset(t)
            print(f"    t={t:6.0f} min: f={freq/1e6:8.3f} MHz")
            t += 30
