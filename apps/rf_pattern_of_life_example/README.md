# RF Pattern of Life Simulation

A comprehensive 24-hour satellite spectrum simulator that generates realistic pattern-of-life data with dynamic carriers, interferers, and time-varying activity.

## Overview

This application simulates a satellite beam with 6 contiguous Ku-band transponders (36 MHz each) over a 24-hour period, generating Power Spectral Density (PSD) snapshots every 5 minutes. The simulation includes:

- **Static carriers**: 5-15 carriers per transponder that are always present
- **Dynamic carriers**: ~20 carriers that appear and disappear throughout the day
- **CW interferers**: 1-5 active interferers at any time, some sweeping in frequency
- **Realistic traffic patterns**: Uneven carrier distribution, varied symbol rates and modulations

## Features

### Carrier Model

**Static Carriers** (Always Present)
- 5-15 carriers per transponder
- Mix of QPSK, 16-QAM, and 16-APSK modulation
- Symbol rates: 1-15 Msps (biased toward lower rates)
- RRC rolloff: 0.20, 0.25, or 0.35
- C/N ratio: 10-25 dB
- Automatically placed to avoid overlap

**Dynamic Carriers** (Time-Varying)
- ~20 carriers with multiple activity windows
- 1-3 time windows per carrier
- Window durations: 30 minutes to 6 hours
- Similar modulation/symbol rate distribution as static carriers

### Interferer Model

**Long-Duration Interferers** (Hours)
- 2-3 interferers active for 3-23 hours
- Start 1 hour into simulation
- C/N ratio: 15-30 dB (strong)
- 40% probability of frequency sweep (10-200 MHz/hr)
- 80% probability of targeting existing carrier

**Short-Duration Interferers** (Minutes)
- 5-10 interferers active for 10 minutes to 3 hours
- Start anytime after 1 hour
- C/N ratio: 10-25 dB (moderate to strong)
- 30% probability of frequency sweep (50-500 MHz/hr)
- 90% probability of targeting existing carrier

**Sweep Types**
- **None**: Static frequency
- **Linear**: Continuous sweep in one direction
- **Sawtooth**: Sweep up, reset, repeat

### Transponder Configuration

- 6 contiguous transponders, 36 MHz each (216 MHz total)
- Ku-band downlink: 12.2 - 12.416 GHz
- Noise power density: -120 dBm/Hz
- Uneven carrier distribution (some transponders busier than others)

## Installation

### Prerequisites

```bash
# Install satellite_downlink_simulator package (from parent directory)
cd ../..
pip install -e .

# Install visualization dependencies
pip install matplotlib numpy
```

### Directory Structure

```
rf_pattern_of_life_example/
├── main.py                        # CLI orchestration script
├── carrier_generator.py           # Carrier configuration generator
├── interferer_generator.py        # CW interferer generator
├── psd_simulator.py               # 24-hour PSD simulation engine
├── visualization.py               # Plotting and visualization tools
├── spectrum_records_utility.py    # Regenerate plots from JSON (NEW)
├── config.json                    # Generated carrier configuration
├── output/                        # Simulation results
│   ├── simulation_metadata.json
│   ├── psd_snapshots.npz
│   ├── activity_log.json
│   ├── spectrum_records_*.json    # JSON export (optional)
│   └── plots/                     # Output plots
│       ├── waterfall_plot.png
│       ├── average_spectrum.png
│       ├── snapshot_comparison.png
│       ├── activity_timeline.png
│       └── animated_spectrogram.gif (or .mp4)
└── README.md                      # This file
```

## Usage

### Quick Start

Run the full 24-hour simulation with default settings:

```bash
python main.py
```

This will:
1. Generate or load carrier configuration from `config.json`
2. Generate CW interferers
3. Run 24-hour PSD simulation (288 snapshots at 5-minute intervals)
4. Save results to `output/`
5. Generate all visualization plots in `output/plots/`

### Command-Line Options

#### Configuration

```bash
# Regenerate carrier configuration (even if config.json exists)
python main.py --regenerate
```

#### Carrier Generation

```bash
# Custom carrier counts
python main.py --static-min 5 --static-max 15 --dynamic 20

# Custom random seed for reproducibility
python main.py --seed-carriers 123
```

#### Interferer Generation

```bash
# Custom interferer counts
python main.py --interferers-long 3 --interferers-short 10

# Custom random seed
python main.py --seed-interferers 456
```

#### Simulation Parameters

```bash
# Custom simulation duration and interval
python main.py --duration 1440 --interval 5

# Custom RBW and VBW
python main.py --rbw 10000 --vbw 1000
```

#### Output Options

```bash
# Custom output directory
python main.py --output-dir my_results --plot-dir my_plots

# Export spectrum records to JSON
python main.py --export-json

# Specify custom start datetime (ISO format)
python main.py --start-datetime 2025-01-15T00:00:00

# Export with specific datetime
python main.py --export-json --start-datetime 2025-01-15T12:00:00
```

### Complete Example

```bash
# Full custom simulation
python main.py \
  --regenerate \
  --static-min 5 --static-max 12 \
  --dynamic 25 \
  --interferers-long 3 --interferers-short 12 \
  --duration 1440 --interval 5 \
  --rbw 10000 --vbw 1000 \
  --seed-carriers 42 --seed-interferers 43 \
  --output-dir output --plot-dir plots
```

## Output Files

### Configuration Files

**`config.json`**
- Complete carrier configuration
- Includes transponder parameters, all carriers, time windows
- Can be reused for reproducible simulations
- Format: JSON with nested carrier and time window objects

### Simulation Results

**`simulation_metadata.json`**
- Simulation parameters (duration, RBW, VBW, etc.)
- Transponder configuration
- Carrier statistics (counts, types)
- Random seeds for reproducibility

**`psd_snapshots.npz`**
- Compressed NumPy archive with time-series PSD data
- Arrays: `time_min`, `frequency_hz`, `psd_dbm_hz`
- Shape: (288 time steps × ~21,600 frequency points)
- Size: ~20-50 MB compressed

**`activity_log.json`**
- Per-snapshot carrier and interferer activity
- Lists which carriers/interferers were active at each time
- Useful for post-processing and analysis

### Visualization Plots

**`waterfall_plot.png`**
- Full 24-hour spectrogram
- X-axis: Frequency (GHz)
- Y-axis: Time (hours)
- Color: PSD level (dBm/Hz)
- Resolution: 3000 DPI

**`average_spectrum.png`**
- Time-averaged spectrum over full 24 hours
- Shows average PSD with 10th-90th percentile shading
- Identifies persistent carriers vs. transient activity

**`snapshot_comparison.png`**
- 6 PSD snapshots at automatically selected "interesting" times
- Times chosen based on significant spectral changes
- Useful for comparing different activity periods

**`activity_timeline.png`**
- Time-series plot showing carrier and interferer counts
- Visualizes pattern-of-life dynamics
- Two subplots: carrier activity and interferer activity

### JSON Export (Optional)

**`spectrum_records_YYYYMMDD-HHMMSS.json`**
- Complete spectrum state snapshots in JSON format
- One record per PSD snapshot with full system state
- Includes:
  - Compressed PSD data (blosc2 compression, ~10-20× reduction)
  - Active carriers at each timestamp (with time windows)
  - Active interferers with sweep parameters
  - Current frequency and sweep percentage for each interferer
  - Overlapping transponder tracking for interferers
  - All simulation parameters (RBW, VBW, frequency range)
- Size: ~5-15 MB for 24-hour simulation (depends on activity)
- Enables offline analysis and visualization replay

**Generation:**
```bash
# Export JSON with default datetime (current time)
python main.py --export-json

# Export with specific start datetime
python main.py --export-json --start-datetime 2025-01-15T00:00:00
```

**Use Cases:**
- Post-processing and offline analysis
- Machine learning training data
- Data sharing in portable format
- Replay analysis without re-running simulation
- Integration testing for downstream systems

## Regenerating Plots from JSON

The `spectrum_records_utility.py` script allows you to regenerate all visualizations from an exported JSON file without re-running the full simulation. This is much faster and enables experimentation with different visualization parameters.

### Usage

```bash
# Basic usage - regenerate all plots
python spectrum_records_utility.py output/spectrum_records_20250115-000000.json

# Specify custom output directory
python spectrum_records_utility.py spectrum_records_20250115-000000.json --output-dir custom_plots

# Skip animation (much faster - only generate static plots)
python spectrum_records_utility.py spectrum_records_20250115-000000.json --no-animation

# Generate MP4 instead of GIF
python spectrum_records_utility.py spectrum_records_20250115-000000.json --format mp4

# Use frame decimation for faster animation (every 5th frame)
python spectrum_records_utility.py spectrum_records_20250115-000000.json --frame-decimation 5
```

### Command-Line Arguments

- **json_file** (required): Path to spectrum_records JSON file
- **--output-dir**: Output directory for plots (default: `plots/` in same directory as JSON file)
- **--no-animation**: Skip animated spectrogram generation (much faster)
- **--format {gif,mp4}**: Animation format (default: gif)
- **--fps**: Animation frames per second (default: 15)
- **--frame-decimation**: Use every Nth frame to reduce file size (default: 1 = no decimation)

### What It Does

1. Loads and parses the spectrum_records JSON file
2. Decompresses all PSD arrays using blosc2
3. Reconstructs time, frequency, and PSD arrays
4. Builds activity log (carrier/interferer counts at each timestamp)
5. Recreates metadata from JSON records
6. Generates all visualization plots:
   - Waterfall plot (spectrogram)
   - Average spectrum with percentiles
   - Snapshot comparison at interesting times
   - Activity timeline
   - Animated spectrogram (optional)

### Performance

- **Loading**: ~2-5 seconds for 289 records (~10 MB file)
- **Static plots only (--no-animation)**: ~10-15 seconds
- **With animation (no decimation)**: ~30-60 seconds for 289 frames
- **With 10× decimation**: ~5-10 seconds for 29 frames
- **Memory usage**: ~500 MB for 24-hour simulation

### Example Workflow

```bash
# Step 1: Run simulation with JSON export
python main.py --export-json --start-datetime 2025-01-15T00:00:00

# Step 2: Quick preview without animation
python spectrum_records_utility.py output/spectrum_records_20250115-000000.json --no-animation

# Step 3: Generate MP4 with decimation for smaller file
python spectrum_records_utility.py output/spectrum_records_20250115-000000.json \
    --format mp4 --frame-decimation 5 --output-dir plots_mp4

# Step 4: Generate high-quality GIF with all frames
python spectrum_records_utility.py output/spectrum_records_20250115-000000.json \
    --format gif --output-dir plots_gif
```

### Benefits

- **Speed**: Regenerate plots in seconds vs. minutes for full simulation
- **Experimentation**: Try different animation formats, decimation factors, output directories
- **Portability**: Share JSON files with colleagues who can generate their own plots
- **Storage**: Archive JSON files (~10 MB) instead of full NPZ data (~50 MB)

## Module Documentation

### carrier_generator.py

**Classes:**
- `TimeWindow`: Activity window for dynamic carriers
- `CarrierConfig`: Complete carrier configuration
- `SimulationConfig`: Full simulation configuration
- `CarrierGenerator`: Generates realistic carrier configurations

**Key Methods:**
- `generate_carriers()`: Create static and dynamic carriers
- `save_config()` / `load_config()`: JSON serialization

**Algorithm:**
- Places carriers without overlap using intelligent frequency allocation
- 20% edge margin to avoid transponder boundaries
- Uneven distribution across transponders for realism

### interferer_generator.py

**Classes:**
- `SweepType`: Enum for sweep types (none, linear, sawtooth, random_walk)
- `InterfererConfig`: CW interferer configuration
- `InterfererGenerator`: Generates realistic interferers

**Key Methods:**
- `generate_interferers()`: Create long and short duration interferers
- `get_frequency_offset()`: Calculate frequency at given time (handles sweeping)
- `count_active_interferers()`: Count active interferers at time point

**Targeting:**
- 80-90% of interferers target existing carriers
- Offset placed within carrier bandwidth for maximum disruption

### psd_simulator.py

**Classes:**
- `PSDSnapshot`: Single PSD snapshot at specific time
- `SimulationMetadata`: Complete simulation metadata
- `PSDSimulator`: 24-hour PSD simulation engine

**Key Methods:**
- `generate_snapshot()`: Generate PSD at specific time
- `run_simulation()`: Complete 24-hour simulation
- `save_results()` / `load_results()`: Data persistence

**Process:**
1. Populate transponders with active carriers/interferers at each time
2. Generate PSD for each transponder using `generate_psd()`
3. Concatenate transponder PSDs into full beam spectrum
4. Track activity for metadata

### visualization.py

**Classes:**
- `Visualizer`: Creates all visualization plots

**Key Methods:**
- `create_waterfall_plot()`: Spectrogram with time/frequency/power
- `create_average_spectrum()`: Time-averaged spectrum
- `create_snapshot_comparison()`: Multiple time snapshots
- `create_activity_timeline()`: Carrier/interferer counts over time
- `create_all_plots()`: Generate all standard plots

**Auto-Selection:**
- Automatically selects "interesting" times based on spectral change magnitude
- Always includes first and last snapshots

## Technical Details

### Simulation Parameters

**Default Settings:**
- Duration: 1440 minutes (24 hours)
- Snapshot interval: 5 minutes (288 snapshots)
- RBW: 10 kHz (resolution bandwidth)
- VBW: 1 kHz (video bandwidth, affects noise variance)
- Sample rate: Auto (based on transponder bandwidth)

**Performance:**
- Full 24-hour simulation: ~5-15 minutes (depending on hardware)
- Memory usage: ~500 MB - 1 GB
- Output data: ~20-50 MB compressed

### Carrier Placement Algorithm

1. Calculate usable frequency range (80% of transponder bandwidth)
2. Sort occupied ranges by frequency
3. Find gaps between existing carriers
4. Place new carrier randomly within gap (if fits)
5. Validate no overlap (unless `allow_overlap=True`)

### Interferer Frequency Calculation

**Static** (`sweep_type = "none"`):
```
f(t) = f_base
```

**Linear** (`sweep_type = "linear"`):
```
f(t) = f_base + sweep_rate × (t - t_start)
```

**Sawtooth** (`sweep_type = "sawtooth"`):
```
period = sweep_range / sweep_rate
phase = ((t - t_start) mod period) / period
f(t) = f_base + phase × sweep_range
```

### PSD Generation

Uses `satellite_downlink_simulator.generate_psd()`:
- Frequency-domain calculation (no IQ generation)
- RRC-shaped carrier spectra
- Noise floor with realistic roll-off
- STATIC_CW carriers rendered as narrow Gaussian peaks (100 Hz FWHM)
- Measurement noise scaled by VBW/RBW ratio

## Customization

### Modifying Carrier Distribution

Edit `carrier_generator.py`, method `_select_transponder()`:

```python
# Make transponders 0, 2, 4 busier
weights = np.array([1.5, 0.8, 1.5, 0.6, 1.5, 0.7])
```

### Changing Modulation Mix

Edit `carrier_generator.py`, method `_generate_carrier_params()`:

```python
# More 16-APSK, less QPSK
mod_choices = ["QPSK", "QAM16", "QAM16", "APSK16", "APSK16"]
modulation = self.rng.choice(mod_choices)
```

### Custom Sweep Patterns

Add new sweep type to `interferer_generator.py`, class `SweepType`:

```python
class SweepType(Enum):
    CUSTOM = "custom"
```

Then implement in `InterfererConfig.get_frequency_offset()`.

### Adjusting Visualization

Edit `visualization.py` to customize:
- Colormap: Change `cmap` parameter in `create_waterfall_plot()`
- Time selection: Modify `_select_interesting_times()` algorithm
- Plot layout: Adjust figure sizes and subplot arrangements

## Troubleshooting

### Common Issues

**"Could not fit carrier X/Y in transponder Z"**
- Too many carriers for available bandwidth
- Solution: Reduce `--static-max` or `--dynamic` count
- Or increase edge margin tolerance in `carrier_generator.py`

**Memory error during simulation**
- Simulation too long or RBW too fine
- Solution: Reduce `--duration`, increase `--rbw`, or increase `--interval`

**Plots look noisy**
- VBW too high relative to RBW
- Solution: Decrease `--vbw` (e.g., `--vbw 100`)

**Simulation too slow**
- Fine RBW or many carriers
- Solution: Increase `--rbw` (e.g., `--rbw 50000` for 50 kHz)
- Or reduce carrier counts

### Performance Tips

1. **Faster simulation**: Increase RBW to 50-100 kHz
2. **Smoother plots**: Decrease VBW/RBW ratio (e.g., VBW = RBW/10)
3. **Less memory**: Increase snapshot interval to 10-15 minutes
4. **More detail**: Decrease RBW to 1-5 kHz (slower but higher resolution)

## Example Workflow

### Generate and Analyze

```bash
# Step 1: Generate configuration
python main.py --regenerate --seed-carriers 42

# Step 2: Run simulation with custom parameters
python main.py --rbw 20000 --vbw 2000

# Step 3: Load and analyze results (Python)
from psd_simulator import PSDSimulator
import numpy as np

time, freq, psd, meta = PSDSimulator.load_results('output')

# Calculate average power over time
power_vs_time = np.array([np.trapz(10**(p/10), freq) for p in psd])

# Find peak activity time
peak_idx = np.argmax(power_vs_time)
print(f"Peak activity at t={time[peak_idx]/60:.1f} hours")
```

### Compare Multiple Runs

```bash
# Run 1: Default carriers
python main.py --output-dir run1

# Run 2: More dynamic carriers
python main.py --regenerate --dynamic 40 --output-dir run2

# Run 3: More interferers
python main.py --interferers-long 5 --interferers-short 15 --output-dir run3
```

## References

### Related Documentation

- `../../CLAUDE.md`: Codebase design decisions and context
- `../../README.md`: Main satellite_downlink_simulator package docs
- `../../TODO.md`: Original requirements for this application

### Key Concepts

- **C/N (Carrier-to-Noise Ratio)**: Power ratio in dB, used instead of absolute power
- **RBW (Resolution Bandwidth)**: Determines frequency resolution
- **VBW (Video Bandwidth)**: Controls noise variance via integration time
- **RRC (Root-Raised-Cosine)**: Pulse-shaping filter for digital modulation
- **TDMA vs FDMA**: Time-division vs frequency-division multiple access

## Contributing

To extend this simulation:

1. **New carrier types**: Add to `ModulationType` in main package
2. **New sweep patterns**: Extend `SweepType` enum and implement calculation
3. **Custom metrics**: Add analysis functions to load and process saved data
4. **New visualizations**: Subclass `Visualizer` or add methods

## License

Part of the `satellite_downlink_simulator` package.

## Authors

Generated as part of the RF Pattern of Life example application.

---

**Questions or Issues?** Refer to the main package documentation or raise an issue.
