# CLAUDE.md - Context for Future Claude Sessions

This document provides critical context about the Satellite Spectrum Emulator codebase to help future Claude sessions understand the design decisions, implementation details, and evolution of this project.

## Project Overview

This is a Python library for modeling and simulating satellite communication systems. It provides:

1. **Object Model**: Hierarchical representation of satellite downlinks using Carrier, Transponder, and Beam objects
2. **Signal Generation**: Functions to produce realistic Power Spectral Density (PSD) and In-phase/Quadrature (IQ) data
3. **Applications**: Example use cases demonstrating the library's capabilities

### Core Library (`satellite_downlink_simulator/`)

The core library provides object-oriented abstractions for satellite communication systems:

- **Objects** (`objects/`): Carrier, Transponder, Beam classes with physical parameter validation
- **Simulation** (`simulation/`): PSD and IQ generation functions operating on these objects
- **Utilities** (`utils.py`): Signal processing functions (RRC filters, constellations, noise, validation)

**Key Design Philosophy**:
- Objects represent physical RF architecture (what exists in the satellite system)
- Simulation functions generate synthetic measurements (what you'd see on test equipment)
- Separation allows flexible composition: build any scenario, generate any measurement type

### Example Applications (`apps/`)

The library includes example applications demonstrating real-world use cases:

**RF Pattern of Life** (`apps/rf_pattern_of_life_example/`): Simulates 24-hour temporal evolution of satellite spectrum with dynamic carriers and interferers. Demonstrates using the core library to model time-varying scenarios with visualization and analysis tools.

**Primary Use Cases**:
- Create synthetic satellite spectrum data for testing without actual satellite hardware
- Model realistic multi-carrier scenarios for spectrum planning and analysis
- Generate IQ samples for demodulator and signal processing algorithm development
- Produce training data for machine learning applications in satellite communications
- Visualize spectrum occupancy and interference scenarios

## Key Design Decisions

### 1. Carrier Power Specification Using C/N Instead of Absolute Power

**Location**: `satellite_spectrum_emulator/carrier.py`

**Important**: Carriers are specified using **Carrier-to-Noise ratio (C/N in dB)**, NOT absolute power in Watts.

**Why**: In satellite communications, operators think in terms of C/N ratios rather than absolute power levels. This is more intuitive and matches industry practice.

**Implementation**:
```python
# carrier.py lines 88-98
def calculate_power_watts(self, noise_power_density_watts_per_hz: float) -> float:
    """Calculate carrier power from C/N and transponder noise density."""
    noise_power_watts = noise_power_density_watts_per_hz * self.bandwidth_hz
    cn_linear = 10 ** (self.cn_db / 10)
    carrier_power_watts = cn_linear * noise_power_watts
    return carrier_power_watts
```

**Historical Note**: This was changed from an earlier design that used absolute `power_watts`. The parameter was renamed from `carrier_to_noise_db` to `cn_db` for brevity.

### 2. VBW Controls Noise Variance, Not Smoothing

**Location**: `satellite_spectrum_emulator/generation.py` lines 137-147

**Critical Understanding**: Video Bandwidth (VBW) does NOT apply smoothing to the PSD trace. Instead, it controls the capture/integration time, which affects noise variance.

**Physical Relationship**:
```
Capture Time ≈ RBW / VBW
Noise Variance ∝ 1 / Capture_Time
Noise Std Dev = Base_Noise × √(VBW / RBW)
```

**Implementation**:
```python
# generation.py lines 141-147
if add_noise:
    vbw_rbw_ratio = vbw_hz / rbw_hz
    effective_noise_std_db = noise_factor_db * np.sqrt(vbw_rbw_ratio)
    psd_linear = add_measurement_noise(psd_linear, effective_noise_std_db)
```

**What This Means**:
- VBW = RBW: Standard noise level
- VBW < RBW: Longer integration time → LESS noise on trace
- VBW > RBW: Shorter integration time → MORE noise on trace

**Historical Note**: An earlier implementation incorrectly used `apply_vbw_smoothing()` which was a smoothing filter. This was removed because VBW in a real spectrum analyzer doesn't smooth the signal—it controls integration time which affects noise statistics.

### 3. Random Carrier Placement Algorithm

**Location**: `satellite_spectrum_emulator/transponder.py` lines 148-291

The `populate_with_random_carriers()` method implements a sophisticated algorithm for automatically placing non-overlapping carriers within a transponder.

**Key Algorithm Steps** (see transponder.py lines 232-289):

1. **Pick random center frequency** (100 kHz increments)
   - Must be within usable range (20% margin from edges by default)
   - Line 234-235: `center_offset_hz = np.random.uniform(usable_lower, usable_upper)`

2. **Calculate maximum symbol rate** based on edge distances
   - Line 244-249: Considers distance to both lower and upper edges
   - Accounts for RRC rolloff: `BW = SR × (1 + rolloff)`

3. **Pick random symbol rate** (100 kHz increments)
   - Line 260-267: Uses discrete steps for realistic symbol rates

4. **Pick random C/N and modulation**
   - Line 238-240: Uniform distribution within specified ranges

5. **Attempt to add carrier**
   - Line 282: `self.add_carrier(carrier)` validates fit and overlap
   - On failure, increment `iterations_without_success` and retry

**Edge Margin**: Default is 20% (line 225), meaning carriers must stay 20% of transponder bandwidth away from edges. This was changed from 10% to reduce overlap failures.

### 4. STATIC_CW Carrier Type for Unmodulated Tones

**Location**: `satellite_downlink_simulator/objects/enums.py` and `objects/carrier.py`

**What**: STATIC_CW is a special modulation type representing unmodulated continuous wave (CW) carriers - pure sinusoidal tones used for testing, beacons, or calibration.

**Key Characteristics**:
- **No symbol rate**: `symbol_rate_sps` must be `None` (validation enforced)
- **Fixed bandwidth**: Returns 100 Hz (representing oscillator phase noise width)
- **Gaussian PSD shape**: Rendered as narrow Gaussian peak instead of RRC-shaped spectrum
- **Power specification**: Still uses C/N like other carriers

**Implementation** (carrier.py):
```python
if self.modulation == ModulationType.STATIC_CW:
    if self.symbol_rate_sps is not None:
        raise ValueError("STATIC_CW carriers should not specify symbol_rate_sps")
    return 100.0  # Fixed bandwidth in Hz
```

**PSD Generation** (simulation/psd.py):
```python
if carrier.modulation == ModulationType.STATIC_CW:
    # Generate narrow Gaussian peak (100 Hz FWHM)
    sigma_hz = 100.0 / (2 * np.sqrt(2 * np.log(2)))
    carrier_shape = np.exp(-0.5 * (freq_rel_to_carrier / sigma_hz) ** 2)
    # Normalize so integral equals carrier power
    normalization = sigma_hz * np.sqrt(2 * np.pi)
    carrier_psd = power_watts / normalization * carrier_shape
```

**Usage Example**:
```python
cw_carrier = Carrier(
    name="Beacon",
    frequency_offset_hz=5e6,
    cn_db=20.0,
    modulation=ModulationType.STATIC_CW,
    carrier_type=CarrierType.FDMA,
    # No symbol_rate_sps or rrc_rolloff needed
)
```

**Why**: Satellite systems commonly use CW tones for frequency references, signal presence indication, and network synchronization. This carrier type allows realistic simulation of these signals.

### 5. Configurable Sample Rate with Nyquist Validation

**Location**: `satellite_downlink_simulator/generation.py` lines 220-229

**Design**: The `generate_iq()` function accepts an optional `sample_rate_hz` parameter that allows users to override the default 1.25× bandwidth oversampling.

**Why**: Different applications require different oversampling factors:
- **Standard use**: 1.25× provides adequate headroom (default)
- **High-fidelity analysis**: 2× or higher for better frequency domain resolution
- **Efficient storage**: Minimal oversampling when file size matters
- **Hardware testing**: Match specific ADC/DAC sample rates

**Validation**:
```python
# generation.py lines 224-229
if sample_rate_hz < bandwidth_hz:
    raise ValueError(
        f"sample_rate_hz must be >= bandwidth to satisfy Nyquist criterion. "
        f"Minimum required: {bandwidth_hz:.2f} Hz, got: {sample_rate_hz:.2f} Hz"
    )
```

**Trade-offs**:
- Lower sample rates (closer to bandwidth): Smaller files, faster processing, but less frequency headroom
- Higher sample rates: Better spectral clarity, easier filtering, but larger files and slower processing
- Validation prevents aliasing by enforcing the Nyquist criterion

## Important Implementation Details

### Power Spectral Density (PSD) Generation

**Function**: `generate_psd()` in `generation.py` lines 19-161

**Fast Frequency-Domain Approach**: Does NOT generate IQ data. Instead, directly computes PSD in frequency domain.

**Steps**:
1. Create frequency array based on RBW (line 82-87)
2. For each transponder:
   - Add shaped noise floor using RRC filter (lines 106-110)
   - For each carrier, add shaped carrier PSD (lines 113-135)
3. Add measurement noise scaled by VBW/RBW (lines 137-147)
4. Convert to dBm/Hz (line 150)

**RRC Filter Normalization** (lines 129-132):
```python
# Power spectral density = power / symbol_rate * |H(f)|²
# This normalization ensures ∫|H(f)|² df = symbol_rate
carrier_psd = power_watts / carrier.symbol_rate_sps * (carrier_shape ** 2)
```

### IQ Data Generation

**Function**: `generate_iq()` in `generation.py` lines 164-287

**Real Time-Domain Approach**: Generates actual modulated symbols with pulse shaping.

**Key Points**:
- Sample rate = 1.25 × bandwidth (default, configurable via `sample_rate_hz` parameter)
- Sample rate validation ensures Nyquist criterion is met (sample_rate >= bandwidth)
- Each carrier gets actual constellation symbols (line 327-329)
- RRC pulse shaping applied in time domain (lines 336-344)
- TDMA bursting applied via masking (lines 361-367, 372-408)
- All components frequency-shifted and summed (lines 258-276)

**TDMA Bursting** (`_apply_tdma_bursting()` lines 372-408):
- Creates periodic on/off mask
- Frame period = burst_time / duty_cycle
- Bursts start at frame boundaries
- Guard time = frame_period - burst_time

### Carrier Bandwidth Calculation

**Location**: `carrier.py` lines 121-133

```python
@property
def bandwidth_hz(self) -> float:
    """Occupied bandwidth including RRC rolloff."""
    return self.symbol_rate_sps * (1 + self.rrc_rolloff)
```

**Example**: 10 Msps with 0.35 rolloff = 13.5 MHz bandwidth

### TDMA Average Power

**Location**: `carrier.py` lines 135-145

For TDMA carriers, average power is scaled by duty cycle:
```python
def calculate_average_power_watts(self, noise_power_density_watts_per_hz: float) -> float:
    power_watts = self.calculate_power_watts(noise_power_density_watts_per_hz)
    if self.carrier_type == CarrierType.TDMA and self.duty_cycle:
        return power_watts * self.duty_cycle
    return power_watts
```

This is used in PSD generation to show average power level (what you'd see on a spectrum analyzer).

## File Structure and Responsibilities

```
satellite_downlink_simulator/
├── objects/                    # Object definitions
│   ├── __init__.py            # Exports all objects
│   ├── carrier.py             # Carrier class - individual signals
│   ├── transponder.py         # Transponder class - contains carriers
│   ├── beam.py                # Beam class - contains transponders
│   ├── metadata.py            # PSDMetadata and IQMetadata classes
│   └── enums.py               # All enumerations (Band, Modulation, etc.)
├── simulation/                 # Signal generation functions
│   ├── __init__.py            # Exports generation functions
│   ├── psd.py                 # PSD generation (frequency domain)
│   └── iq.py                  # IQ generation (time domain)
├── utils.py                    # Signal processing utilities (RRC, constellations, validation)
└── __init__.py                 # Top-level exports (backward compatible)
```

**Note**: The top-level `__init__.py` re-exports all classes and functions for backward compatibility. Old code using imports like `from satellite_downlink_simulator.carrier import Carrier` will continue to work.

### Key Classes

**Carrier** (`carrier.py`):
- Represents a single communication signal
- Stores: frequency offset, C/N, symbol rate, modulation, carrier type
- Calculates: power from C/N, bandwidth from symbol rate + rolloff

**Transponder** (`transponder.py`):
- Represents a satellite transponder containing multiple carriers
- Stores: center frequency, bandwidth, noise density, carrier list
- Validates: carriers fit within bandwidth, no overlaps (unless allowed)
- Method: `populate_with_random_carriers()` for auto-generation

**Beam** (`beam.py`):
- Represents a satellite beam containing multiple transponders
- Stores: band, polarization, direction, transponder list
- Calculates: total bandwidth, center frequency, total carriers

## Common Patterns and Conventions

### Frequency Representation

- **Carrier frequencies**: Relative to transponder center (offset in Hz)
- **Transponder frequencies**: Absolute downlink frequencies (Hz)
- **Beam center**: Calculated from min/max of all transponders

### Validation Pattern

Classes use `attrs` with validators and `__attrs_post_init__()`:

```python
@attrs.define
class Transponder:
    center_frequency_hz: float = attrs.field()

    @center_frequency_hz.validator
    def _validate_center_frequency_hz(self, attribute, value):
        validate_positive(value, "center_frequency_hz")

    def __attrs_post_init__(self):
        self.validate_carriers()  # Additional validation after init
```

### Parameter Naming

- `_hz` suffix: Frequency in Hertz
- `_sps` suffix: Symbol rate in symbols per second
- `_db` suffix: Value in decibels
- `_s` suffix: Time in seconds
- `_watts` or `_watts_per_hz`: Power values

## Testing and Examples

**Main Examples**: `main.py` contains 5 comprehensive examples:

1. **example1_single_transponder_fdma()**: Basic FDMA carriers
2. **example2_tdma_carrier()**: TDMA bursting demonstration
3. **example3_multi_transponder_beam()**: 6 transponders with random carriers
4. **example4_validation_tests()**: Error handling and validation
5. **example5_modulation_comparison()**: All modulation types

**Running Examples**:
```bash
python main.py
```

Generates 5 PNG plots demonstrating all features.

## Evolution History

### Major Changes Made During Development

1. **Carrier Power Specification Change**:
   - From: `power_watts` parameter
   - To: `cn_db` parameter with power calculated from transponder noise density
   - Reason: More intuitive for satellite communications

2. **VBW Implementation Fix**:
   - From: VBW used for trace smoothing via `apply_vbw_smoothing()`
   - To: VBW controls noise variance via capture time relationship
   - Reason: Match real spectrum analyzer behavior

3. **Random Carrier Placement**:
   - Added `populate_with_random_carriers()` method to Transponder
   - Enables automatic generation of realistic multi-carrier scenarios
   - Uses 20% edge margin (changed from 10% for better success rate)

4. **Example 3 Modifications**:
   - Initially tried 10 transponders (too slow for IQ generation)
   - Reduced to 6 transponders (user modification)
   - Skips IQ generation for large beams (only generates PSD)

5. **Configurable Sample Rate for IQ Generation**:
   - From: Hard-coded sample rate at 1.25× bandwidth
   - To: Optional `sample_rate_hz` parameter with validation
   - Default: Still 1.25× bandwidth when not specified
   - Validation: Enforces Nyquist criterion (sample_rate >= bandwidth)
   - Reason: Allows users to control oversampling factor for specific use cases while maintaining safe defaults

6. **STATIC_CW Modulation Type**:
   - Added: STATIC_CW to ModulationType enum for unmodulated carriers
   - Carrier changes: Made `symbol_rate_sps` optional; STATIC_CW must NOT have it
   - Bandwidth: STATIC_CW returns fixed 100 Hz (phase noise width)
   - PSD rendering: Gaussian peak instead of RRC-shaped spectrum
   - Reason: Enable simulation of beacon tones, frequency references, and calibration signals

7. **Code Restructuring into objects/ and simulation/ subdirectories**:
   - Reorganized: Split flat structure into logical subdirectories
   - objects/: carrier.py, transponder.py, beam.py, metadata.py, enums.py
   - simulation/: psd.py (was part of generation.py), iq.py (was part of generation.py)
   - Backward compatibility: Top-level __init__.py re-exports everything
   - Reason: Better code organization, clearer separation of concerns, easier to navigate

## Potential Gotchas

### 1. Transponder Noise Density Required for Carrier Power

Carrier power cannot be calculated without knowing the transponder's noise power density. Always ensure carriers are added to a transponder before trying to calculate their absolute power.

### 2. RBW Determines Number of Points

```python
num_points = int(span_hz / rbw_hz) + 1
```

Small RBW values create very large arrays. For a 216 MHz beam with 1 kHz RBW, you get 216,000 frequency points.

### 3. Allow Overlap Flag

By default, transponders reject overlapping carriers. Set `allow_overlap=True` if you need carriers to overlap (e.g., for interference scenarios).

### 4. TDMA Parameter Validation

TDMA carriers MUST have `burst_time_s` and `duty_cycle`. FDMA carriers MUST NOT have these parameters. Validation will raise `ValueError` if this is violated.

### 5. Sample Rate Configuration

**Default Behavior**: IQ generation automatically sets sample rate to 1.25× bandwidth if not specified.

**Override**: You can specify a custom `sample_rate_hz` parameter in `generate_iq()`, but it MUST be >= bandwidth_hz to satisfy the Nyquist criterion. The function will raise a `ValueError` with a descriptive message if this validation fails.

**Example**:
```python
# Use default 1.25× bandwidth
iq_data, metadata = generate_iq(transponder, duration_s=0.01)

# Use custom sample rate (2× bandwidth for extra headroom)
iq_data, metadata = generate_iq(transponder, duration_s=0.01, sample_rate_hz=2.0 * transponder.bandwidth_hz)

# This will raise ValueError (violates Nyquist criterion)
iq_data, metadata = generate_iq(transponder, duration_s=0.01, sample_rate_hz=0.5 * transponder.bandwidth_hz)
```

### 6. STATIC_CW Parameter Requirements

**STATIC_CW carriers have different parameter requirements than modulated carriers:**

**MUST NOT specify**:
- `symbol_rate_sps` - will raise ValueError if provided
- `rrc_rolloff` - ignored (can be left at default, but has no effect)

**MUST specify**:
- `modulation=ModulationType.STATIC_CW`
- `cn_db` - power level relative to noise
- `carrier_type` - FDMA or TDMA
- `frequency_offset_hz` - position within transponder

**Modulated carriers (BPSK, QPSK, etc.) MUST specify `symbol_rate_sps`**, otherwise validation will raise ValueError.

## Useful Utilities

### RRC Filter Functions (`utils.py`)

```python
# Frequency-domain RRC response (for PSD generation)
rrc_filter_freq(freq, symbol_rate, rolloff)

# Time-domain RRC taps (for IQ filtering)
rrc_filter_time(symbol_rate, rolloff, span=10, samples_per_symbol=8)
```

### Constellation Generation (`utils.py`)

```python
# Returns constellation points for a modulation type
generate_constellation(ModulationType.QPSK)  # Returns array of complex symbols
```

### Measurement Noise (`utils.py`)

```python
# Adds multiplicative noise in dB
add_measurement_noise(linear_array, noise_std_db)
```

## Dependencies and Requirements

- **numpy**: All numerical operations and array processing
- **attrs**: Class definitions with validation
- **matplotlib**: Plotting (only in examples, not core library)

**No scipy dependency**: All signal processing is implemented from scratch.

## Future Enhancement Ideas

If extending this library, consider:

1. **Phase noise modeling**: Add carrier phase noise
2. **Frequency errors**: Carrier frequency offset simulation
3. **Non-linear effects**: TWTA saturation, intermodulation
4. **Variable symbol rates**: Time-varying symbol rate carriers
5. **Custom constellations**: User-defined constellation points
6. **Polarization effects**: Cross-pol interference
7. **Multi-beam interference**: Overlapping beam scenarios
8. **Rain fade**: Atmospheric attenuation modeling

## Key Equations Reference

### Carrier Power from C/N
```
P_carrier = N₀ × BW × 10^(C/N_dB / 10)
```

### RRC Bandwidth
```
BW = Symbol_Rate × (1 + Rolloff)
```

### TDMA Timing
```
Frame_Period = Burst_Time / Duty_Cycle
Guard_Time = Frame_Period - Burst_Time
```

### VBW and Noise
```
Capture_Time = RBW / VBW
Noise_StdDev = Base_Noise × √(VBW / RBW)
```

### PSD Normalization
```
PSD(f) = (P / SR) × |H_RRC(f)|²

where ∫|H_RRC(f)|² df = SR (symbol rate)
```

## Debug Tips

### Visualizing Carrier Placement

```python
transponder = Transponder(...)
num_created = transponder.populate_with_random_carriers(num_carriers=10, seed=42)
print(f"Created {num_created}/10 carriers")

for carrier in transponder.carriers:
    lower = carrier.frequency_offset_hz - carrier.bandwidth_hz/2
    upper = carrier.frequency_offset_hz + carrier.bandwidth_hz/2
    print(f"{carrier.name}: [{lower/1e6:.3f}, {upper/1e6:.3f}] MHz")
```

### Checking Power Levels

```python
for carrier in transponder.carriers:
    power_w = carrier.calculate_power_watts(transponder.noise_power_density_watts_per_hz)
    power_dbm = 10 * np.log10(power_w * 1000)
    print(f"{carrier.name}: C/N={carrier.cn_db} dB, Power={power_dbm:.1f} dBm")
```

### Verifying PSD Integration

To check if PSD integrates to correct power:
```python
freq, psd_dbm_hz, meta = generate_psd(transponder, rbw_hz=1e3, vbw_hz=1e3)
psd_linear = 10 ** (psd_dbm_hz / 10) / 1000  # Convert dBm/Hz to W/Hz
total_power = np.trapz(psd_linear, freq)
print(f"Total integrated power: {total_power:.6e} W")
```

## Parameter Recommendations

### For Realistic Simulations

```python
# Transponder
bandwidth_hz = 36e6  # 36 MHz (typical Ku-band)
noise_power_density_watts_per_hz = 1e-15  # -120 dBm/Hz noise floor

# Carrier
cn_db = 10.0 to 25.0  # Typical operating range
symbol_rate_sps = 1e6 to 30e6  # 1 to 30 Msps
rrc_rolloff = 0.20 to 0.35  # Standard satellite values

# PSD Generation
rbw_hz = 10e3  # 10 kHz (good balance of resolution and speed)
vbw_hz = rbw_hz / 10  # 10:1 ratio for smooth traces

# IQ Generation
duration_s = 0.001 to 0.01  # 1-10 ms (sufficient for most analysis)
sample_rate_hz = None  # Use default 1.25× bandwidth, or specify custom rate
# For high-fidelity: sample_rate_hz = 2.0 * bandwidth_hz
# For hardware matching: sample_rate_hz = <your_adc_sample_rate>
```

## RF Pattern of Life Application

**Location**: `apps/rf_pattern_of_life_example/`

This application demonstrates using the core library to simulate temporal evolution of satellite spectrum over 24 hours, with time-varying carrier activity and interferers.

### Application Architecture

- **main.py**: CLI orchestration with argparse for simulation parameters
- **carrier_generator.py**: Generates static and dynamic carrier configurations with time windows
- **interferer_generator.py**: Creates CW interferers (STATIC_CW carriers) with sweeping behavior
- **psd_simulator.py**: Manages temporal simulation, generating PSD snapshots at regular intervals
- **visualization.py**: Creates plots (waterfall, activity timeline, snapshots, animated GIF)
- **config.json**: Example configuration file for carrier/interferer parameters

### Key Features

1. **Static Carriers**: Always-on carriers with optional time windows for activity periods
2. **Dynamic TDMA Carriers**: Bursting carriers with duty cycles
3. **Long-Duration Interferers**: Hours-long CW tones, often targeting specific carriers
4. **Short-Duration Interferers**: Minutes-long CW tones with faster sweeps
5. **Frequency Sweeping**: Linear and sawtooth sweep patterns for interferers
6. **Visualization Suite**: Waterfall plots, activity timelines, snapshot comparisons, animated GIFs

### CLI Arguments (with unit suffixes)

All time, frequency, and power arguments include unit suffixes for clarity:
- `--duration-min`: Simulation duration in minutes (default: 1440 = 24 hours)
- `--interval-min`: Snapshot interval in minutes (default: 5)
- `--rbw-hz`: Resolution bandwidth in Hz (default: 100000 = 100 kHz)
- `--vbw-hz`: Video bandwidth in Hz (default: 1000 = 1 kHz)

### Interferer Generation Strategy

Interferers are implemented as STATIC_CW carriers with boosted C/N to ensure visibility:

**Long-Duration Interferers** (3-23 hours):
- 80% chance to target a static carrier
- C/N boost: +5 to +15 dB above target carrier
- 40% chance to sweep (10-200 MHz/hr)
- Frequency positioned within or near target carrier bandwidth

**Short-Duration Interferers** (10 min - 3 hours):
- 90% chance to target any carrier (static or dynamic)
- C/N boost: +5 to +12 dB above target carrier
- 30% chance to sweep (50-500 MHz/hr)
- More aggressive sweep rates for short durations

**C/N Boosting Logic**: Critical for interferer visibility. Interferers must have higher C/N than their target carriers to appear prominently in PSD plots. The boost ensures interferers create noticeable spectral features.

### Animated Spectrogram

The `create_animated_spectrogram()` method generates 1920x1080 animated GIFs showing temporal evolution:
- Top 1/3: Current PSD line plot
- Bottom 2/3: Full 24-hour spectrogram with white line marking current time
- Configurable: `figsize` (resolution), `frame_decimation` (skip frames), `fps` (frame rate)
- Uses imageio library with matplotlib Agg backend for headless rendering

### Output Files

- `psd_snapshots.npz`: Compressed numpy arrays (time, frequency, PSD)
- `simulation_metadata.json`: Simulation parameters and statistics
- `activity_log.json`: Carrier/interferer counts at each snapshot
- `plots/waterfall_plot.png`: 24-hour spectrogram
- `plots/activity_timeline.png`: Carrier and interferer activity over time
- `plots/snapshot_comparison.png`: PSD at selected interesting times
- `plots/average_spectrum.png`: Mean spectrum with percentiles
- `plots/animated_spectrogram.gif`: Animated evolution (if created)

### Implementation Notes

**STATIC_CW Rendering**: Interferers use the STATIC_CW modulation type, which renders as impulses (delta functions) in the PSD. Power is concentrated in a single frequency bin: `PSD = power_watts / 100 Hz`. This creates sharp spectral lines characteristic of CW tones.

**Overlap Handling**: Interferers are added with `allow_overlap=True` temporarily enabled, since they intentionally overlap with existing carriers (that's the point of interference).

**Time Windows**: Static carriers can have activity windows (start/end times) to simulate scheduled communications or temporary outages.

## Contact and Maintenance

This library was developed iteratively with the following key objectives:
- Physical accuracy (matches real spectrum analyzer behavior)
- Computational efficiency (PSD generation without IQ processing)
- Ease of use (intuitive C/N-based power specification)
- Flexibility (random carrier generation, multiple modulation types)

When in doubt about implementation details, refer to the inline comments in the simulation code which contain detailed explanations of the physical modeling.
