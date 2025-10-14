# CLAUDE.md - Satellite Downlink Simulator Context

**Last Updated**: October 2025 - MODULATED interferer support and SpectrumRecord JSON export

## Project Overview

Python library for modeling and simulating satellite communication systems with:
1. **Object Model**: Carrier, Transponder, Beam hierarchy
2. **Signal Generation**: PSD (frequency domain) and IQ (time domain) synthesis
3. **Applications**: RF Pattern of Life with 24-hour temporal simulations

**Core Library** (`satellite_downlink_simulator/`):
- `objects/`: Carrier, Transponder, Beam classes with validation
- `simulation/`: PSD and IQ generation (psd.py, iq.py)
- `utils.py`: Signal processing (RRC filters, constellations, noise)

**Key Philosophy**: Objects represent physical RF architecture; simulation functions generate synthetic measurements

## Critical Design Decisions

### 1. C/N Power Specification
**Location**: `satellite_downlink_simulator/objects/carrier.py:88-98`

Carriers use **C/N ratio (dB)**, NOT absolute power. Industry-standard approach where power is calculated from transponder noise density:
```python
carrier_power_watts = (10^(cn_db/10)) × noise_power_watts
```

### 2. VBW Controls Noise Variance, Not Smoothing
**Location**: `satellite_downlink_simulator/simulation/psd.py:137-147`

VBW affects capture time, which controls noise variance:
- Capture Time ≈ RBW / VBW
- Noise Std Dev = Base_Noise × √(VBW / RBW)
- VBW < RBW → Less noise (longer integration)
- VBW > RBW → More noise (shorter integration)

### 3. Random Carrier Placement
**Location**: `satellite_downlink_simulator/objects/transponder.py:148-291`

`populate_with_random_carriers()` algorithm:
1. Pick random center frequency (100 kHz increments, 20% edge margin)
2. Calculate max symbol rate based on edge distances
3. Pick random symbol rate (100 kHz increments)
4. Pick random C/N and modulation
5. Validate fit and no overlap

### 4. STATIC_CW Modulation Type
**Location**: `satellite_downlink_simulator/objects/enums.py`, `objects/carrier.py`

Special modulation for unmodulated CW tones:
- **No symbol_rate_sps** (must be None)
- **Fixed 100 Hz bandwidth** (phase noise width)
- **Gaussian PSD shape** instead of RRC-shaped
- Used for beacons, frequency references, calibration

### 5. Configurable Sample Rate
**Location**: `satellite_downlink_simulator/simulation/iq.py:220-229`

`generate_iq()` accepts optional `sample_rate_hz` parameter:
- Default: 1.25× bandwidth
- Must satisfy Nyquist (≥ bandwidth_hz)
- Allows control of oversampling for different applications

## Key Implementation Details

### PSD Generation (Frequency Domain)
**Function**: `generate_psd()` in `simulation/psd.py:19-161`

Fast approach that directly computes PSD without IQ generation:
1. Create frequency array based on RBW
2. Add shaped noise floor (RRC filtered)
3. Add shaped carrier PSDs with normalization: `PSD = (P/SR) × |H_RRC(f)|²`
4. Add measurement noise scaled by VBW/RBW

### IQ Generation (Time Domain)
**Function**: `generate_iq()` in `simulation/iq.py:164-287`

Generates actual modulated symbols:
- Constellation symbols for each carrier
- RRC pulse shaping in time domain
- TDMA bursting via masking
- Frequency shifting and summation

## File Structure

```
satellite_downlink_simulator/
├── objects/          # Carrier, Transponder, Beam, metadata, enums
├── simulation/       # psd.py, iq.py, spectrum_record.py
├── utils.py          # Signal processing utilities
└── __init__.py       # Top-level exports (backward compatible)

apps/rf_pattern_of_life_example/
├── main.py                      # CLI orchestration
├── carrier_generator.py         # Static/dynamic carrier configs
├── interferer_generator.py      # CW and MODULATED interferers
├── psd_simulator.py             # Temporal PSD simulation
├── visualization.py             # Waterfall, timeline, animated plots
├── spectrum_records_utility.py  # Regenerate plots from JSON
└── config.json                  # Example configuration
```

## RF Pattern of Life Application
**Location**: `apps/rf_pattern_of_life_example/`

Simulates 24-hour spectrum evolution with:
- **Static carriers**: Always-on with optional time windows
- **Dynamic TDMA carriers**: Bursting with duty cycles
- **CW interferers**: Unmodulated tones (100 Hz bandwidth) with optional frequency sweeping
  - Long-duration: 3-23 hours, 80% target carriers, C/N boost +5-15 dB
  - Short-duration: 10 min-3 hours, 90% target carriers, C/N boost +5-12 dB
  - Sweep types: None, linear, sawtooth
- **MODULATED interferers**: Modulated signals (BPSK, QPSK, QAM16, APSK16) with static frequencies
  - Long-duration: 3-23 hours, 80% target carriers, C/N boost +5-15 dB, symbol rates 1-15 Msps
  - Short-duration: 10 min-3 hours, 90% target carriers, C/N boost +5-12 dB, symbol rates 1-10 Msps
  - Always static frequency (no sweeping)
  - Bandwidth: symbol_rate × (1 + rolloff) [typically 1.2-20 MHz]

**CLI Arguments**:
- `--duration-min`: Simulation duration (default: 1440 = 24 hours)
- `--interval-min`: Snapshot interval (default: 5)
- `--rbw-hz`: Resolution bandwidth (default: 100000 Hz)
- `--vbw-hz`: Video bandwidth (default: 1000 Hz)
- `--interferers-long`: Number of long-duration CW interferers (default: 2)
- `--interferers-short`: Number of short-duration CW interferers (default: 8)
- `--interferers-long-modulated`: Number of long-duration MODULATED interferers (default: 0)
- `--interferers-short-modulated`: Number of short-duration MODULATED interferers (default: 0)
- `--export-json`: Enable JSON export
- `--start-datetime`: Start time (ISO format)

**Output Files**:
- `psd_snapshots.npz`: Compressed numpy arrays
- `simulation_metadata.json`: Parameters and statistics
- `activity_log.json`: Carrier/interferer counts per snapshot
- `plots/*.png`: Waterfall, activity timeline, snapshots, average spectrum
- `plots/animated_spectrogram.{gif,mp4}`: Animated evolution with interferer highlighting

**Visualization Highlighting**:
- Animated spectrogram highlights interferers with translucent red overlays
- CW interferers (100 Hz bandwidth): Minimum 5 MHz span for visibility
- MODULATED interferers: Actual bandwidth based on symbol_rate × (1 + rolloff)
- Highlighting adapts automatically based on interferer bandwidth

### JSON Export (SpectrumRecord)
**Added**: October 2025

Exports complete spectrum state with:
- **SpectrumRecord**: Timestamp, CF, BW, RBW, VBW, compressed PSD, beam hierarchy
- **InterfererRecord**: Sweep parameters, current frequency, overlap tracking
- **blosc2 compression**: Typically 10-20× reduction

**Use Cases**: ML training data, post-processing, data sharing, replay analysis

**Loading**:
```python
from satellite_downlink_simulator.simulation import SpectrumRecord
records = SpectrumRecord.from_file('output/spectrum_records_*.json')
psd_array = records[0].get_psd()  # Decompress
```

### Spectrum Records Utility
**Location**: `apps/rf_pattern_of_life_example/spectrum_records_utility.py`

Regenerates visualizations from exported JSON without re-running simulation.

**Usage**: `python spectrum_records_utility.py <json_file> [--no-animation] [--format mp4]`

## Testing
**Location**: `tests/` (130 tests, ~16s execution)

**Structure**: test_carrier.py (22), test_transponder.py (12), test_beam.py (14), test_psd.py (12), test_iq.py (13), test_utils.py (27), comparison tests (8)

**Key Tests**:
- Object validation (STATIC_CW vs modulated requirements)
- Signal generation (PSD, IQ, TDMA bursting)
- **Critical**: PSD vs IQ-FFT comparison validates frequency/time domain equivalence

**Running Tests**:
```bash
pytest                              # Basic run
pytest --html=test_report.html      # With embedded plots
pytest -v tests/test_carrier.py     # Specific file
```

**Shared Fixtures** (`tests/conftest.py`):
- `simple_transponder`: 36 MHz at 12.5 GHz
- `simple_fdma_carrier`: 10 Msps QPSK
- `transponder_with_carriers`: 2 non-overlapping carriers (carefully positioned)

**Current Status**: All 130 tests pass, zero warnings (January 2025)

## Common Patterns

**Frequency Representation**:
- Carrier: Offset from transponder center (Hz)
- Transponder: Absolute downlink frequency (Hz)
- Beam: Calculated from min/max transponders

**Validation Pattern**:
```python
@attrs.define
class Transponder:
    @field.validator
    def _validate(self, attribute, value):
        validate_positive(value, "field_name")
```

**Parameter Naming**: `_hz`, `_sps`, `_db`, `_s`, `_watts`, `_watts_per_hz`

## Key Equations

```
Carrier Power: P = N₀ × BW × 10^(C/N_dB / 10)
RRC Bandwidth: BW = SR × (1 + rolloff)
TDMA Timing: Frame_Period = Burst_Time / Duty_Cycle
VBW Noise: Noise_StdDev = Base × √(VBW / RBW)
PSD Normalization: PSD(f) = (P/SR) × |H_RRC(f)|²
```

## Dependencies

- **numpy ≥ 1.20.0**: Numerical operations
- **scipy ≥ 1.7.0**: Signal processing
- **attrs ≥ 21.0.0**: Class definitions with validation
- **matplotlib ≥ 3.3.0**: Plotting
- **imageio[ffmpeg] ≥ 2.9.0**: Animated GIF/MP4 export
- **blosc2 ≥ 2.0.0**: PSD compression for JSON export

## Parameter Recommendations

```python
# Transponder
bandwidth_hz = 36e6                          # 36 MHz (typical Ku-band)
noise_power_density_watts_per_hz = 1e-15     # -120 dBm/Hz

# Carrier
cn_db = 10.0 to 25.0                         # Typical range
symbol_rate_sps = 1e6 to 30e6                # 1-30 Msps
rrc_rolloff = 0.20 to 0.35                   # Standard values

# PSD Generation
rbw_hz = 10e3                                # 10 kHz
vbw_hz = rbw_hz / 10                         # 10:1 ratio

# IQ Generation
duration_s = 0.001 to 0.01                   # 1-10 ms
sample_rate_hz = None                        # Use default 1.25×, or specify custom
```

## Critical Gotchas

1. **Transponder Noise Density Required**: Carrier power needs transponder noise density
2. **RBW Determines Points**: `num_points = span_hz / rbw_hz + 1` - small RBW → large arrays
3. **Allow Overlap Flag**: Default rejects overlaps; set `allow_overlap=True` for interference
4. **TDMA Validation**: TDMA carriers MUST have `burst_time_s` and `duty_cycle`; FDMA MUST NOT
5. **Sample Rate**: Must be ≥ bandwidth_hz (Nyquist criterion enforced)
6. **STATIC_CW**: Must NOT have `symbol_rate_sps`; modulated carriers MUST have it

## Git Workflow

**IMPORTANT**: Do NOT commit without explicit user approval.

Workflow:
1. Make code changes as requested
2. Run tests and verify functionality
3. **STOP before committing** - wait for explicit user instruction
4. Only run `git commit` and `git push` when user explicitly says so

## Evolution History

1. **Carrier Power**: Changed from `power_watts` to `cn_db` for industry alignment
2. **VBW Implementation**: Fixed from smoothing to noise variance control
3. **Random Carrier Placement**: Added auto-generation with 20% edge margin
4. **Configurable Sample Rate**: Added optional `sample_rate_hz` parameter with Nyquist validation
5. **STATIC_CW Type**: Added for unmodulated CW tones
6. **Code Restructuring**: Organized into objects/ and simulation/ subdirectories
7. **JSON Export**: Added SpectrumRecord serialization with blosc2 compression
8. **MODULATED Interferers** (October 2025): Added support for modulated interferers in RF Pattern of Life
   - New interferer type: `InterfererType` enum with CW and MODULATED options
   - MODULATED interferers use regular modulation (BPSK, QPSK, QAM16, APSK16)
   - Always static frequency (no sweeping, unlike CW interferers)
   - CLI arguments: `--interferers-long-modulated`, `--interferers-short-modulated`
   - Visualization highlighting adapts to bandwidth (CW=5MHz min, MODULATED=actual BW)
   - Updated `PSDSnapshot` dataclass to include `interferer_bandwidths_hz` field

## Test Maintenance

**When tests fail**:
1. Check fixtures for carrier overlap (common issue)
2. Verify STATIC_CW vs modulated parameter requirements
3. Ensure data types match (complex64, not complex128)
4. Use `-v` flag for detailed output

**Major fixes (January 2025)**:
- Carrier overlap in fixtures (repositioned for clearance)
- STATIC_CW IQ generation (added special handling)
- `np.trapz` → `np.trapezoid` (NumPy 2.0 deprecation)

## Contact and Maintenance

Developed with focus on:
- Physical accuracy (matches real spectrum analyzers)
- Computational efficiency (PSD without IQ generation)
- Ease of use (intuitive C/N specification)
- Flexibility (random generation, multiple modulation types)

Refer to inline code comments for detailed physical modeling explanations.
