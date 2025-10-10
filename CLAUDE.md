# CLAUDE.md - Context for Future Claude Sessions

This document provides critical context about the Satellite Spectrum Emulator codebase to help future Claude sessions understand the design decisions, implementation details, and evolution of this project.

## Project Overview

This is a Python library for simulating realistic satellite communication signals. It generates both Power Spectral Density (PSD) plots and time-domain IQ data for satellite transponders and beams.

**Primary Use Case**: Create realistic synthetic satellite spectrum data for testing, analysis, and visualization without requiring actual satellite hardware.

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
- Sample rate = 1.25 × bandwidth (line 217)
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
satellite_spectrum_emulator/
├── carrier.py          # Carrier class - individual signals
├── transponder.py      # Transponder class - contains carriers
├── beam.py             # Beam class - contains transponders
├── generation.py       # PSD and IQ generation functions
├── utils.py            # Signal processing (RRC filters, constellations)
├── metadata.py         # PSDMetadata and IQMetadata classes
└── enums.py            # All enumerations (Band, Modulation, etc.)
```

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

### 5. Sample Rate Auto-Calculation

IQ generation automatically sets sample rate to 1.25× bandwidth. You cannot override this. If you need a different sample rate, you'll need to modify `generation.py` line 217.

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
```

## Contact and Maintenance

This library was developed iteratively with the following key objectives:
- Physical accuracy (matches real spectrum analyzer behavior)
- Computational efficiency (PSD generation without IQ processing)
- Ease of use (intuitive C/N-based power specification)
- Flexibility (random carrier generation, multiple modulation types)

When in doubt about implementation details, refer to the inline comments in `generation.py` which contain detailed explanations of the physical modeling.
