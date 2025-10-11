"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np
from satellite_downlink_simulator.objects.carrier import Carrier
from satellite_downlink_simulator.objects.transponder import Transponder
from satellite_downlink_simulator.objects.beam import Beam
from satellite_downlink_simulator.objects.enums import (
    CarrierType,
    ModulationType,
    CarrierStandard,
    Band,
    Polarization,
    BeamDirection
)


@pytest.fixture
def simple_fdma_carrier():
    """Create a simple FDMA QPSK carrier for testing."""
    return Carrier(
        frequency_offset_hz=0.0,
        cn_db=15.0,
        symbol_rate_sps=10e6,
        modulation=ModulationType.QPSK,
        carrier_type=CarrierType.FDMA,
        rrc_rolloff=0.35,
        name="Test FDMA Carrier"
    )


@pytest.fixture
def simple_tdma_carrier():
    """Create a simple TDMA QPSK carrier for testing."""
    return Carrier(
        frequency_offset_hz=5e6,
        cn_db=18.0,
        symbol_rate_sps=5e6,
        modulation=ModulationType.QPSK,
        carrier_type=CarrierType.TDMA,
        rrc_rolloff=0.25,
        burst_time_s=0.001,
        duty_cycle=0.3,
        name="Test TDMA Carrier"
    )


@pytest.fixture
def static_cw_carrier():
    """Create a STATIC_CW carrier for testing."""
    return Carrier(
        frequency_offset_hz=-8e6,
        cn_db=20.0,
        modulation=ModulationType.STATIC_CW,
        carrier_type=CarrierType.FDMA,
        name="Test CW Carrier"
    )


@pytest.fixture
def simple_transponder():
    """Create a simple transponder for testing."""
    return Transponder(
        center_frequency_hz=12.5e9,
        bandwidth_hz=36e6,
        noise_power_density_watts_per_hz=1e-15,
        name="Test Transponder"
    )


@pytest.fixture
def transponder_with_carriers(simple_transponder, simple_fdma_carrier):
    """Create a transponder with carriers for testing."""
    simple_transponder.add_carrier(simple_fdma_carrier)

    # Add a second carrier that doesn't overlap
    # simple_fdma_carrier is at 0 MHz with 13.5 MHz BW (±6.75 MHz)
    # Place carrier2 at -10 MHz with 10.8 MHz BW (±5.4 MHz)
    carrier2 = Carrier(
        frequency_offset_hz=-10e6,
        cn_db=12.0,
        symbol_rate_sps=8e6,
        modulation=ModulationType.BPSK,
        carrier_type=CarrierType.FDMA,
        name="Carrier 2"
    )
    simple_transponder.add_carrier(carrier2)

    return simple_transponder


@pytest.fixture
def simple_beam():
    """Create a simple beam for testing."""
    return Beam(
        band=Band.KA,
        polarization=Polarization.RHCP,
        direction=BeamDirection.DOWNLINK,
        name="Test Beam"
    )


@pytest.fixture
def beam_with_transponders(simple_beam, transponder_with_carriers):
    """Create a beam with transponders for testing."""
    simple_beam.add_transponder(transponder_with_carriers)

    # Add a second transponder
    transponder2 = Transponder(
        center_frequency_hz=12.6e9,
        bandwidth_hz=36e6,
        noise_power_density_watts_per_hz=1e-15,
        name="Transponder 2"
    )
    simple_beam.add_transponder(transponder2)

    return simple_beam


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42
