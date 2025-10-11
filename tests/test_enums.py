"""Smoke tests for enum classes."""

from satellite_downlink_simulator.objects.enums import (
    Band,
    Polarization,
    BeamDirection,
    CarrierType,
    ModulationType,
    CarrierStandard
)


class TestEnumInstantiation:
    """Test that all enums can be instantiated."""

    def test_band_enum(self):
        """Test Band enum values."""
        assert Band.C is not None
        assert Band.KA is not None
        assert Band.L is not None
        assert Band.X is not None

    def test_polarization_enum(self):
        """Test Polarization enum values."""
        assert Polarization.RHCP is not None
        assert Polarization.LHCP is not None
        assert Polarization.VERTICAL is not None
        assert Polarization.HORIZONTAL is not None

    def test_beam_direction_enum(self):
        """Test BeamDirection enum values."""
        assert BeamDirection.UPLINK is not None
        assert BeamDirection.DOWNLINK is not None

    def test_carrier_type_enum(self):
        """Test CarrierType enum values."""
        assert CarrierType.FDMA is not None
        assert CarrierType.TDMA is not None

    def test_modulation_type_enum(self):
        """Test ModulationType enum values."""
        assert ModulationType.BPSK is not None
        assert ModulationType.QPSK is not None
        assert ModulationType.QAM16 is not None
        assert ModulationType.APSK16 is not None
        assert ModulationType.APSK32 is not None
        assert ModulationType.STATIC_CW is not None

    def test_carrier_standard_enum(self):
        """Test CarrierStandard enum values."""
        assert CarrierStandard.NONE is not None
        assert CarrierStandard.DVB_S is not None
        assert CarrierStandard.DVB_S2 is not None
        assert CarrierStandard.IESS_308 is not None


class TestEnumValues:
    """Test enum string values."""

    def test_band_values(self):
        """Test Band enum string values."""
        assert Band.C.value == "C"
        assert Band.KA.value == "KA"
        assert Band.L.value == "L"

    def test_polarization_values(self):
        """Test Polarization enum string values."""
        assert Polarization.RHCP.value == "RHCP"
        assert Polarization.LHCP.value == "LHCP"

    def test_modulation_type_values(self):
        """Test ModulationType enum string values."""
        assert ModulationType.BPSK.value == "BPSK"
        assert ModulationType.QPSK.value == "QPSK"
        assert ModulationType.STATIC_CW.value == "STATIC_CW"
