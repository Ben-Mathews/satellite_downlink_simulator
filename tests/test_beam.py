"""Smoke tests for Beam class."""

import pytest
from satellite_downlink_simulator.objects.beam import Beam
from satellite_downlink_simulator.objects.enums import Band, Polarization, BeamDirection


class TestBeamInstantiation:
    """Test beam instantiation."""

    def test_create_simple_beam(self, simple_beam):
        """Test creating a basic beam."""
        assert simple_beam is not None
        assert simple_beam.band == Band.KA
        assert simple_beam.polarization == Polarization.RHCP
        assert simple_beam.direction == BeamDirection.DOWNLINK
        assert len(simple_beam.transponders) == 0

    def test_create_beam_with_transponders(self, beam_with_transponders):
        """Test creating a beam with transponders."""
        assert beam_with_transponders is not None
        assert len(beam_with_transponders.transponders) == 2


class TestBeamTransponderManagement:
    """Test adding transponders to beams."""

    def test_add_transponder(self, simple_beam, simple_transponder):
        """Test adding a transponder to beam."""
        simple_beam.add_transponder(simple_transponder)
        assert len(simple_beam.transponders) == 1
        assert simple_beam.transponders[0] == simple_transponder

    def test_add_multiple_transponders(self, simple_beam, simple_transponder, transponder_with_carriers):
        """Test adding multiple transponders."""
        simple_beam.add_transponder(simple_transponder)
        simple_beam.add_transponder(transponder_with_carriers)
        assert len(simple_beam.transponders) == 2


class TestBeamProperties:
    """Test beam property calculations."""

    def test_center_frequency_single_transponder(self, simple_beam, simple_transponder):
        """Test center frequency with single transponder."""
        simple_beam.add_transponder(simple_transponder)
        # With one transponder, center should be transponder center
        assert simple_beam.center_frequency_hz == simple_transponder.center_frequency_hz

    def test_center_frequency_multiple_transponders(self, beam_with_transponders):
        """Test center frequency with multiple transponders."""
        center = beam_with_transponders.center_frequency_hz
        assert center is not None
        assert center > 0

    def test_center_frequency_empty_beam(self, simple_beam):
        """Test center frequency for empty beam returns 0."""
        assert simple_beam.center_frequency_hz == 0.0

    def test_total_bandwidth_single_transponder(self, simple_beam, simple_transponder):
        """Test total bandwidth with single transponder."""
        simple_beam.add_transponder(simple_transponder)
        assert simple_beam.total_bandwidth_hz == simple_transponder.bandwidth_hz

    def test_total_bandwidth_multiple_transponders(self, beam_with_transponders):
        """Test total bandwidth with multiple transponders."""
        bandwidth = beam_with_transponders.total_bandwidth_hz
        assert bandwidth is not None
        assert bandwidth > 0

    def test_total_bandwidth_empty_beam(self, simple_beam):
        """Test total bandwidth for empty beam returns 0."""
        assert simple_beam.total_bandwidth_hz == 0.0

    def test_total_carriers(self, beam_with_transponders):
        """Test total carrier count across all transponders."""
        total = beam_with_transponders.total_carriers
        # transponder_with_carriers has 2 carriers, transponder2 has 0
        assert total == 2

    def test_total_carriers_empty_beam(self, simple_beam):
        """Test total carriers for empty beam."""
        assert simple_beam.total_carriers == 0


class TestBeamStringRepresentation:
    """Test beam string representation."""

    def test_str_beam(self, simple_beam):
        """Test string representation of beam."""
        beam_str = str(simple_beam)
        assert "Beam" in beam_str
        assert "KA" in beam_str
        assert "RHCP" in beam_str
        assert "DOWNLINK" in beam_str

    def test_str_with_transponders(self, beam_with_transponders):
        """Test string representation includes transponder count."""
        beam_str = str(beam_with_transponders)
        assert "2 transponder(s)" in beam_str
