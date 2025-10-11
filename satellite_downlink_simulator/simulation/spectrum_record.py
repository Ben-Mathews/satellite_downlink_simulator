"""SpectrumRecord class for serializing PSD snapshots with metadata."""

import base64
import json
import numpy as np
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
import attrs

try:
    import blosc2
    BLOSC2_AVAILABLE = True
except ImportError:
    BLOSC2_AVAILABLE = False
    blosc2 = None

from ..objects.beam import Beam
from ..objects.transponder import Transponder
from ..objects.carrier import Carrier
from ..objects.enums import (
    Band, Polarization, BeamDirection,
    CarrierType, ModulationType, CarrierStandard
)


@attrs.define
class InterfererRecord:
    """
    Record of an interferer with sweep information and overlap tracking.

    Parameters
    ----------
    carrier : Carrier
        The carrier object representing this interferer
    start_time : datetime
        Start time when this interferer becomes active
    end_time : datetime
        End time when this interferer becomes inactive
    is_sweeping : bool
        Whether this interferer is sweeping in frequency
    sweep_rate_hz_per_s : float, optional
        Sweep rate in Hz/second (None if not sweeping)
    sweep_type : str, optional
        Type of sweep: 'linear', 'sawtooth', etc. (None if not sweeping)
    sweep_start_freq_hz : float, optional
        Starting frequency for sweep (None if not sweeping)
    sweep_end_freq_hz : float, optional
        Ending frequency for sweep (None if not sweeping)
    current_frequency_hz : float
        Current absolute frequency at this timestamp
    sweep_percentage : float
        Percentage through sweep cycle (0.0 to 1.0, always 0.0 if not sweeping)
    overlapping_transponders : List[str]
        Names of transponders currently overlapping with this interferer
    time_windows : List[Tuple[datetime, datetime]]
        All scheduled active time windows for this interferer
    """
    carrier: Carrier = attrs.field()
    start_time: datetime = attrs.field()
    end_time: datetime = attrs.field()
    is_sweeping: bool = attrs.field()
    current_frequency_hz: float = attrs.field()
    sweep_rate_hz_per_s: Optional[float] = attrs.field(default=None)
    sweep_type: Optional[str] = attrs.field(default=None)
    sweep_start_freq_hz: Optional[float] = attrs.field(default=None)
    sweep_end_freq_hz: Optional[float] = attrs.field(default=None)
    sweep_percentage: float = attrs.field(default=0.0)
    overlapping_transponders: List[str] = attrs.field(factory=list)
    time_windows: List[Tuple[datetime, datetime]] = attrs.field(factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert interferer record to dictionary for JSON serialization."""
        return {
            'carrier': _carrier_to_dict(self.carrier),
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'is_sweeping': self.is_sweeping,
            'sweep_rate_hz_per_s': self.sweep_rate_hz_per_s,
            'sweep_type': self.sweep_type,
            'sweep_start_freq_hz': self.sweep_start_freq_hz,
            'sweep_end_freq_hz': self.sweep_end_freq_hz,
            'current_frequency_hz': self.current_frequency_hz,
            'sweep_percentage': self.sweep_percentage,
            'overlapping_transponders': self.overlapping_transponders,
            'time_windows': [
                (start.isoformat(), end.isoformat())
                for start, end in self.time_windows
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InterfererRecord':
        """Create interferer record from dictionary."""
        carrier = _carrier_from_dict(data['carrier'])
        start_time = datetime.fromisoformat(data['start_time'])
        end_time = datetime.fromisoformat(data['end_time'])
        time_windows = [
            (datetime.fromisoformat(start), datetime.fromisoformat(end))
            for start, end in data['time_windows']
        ]

        return cls(
            carrier=carrier,
            start_time=start_time,
            end_time=end_time,
            is_sweeping=data['is_sweeping'],
            sweep_rate_hz_per_s=data.get('sweep_rate_hz_per_s'),
            sweep_type=data.get('sweep_type'),
            sweep_start_freq_hz=data.get('sweep_start_freq_hz'),
            sweep_end_freq_hz=data.get('sweep_end_freq_hz'),
            current_frequency_hz=data['current_frequency_hz'],
            sweep_percentage=data.get('sweep_percentage', 0.0),
            overlapping_transponders=data['overlapping_transponders'],
            time_windows=time_windows
        )


@attrs.define
class CarrierRecord:
    """
    Record of a carrier with time window information.

    Parameters
    ----------
    carrier : Carrier
        The carrier object
    time_windows : List[Tuple[datetime, datetime]]
        All scheduled active time windows for this carrier
    """
    carrier: Carrier = attrs.field()
    time_windows: List[Tuple[datetime, datetime]] = attrs.field(factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert carrier record to dictionary for JSON serialization."""
        return {
            'carrier': _carrier_to_dict(self.carrier),
            'time_windows': [
                (start.isoformat(), end.isoformat())
                for start, end in self.time_windows
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CarrierRecord':
        """Create carrier record from dictionary."""
        carrier = _carrier_from_dict(data['carrier'])
        time_windows = [
            (datetime.fromisoformat(start), datetime.fromisoformat(end))
            for start, end in data['time_windows']
        ]
        return cls(carrier=carrier, time_windows=time_windows)


@attrs.define
class SpectrumRecord:
    """
    Record of a PSD snapshot at a specific timestamp with full system state.

    This class captures a complete snapshot of the RF spectrum at a given time,
    including all active carriers, interferers, and the PSD data itself.

    Parameters
    ----------
    timestamp : datetime
        The time of this PSD snapshot
    cf_hz : float
        Center frequency of the PSD in Hz
    bw_hz : float
        Bandwidth of the PSD in Hz
    rbw_hz : float
        Resolution bandwidth used for PSD generation in Hz
    vbw_hz : float
        Video bandwidth used for PSD generation in Hz
    psd_compressed : bytes
        Blosc2-compressed PSD data
    beams : List[Beam]
        List of beams with their transponders and active carriers at this timestamp
    interferers : List[InterfererRecord]
        List of active interferers with sweep and overlap information
    """
    timestamp: datetime = attrs.field()
    cf_hz: float = attrs.field()
    bw_hz: float = attrs.field()
    rbw_hz: float = attrs.field()
    vbw_hz: float = attrs.field()
    psd_compressed: bytes = attrs.field()
    psd_shape: Tuple[int, ...] = attrs.field()
    beams: List[Beam] = attrs.field(factory=list)
    interferers: List[InterfererRecord] = attrs.field(factory=list)

    @staticmethod
    def compress_psd(psd: np.ndarray) -> bytes:
        """
        Compress PSD data using blosc2.

        Parameters
        ----------
        psd : np.ndarray
            PSD array to compress

        Returns
        -------
        bytes
            Compressed PSD data

        Raises
        ------
        ImportError
            If blosc2 is not installed
        """
        if not BLOSC2_AVAILABLE:
            raise ImportError(
                "blosc2 is required for PSD compression. "
                "Install it with: pip install blosc2"
            )

        # Ensure array is contiguous and in native byte order
        psd_array = np.ascontiguousarray(psd, dtype=np.float64)

        # Compress with blosc2 (default settings)
        compressed = blosc2.compress(psd_array.tobytes())
        return compressed

    @staticmethod
    def decompress_psd(compressed: bytes, shape: Tuple[int, ...]) -> np.ndarray:
        """
        Decompress PSD data using blosc2.

        Parameters
        ----------
        compressed : bytes
            Compressed PSD data
        shape : Tuple[int, ...]
            Shape of the original array

        Returns
        -------
        np.ndarray
            Decompressed PSD array

        Raises
        ------
        ImportError
            If blosc2 is not installed
        """
        if not BLOSC2_AVAILABLE:
            raise ImportError(
                "blosc2 is required for PSD decompression. "
                "Install it with: pip install blosc2"
            )

        # Decompress
        decompressed = blosc2.decompress(compressed)

        # Reconstruct array
        psd = np.frombuffer(decompressed, dtype=np.float64)
        return psd.reshape(shape)

    def get_psd(self) -> np.ndarray:
        """
        Get decompressed PSD data.

        Returns
        -------
        np.ndarray
            Decompressed PSD array
        """
        # Use stored shape
        return self.decompress_psd(self.psd_compressed, self.psd_shape)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert record to dictionary for JSON serialization.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation suitable for JSON
        """
        return {
            'timestamp': self.timestamp.isoformat(),
            'cf_hz': self.cf_hz,
            'bw_hz': self.bw_hz,
            'rbw_hz': self.rbw_hz,
            'vbw_hz': self.vbw_hz,
            'psd_compressed_base64': base64.b64encode(self.psd_compressed).decode('utf-8'),
            'psd_shape': list(self.psd_shape),  # Store actual shape
            'beams': [_beam_to_dict(beam) for beam in self.beams],
            'interferers': [interferer.to_dict() for interferer in self.interferers]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SpectrumRecord':
        """
        Create SpectrumRecord from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary representation from to_dict()

        Returns
        -------
        SpectrumRecord
            Reconstructed record
        """
        timestamp = datetime.fromisoformat(data['timestamp'])
        psd_compressed = base64.b64decode(data['psd_compressed_base64'])
        psd_shape = tuple(data['psd_shape'])
        beams = [_beam_from_dict(beam_data) for beam_data in data['beams']]
        interferers = [
            InterfererRecord.from_dict(int_data)
            for int_data in data['interferers']
        ]

        return cls(
            timestamp=timestamp,
            cf_hz=data['cf_hz'],
            bw_hz=data['bw_hz'],
            rbw_hz=data['rbw_hz'],
            vbw_hz=data['vbw_hz'],
            psd_compressed=psd_compressed,
            psd_shape=psd_shape,
            beams=beams,
            interferers=interferers
        )

    @classmethod
    def from_file(cls, filepath: str) -> List['SpectrumRecord']:
        """
        Load list of SpectrumRecords from JSON file.

        Parameters
        ----------
        filepath : str
            Path to JSON file

        Returns
        -------
        List[SpectrumRecord]
            List of spectrum records
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        return [cls.from_dict(record_data) for record_data in data]

    @staticmethod
    def to_file(records: List['SpectrumRecord'], filepath: str) -> None:
        """
        Save list of SpectrumRecords to JSON file.

        Parameters
        ----------
        records : List[SpectrumRecord]
            List of spectrum records to save
        filepath : str
            Path to output JSON file
        """
        data = [record.to_dict() for record in records]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# Helper functions for serialization/deserialization

def _carrier_to_dict(carrier: Carrier) -> Dict[str, Any]:
    """Convert Carrier object to dictionary."""
    return {
        'frequency_offset_hz': carrier.frequency_offset_hz,
        'cn_db': carrier.cn_db,
        'modulation': carrier.modulation.value,
        'carrier_type': carrier.carrier_type.value,
        'symbol_rate_sps': carrier.symbol_rate_sps,
        'rrc_rolloff': carrier.rrc_rolloff,
        'standard': carrier.standard.value,
        'burst_time_s': carrier.burst_time_s,
        'duty_cycle': carrier.duty_cycle,
        'name': carrier.name
    }


def _carrier_from_dict(data: Dict[str, Any]) -> Carrier:
    """Create Carrier object from dictionary."""
    return Carrier(
        frequency_offset_hz=data['frequency_offset_hz'],
        cn_db=data['cn_db'],
        modulation=ModulationType(data['modulation']),
        carrier_type=CarrierType(data['carrier_type']),
        symbol_rate_sps=data['symbol_rate_sps'],
        rrc_rolloff=data['rrc_rolloff'],
        standard=CarrierStandard(data['standard']),
        burst_time_s=data['burst_time_s'],
        duty_cycle=data['duty_cycle'],
        name=data['name']
    )


def _transponder_to_dict(transponder: Transponder, active_carriers: List[CarrierRecord]) -> Dict[str, Any]:
    """Convert Transponder object to dictionary with active carriers."""
    return {
        'center_frequency_hz': transponder.center_frequency_hz,
        'bandwidth_hz': transponder.bandwidth_hz,
        'noise_power_density_watts_per_hz': transponder.noise_power_density_watts_per_hz,
        'noise_rolloff': transponder.noise_rolloff,
        'name': transponder.name,
        'carriers': [carrier_rec.to_dict() for carrier_rec in active_carriers]
    }


def _transponder_from_dict(data: Dict[str, Any]) -> Transponder:
    """Create Transponder object from dictionary."""
    carriers = [
        CarrierRecord.from_dict(carrier_data).carrier
        for carrier_data in data['carriers']
    ]

    return Transponder(
        center_frequency_hz=data['center_frequency_hz'],
        bandwidth_hz=data['bandwidth_hz'],
        noise_power_density_watts_per_hz=data['noise_power_density_watts_per_hz'],
        carriers=carriers,
        noise_rolloff=data['noise_rolloff'],
        name=data['name'],
        allow_overlap=True  # Allow overlap for deserialization
    )


def _beam_to_dict(beam: Beam) -> Dict[str, Any]:
    """Convert Beam object to dictionary."""
    # Note: This assumes beam.transponders contains only active carriers
    # The calling code should filter carriers before creating the beam
    return {
        'band': beam.band.value,
        'polarization': beam.polarization.value,
        'direction': beam.direction.value,
        'name': beam.name,
        'transponders': [
            _transponder_to_dict(
                transponder,
                [CarrierRecord(carrier=c, time_windows=[]) for c in transponder.carriers]
            )
            for transponder in beam.transponders
        ]
    }


def _beam_from_dict(data: Dict[str, Any]) -> Beam:
    """Create Beam object from dictionary."""
    transponders = [
        _transponder_from_dict(transponder_data)
        for transponder_data in data['transponders']
    ]

    return Beam(
        band=Band(data['band']),
        polarization=Polarization(data['polarization']),
        direction=BeamDirection(data['direction']),
        transponders=transponders,
        name=data['name']
    )
