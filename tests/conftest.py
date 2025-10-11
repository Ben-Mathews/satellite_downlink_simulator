"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np
import os
import base64
from datetime import datetime
from pathlib import Path
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
    # simple_fdma_carrier is at 0 MHz with 13.5 MHz BW (±6.75 MHz): -6.75 to +6.75 MHz
    # Place carrier2 at -12 MHz with 6.75 MHz BW (±3.375 MHz): -15.375 to -8.625 MHz
    # Gap between carriers: -8.625 to -6.75 = 1.875 MHz clearance
    carrier2 = Carrier(
        frequency_offset_hz=-12e6,
        cn_db=12.0,
        symbol_rate_sps=5e6,
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


# ============================================================================
# pytest-html hooks for HTML report generation with plots
# ============================================================================

@pytest.fixture(scope='session')
def report_dir(request):
    """
    Create timestamped report directory for HTML reports and plots.

    This fixture is used when --html flag is provided to pytest.
    It creates a directory structure:
        tests-reports/test-YYYYMMDD-HHMMSS/
        tests-reports/test-YYYYMMDD-HHMMSS/plots/
    """
    # Check if --html flag was used
    html_path = request.config.getoption('--html', default=None)
    if html_path is None:
        # No HTML report requested, return None
        return None

    # Extract directory from HTML path
    html_path = Path(html_path)
    report_base = html_path.parent

    # Create plots subdirectory
    plots_dir = report_base / 'plots'

    # Create directories
    report_base.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    # Store paths in config for later use
    request.config._report_dir = report_base
    request.config._plots_dir = plots_dir

    return plots_dir


@pytest.fixture
def plots_dir(request, report_dir):
    """
    Provide access to plots directory for individual tests.

    Tests can use this fixture to save plots that will be embedded
    in the HTML report.
    """
    if report_dir is None:
        # No HTML report, return None
        return None
    return report_dir


def pytest_configure(config):
    """
    Configure pytest-html plugin with custom CSS and setup.

    This hook runs once at the beginning of the test session.
    """
    # Only configure if --html flag is present
    if config.getoption('--html', default=None) is None:
        return

    # Add custom metadata
    if hasattr(config, '_metadata'):
        config._metadata['Project'] = 'Satellite Downlink Simulator'
        config._metadata['Test Type'] = 'Unit Tests with Visual Reports'


def pytest_html_report_title(report):
    """Set custom title for HTML report."""
    report.title = "Satellite Downlink Simulator - Test Report"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Hook to capture test results and attach plots to HTML report.

    This allows tests to attach plots by setting item.user_properties.
    """
    outcome = yield
    report = outcome.get_result()

    # Only process on test call phase (not setup/teardown)
    if report.when == 'call':
        # Check if HTML report is enabled
        html_enabled = item.config.getoption('--html', default=None) is not None

        if html_enabled:
            # Check if test has plots attached
            plots = [prop for prop in item.user_properties if prop[0] == 'plot']

            if plots:
                # Add plots to HTML report
                for _, plot_path in plots:
                    if os.path.exists(plot_path):
                        # Create HTML for plot with thumbnail
                        plot_html = create_plot_html(plot_path, item.config._plots_dir)
                        extra = {
                            'name': os.path.basename(plot_path),
                            'content': plot_html,
                            'format_type': 'html',
                            'extension': 'html'
                        }
                        report.extra = getattr(report, 'extra', [])
                        report.extra.append(extra)


def create_plot_html(plot_path, plots_dir):
    """
    Create HTML snippet for embedding a plot with thumbnail.

    Parameters
    ----------
    plot_path : str or Path
        Path to the plot image file
    plots_dir : Path
        Base plots directory for creating relative paths

    Returns
    -------
    str
        HTML string with thumbnail and full-size image link
    """
    plot_path = Path(plot_path)

    # Read image and encode as base64
    with open(plot_path, 'rb') as f:
        image_data = f.read()

    encoded = base64.b64encode(image_data).decode('utf-8')

    # Create HTML with thumbnail (max 400px wide) and click to view full size
    html = f"""
    <div style="margin: 10px 0;">
        <details>
            <summary style="cursor: pointer; color: #0066cc; font-weight: bold;">
                📊 {plot_path.stem} (click to expand)
            </summary>
            <div style="margin-top: 10px;">
                <img src="data:image/png;base64,{encoded}"
                     style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; padding: 5px;"
                     alt="{plot_path.name}"/>
            </div>
        </details>
    </div>
    """

    return html


@pytest.fixture
def attach_plot(request):
    """
    Fixture to attach plots to test reports.

    Usage in tests:
        def test_something(attach_plot, plots_dir):
            # Generate plot
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3])

            # Save plot
            plot_path = plots_dir / 'my_plot.png'
            fig.savefig(plot_path)
            plt.close(fig)

            # Attach to report
            attach_plot(plot_path)
    """
    def _attach(plot_path):
        """Attach a plot to the current test report."""
        request.node.user_properties.append(('plot', str(plot_path)))

    return _attach
