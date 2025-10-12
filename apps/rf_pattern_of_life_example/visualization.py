"""Visualization tools for pattern-of-life simulation results.

Generates waterfall plots, average spectra, and snapshot comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import os


class Visualizer:
    """Creates visualizations for pattern-of-life simulation."""

    def __init__(self, time_array: np.ndarray, frequency_array: np.ndarray,
                 psd_array: np.ndarray, metadata, output_dir: str):
        """Initialize visualizer with simulation data.

        Args:
            time_array: Time points in minutes since start (shape: N_time)
            frequency_array: Frequency points in Hz (shape: N_freq)
            psd_array: PSD data in dBm/Hz (shape: N_time × N_freq)
            metadata: SimulationMetadata object
            output_dir: Directory to save plots
        """
        self.time_array = time_array
        self.frequency_array = frequency_array
        self.psd_array = psd_array
        self.metadata = metadata
        self.output_dir = output_dir

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Convert time to hours for plotting
        self.time_hours = time_array / 60.0

        # Convert frequency to GHz for plotting
        self.frequency_ghz = frequency_array / 1e9

        print(f"Initialized visualizer")
        print(f"  Time range: {self.time_hours[0]:.1f} - {self.time_hours[-1]:.1f} hrs")
        print(f"  Frequency range: {self.frequency_ghz[0]:.3f} - {self.frequency_ghz[-1]:.3f} GHz")
        print(f"  PSD shape: {psd_array.shape}")

    def create_waterfall_plot(self, vmin: Optional[float] = None,
                             vmax: Optional[float] = None,
                             cmap: str = 'viridis'):
        """Create waterfall/spectrogram plot.

        Args:
            vmin: Minimum PSD value for colormap (auto if None)
            vmax: Maximum PSD value for colormap (auto if None)
            cmap: Matplotlib colormap name
        """
        print(f"\nCreating waterfall plot...")

        # Auto-scale colormap if not specified
        if vmin is None:
            # Use 10th percentile to avoid noise floor dominating
            vmin = np.percentile(self.psd_array, 10)
        if vmax is None:
            # Use 99th percentile to avoid outliers
            vmax = np.percentile(self.psd_array, 99)

        print(f"  PSD range: {vmin:.1f} to {vmax:.1f} dBm/Hz")

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))

        # Create mesh for pcolormesh
        # X: frequency (GHz), Y: time (hours), C: PSD (dBm/Hz)
        X, Y = np.meshgrid(self.frequency_ghz, self.time_hours)

        # Plot waterfall
        im = ax.pcolormesh(X, Y, self.psd_array, shading='auto',
                          cmap=cmap, vmin=vmin, vmax=vmax)

        # Labels and title
        ax.set_xlabel('Frequency (GHz)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Time (hours)', fontsize=12, fontweight='bold')
        ax.set_title('24-Hour Pattern of Life - Waterfall Plot', fontsize=14, fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, label='PSD (dBm/Hz)')
        cbar.ax.tick_params(labelsize=10)

        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')

        # Tight layout
        plt.tight_layout()

        # Save
        output_file = os.path.join(self.output_dir, 'waterfall_plot.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved to {output_file}")

        plt.close()

    def create_average_spectrum(self):
        """Create average spectrum plot."""
        print(f"\nCreating average spectrum plot...")

        # Calculate average and percentiles
        avg_psd = np.mean(self.psd_array, axis=0)
        p10_psd = np.percentile(self.psd_array, 10, axis=0)
        p90_psd = np.percentile(self.psd_array, 90, axis=0)

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot average with shaded percentile region
        ax.plot(self.frequency_ghz, avg_psd, 'b-', linewidth=1.5, label='Average')
        ax.fill_between(self.frequency_ghz, p10_psd, p90_psd,
                        alpha=0.3, color='blue', label='10th-90th Percentile')

        # Labels and title
        ax.set_xlabel('Frequency (GHz)', fontsize=12, fontweight='bold')
        ax.set_ylabel('PSD (dBm/Hz)', fontsize=12, fontweight='bold')
        ax.set_title('Average Spectrum (24-Hour Mean)', fontsize=14, fontweight='bold')

        # Grid and legend
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=10)

        # Tight layout
        plt.tight_layout()

        # Save
        output_file = os.path.join(self.output_dir, 'average_spectrum.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved to {output_file}")

        plt.close()

    def create_snapshot_comparison(self, time_indices: Optional[List[int]] = None,
                                  num_snapshots: int = 6):
        """Create comparison of PSD snapshots at different times.

        Args:
            time_indices: Specific time indices to plot (auto-select if None)
            num_snapshots: Number of snapshots to show if auto-selecting
        """
        print(f"\nCreating snapshot comparison...")

        # Auto-select interesting times if not specified
        if time_indices is None:
            time_indices = self._select_interesting_times(num_snapshots)

        print(f"  Plotting {len(time_indices)} snapshots")

        # Create figure with subplots
        n_rows = (len(time_indices) + 1) // 2  # 2 columns
        n_cols = min(2, len(time_indices))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3*n_rows))

        # Handle single subplot case
        if len(time_indices) == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Plot each snapshot
        for i, time_idx in enumerate(time_indices):
            ax = axes[i]
            time_hr = self.time_hours[time_idx]
            psd = self.psd_array[time_idx, :]

            ax.plot(self.frequency_ghz, psd, 'b-', linewidth=1.0)
            ax.set_xlabel('Frequency (GHz)', fontsize=10)
            ax.set_ylabel('PSD (dBm/Hz)', fontsize=10)
            ax.set_title(f't = {time_hr:.1f} hours', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')

        # Hide unused subplots
        for i in range(len(time_indices), len(axes)):
            axes[i].set_visible(False)

        # Overall title
        fig.suptitle('PSD Snapshots at Selected Times', fontsize=14, fontweight='bold', y=1.00)

        # Tight layout
        plt.tight_layout()

        # Save
        output_file = os.path.join(self.output_dir, 'snapshot_comparison.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved to {output_file}")

        plt.close()

    def create_activity_timeline(self, activity_log: Optional[List] = None):
        """Create timeline showing carrier and interferer activity.

        Args:
            activity_log: Activity log from simulation (optional)
        """
        print(f"\nCreating activity timeline...")

        # If no activity log provided, just show counts from PSD statistics
        # Calculate activity metrics from PSD
        # (This is a simplified version - ideally would use actual activity log)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        if activity_log is not None:
            # Use actual activity log
            times = np.array([a['time_min'] for a in activity_log]) / 60.0
            num_carriers = np.array([a['num_carriers'] for a in activity_log])
            num_interferers = np.array([a['num_interferers'] for a in activity_log])

            # Plot carrier activity
            ax1.plot(times, num_carriers, 'b-', linewidth=2, label='Active Carriers')
            ax1.fill_between(times, 0, num_carriers, alpha=0.3, color='blue')
            ax1.set_ylabel('Number of Active Carriers', fontsize=12, fontweight='bold')
            ax1.set_title('Carrier Activity Over Time', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.legend(loc='upper right')

            # Plot interferer activity
            ax2.plot(times, num_interferers, 'r-', linewidth=2, label='Active Interferers')
            ax2.fill_between(times, 0, num_interferers, alpha=0.3, color='red')
            ax2.set_ylabel('Number of Active Interferers', fontsize=12, fontweight='bold')
            ax2.set_title('Interferer Activity Over Time', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.legend(loc='upper right')

        else:
            # Simplified version using PSD statistics
            # Calculate total power as proxy for activity
            total_power = np.array([np.trapz(10**(psd/10), self.frequency_array)
                                   for psd in self.psd_array])

            ax1.plot(self.time_hours, total_power / 1e-9, 'b-', linewidth=2)
            ax1.set_ylabel('Total Power (nW)', fontsize=12, fontweight='bold')
            ax1.set_title('Total Power Over Time', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3, linestyle='--')

            # Calculate variance as proxy for dynamics
            psd_variance = np.var(self.psd_array, axis=1)
            ax2.plot(self.time_hours, psd_variance, 'r-', linewidth=2)
            ax2.set_ylabel('PSD Variance (dB²)', fontsize=12, fontweight='bold')
            ax2.set_title('Spectrum Variability Over Time', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        # Save
        output_file = os.path.join(self.output_dir, 'activity_timeline.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved to {output_file}")

        plt.close()

    def _select_interesting_times(self, num_times: int = 6) -> List[int]:
        """Auto-select interesting time points based on activity changes.

        Args:
            num_times: Number of times to select

        Returns:
            List of time indices
        """
        # Strategy: Find times with significant changes in spectrum
        # Calculate difference between consecutive time steps
        psd_diff = np.diff(self.psd_array, axis=0)
        change_magnitude = np.sum(np.abs(psd_diff), axis=1)

        # Always include first and last
        selected = [0, len(self.time_array) - 1]

        # Find times with largest changes
        num_middle = num_times - 2
        if num_middle > 0:
            # Get indices of largest changes (excluding first/last)
            change_indices = np.argsort(change_magnitude)[::-1]

            # Select top N, avoiding times too close to first/last
            for idx in change_indices:
                if len(selected) >= num_times:
                    break
                # Adjust index by 1 (since diff reduces array by 1)
                actual_idx = idx + 1
                # Check not too close to boundaries
                if actual_idx > 2 and actual_idx < len(self.time_array) - 2:
                    selected.append(actual_idx)

        # Sort and return
        selected = sorted(selected)

        print(f"  Auto-selected times: {[f'{self.time_hours[i]:.1f}h' for i in selected]}")

        return selected

    def create_animated_spectrogram(self, fps: int = 15, figsize: Tuple[float, float] = (19.2, 10.8),
                                   frame_decimation: int = 1, output_format: str = 'gif',
                                   interferer_data: Optional[List[List[float]]] = None):
        """Create animated GIF or MP4 showing PSD evolution over time.

        Creates an animated visualization with:
        - Top 1/3: Current PSD line plot with translucent red spans for interferers
        - Bottom 2/3: Spectrogram with white line marking current time

        Args:
            fps: Frames per second (default: 15)
            figsize: Figure size in inches (width, height). Default (19.2, 10.8) = 1920x1080 at 100 DPI
            frame_decimation: Frame decimation factor. Use every Nth frame (default: 1, no decimation)
            output_format: Output format, either 'gif' or 'mp4' (default: 'gif')
            interferer_data: Optional list of interferer frequencies (Hz) for each snapshot.
                            Each entry is a list of frequencies where interferers are present.
                            Example: [[12.2e9, 12.3e9], [12.2e9], ...]
        """
        # Validate output format
        output_format = output_format.lower()
        if output_format not in ['gif', 'mp4']:
            raise ValueError(f"output_format must be 'gif' or 'mp4', got '{output_format}'")

        print(f"\nCreating animated spectrogram ({output_format.upper()})...")
        print(f"  Frame rate: {fps} fps")
        print(f"  Figure size: {figsize[0]:.1f}x{figsize[1]:.1f} inches")
        print(f"  Frame decimation: {frame_decimation}")
        print(f"  Output format: {output_format}")

        # Calculate frame indices with decimation
        frame_indices = list(range(0, len(self.time_array), frame_decimation))
        print(f"  Total frames: {len(frame_indices)} (of {len(self.time_array)} snapshots)")

        try:
            import imageio
        except ImportError:
            print("  ERROR: imageio not installed. Install with: pip install 'imageio[ffmpeg]'")
            return

        # Use non-interactive backend for rendering
        import matplotlib
        original_backend = matplotlib.get_backend()
        matplotlib.use('Agg')

        # Auto-scale PSD colormap
        vmin = np.percentile(self.psd_array, 10)
        vmax = np.percentile(self.psd_array, 99)

        # Create frames
        frames = []

        try:
            for i, frame_idx in enumerate(frame_indices):
                if i % 10 == 0:
                    print(f"  Rendering frame {i + 1}/{len(frame_indices)} ({100*(i+1)/len(frame_indices):.0f}%)")

                # Create figure with custom size
                fig = plt.figure(figsize=figsize, dpi=100)

                # Create grid: top 1/3 for PSD, bottom 2/3 for spectrogram
                gs = fig.add_gridspec(3, 1, hspace=0.3)
                ax_psd = fig.add_subplot(gs[0, 0])
                ax_spec = fig.add_subplot(gs[1:, 0])

                # Current time
                current_time_hr = self.time_hours[frame_idx]

                # Top plot: Current PSD
                psd_current = self.psd_array[frame_idx, :]
                ax_psd.plot(self.frequency_ghz, psd_current, 'b-', linewidth=1.5)
                ax_psd.set_ylabel('PSD (dBm/Hz)', fontsize=10, fontweight='bold')
                ax_psd.set_xlim(self.frequency_ghz[0], self.frequency_ghz[-1])
                ax_psd.grid(True, alpha=0.3, linestyle='--')
                ax_psd.set_ylim(vmin - 5, vmax + 5)  # Fixed y-axis for consistency

                # Overlay interferer spans if data provided
                if interferer_data is not None and frame_idx < len(interferer_data):
                    interferer_freqs = interferer_data[frame_idx]
                    if interferer_freqs:
                        # Calculate span width: make CW interferers visible (0.5% of bandwidth or 5 MHz minimum)
                        total_bandwidth_ghz = self.frequency_ghz[-1] - self.frequency_ghz[0]
                        span_width_ghz = max(0.005, total_bandwidth_ghz * 0.005)  # 5 MHz or 0.5% of BW

                        for interferer_freq_hz in interferer_freqs:
                            interferer_freq_ghz = interferer_freq_hz / 1e9
                            # Draw translucent red span centered on interferer
                            ax_psd.axvspan(
                                interferer_freq_ghz - span_width_ghz / 2,
                                interferer_freq_ghz + span_width_ghz / 2,
                                color='red',
                                alpha=0.3,
                                linewidth=0,
                                zorder=10
                            )

                # Bottom plot: Full spectrogram with current time marker
                X, Y = np.meshgrid(self.frequency_ghz, self.time_hours)
                im = ax_spec.pcolormesh(X, Y, self.psd_array, shading='auto',
                                       cmap='viridis', vmin=vmin, vmax=vmax)

                # White horizontal line at current time
                ax_spec.axhline(y=current_time_hr, color='white', linewidth=2, linestyle='-')

                ax_spec.set_xlabel('Frequency (GHz)', fontsize=10, fontweight='bold')
                ax_spec.set_ylabel('Time (hours)', fontsize=10, fontweight='bold')
                ax_spec.set_xlim(self.frequency_ghz[0], self.frequency_ghz[-1])
                ax_spec.set_ylim(self.time_hours[0], self.time_hours[-1])

                # Overall title with current time
                fig.suptitle(f'RF Pattern of Life - t={current_time_hr:.1f} hours',
                            fontsize=16, fontweight='bold')

                # Render to buffer
                fig.canvas.draw()

                # Convert to numpy array (RGB)
                buf = fig.canvas.buffer_rgba()
                image = np.asarray(buf)[:, :, :3]  # Drop alpha channel

                frames.append(image)
                plt.close(fig)

        finally:
            # Restore original backend
            matplotlib.use(original_backend)

        # Save file
        output_file = os.path.join(self.output_dir, f'animated_spectrogram.{output_format}')
        print(f"  Saving {output_format.upper()}...")

        if output_format == 'gif':
            imageio.mimsave(output_file, frames, fps=fps)
        elif output_format == 'mp4':
            # MP4 requires additional codec parameters
            imageio.mimsave(output_file, frames, fps=fps, codec='libx264', quality=8, pixelformat='yuv420p')

        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"  Saved to {output_file}")
        print(f"  File size: {file_size_mb:.1f} MB")
        print(f"  Duration: {len(frames)/fps:.1f} seconds")

    def create_all_plots(self, activity_log: Optional[List] = None,
                        interferer_data: Optional[List[List[float]]] = None):
        """Create all standard plots.

        Args:
            activity_log: Activity log from simulation (optional)
            interferer_data: Interferer frequency data for each snapshot (optional)
        """
        print(f"\n{'='*60}")
        print(f"Creating all visualizations...")
        print(f"{'='*60}")

        self.create_waterfall_plot()
        self.create_average_spectrum()
        self.create_snapshot_comparison()
        self.create_activity_timeline(activity_log)
        self.create_animated_spectrogram(interferer_data=interferer_data)

        print(f"\n{'='*60}")
        print(f"All visualizations complete!")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}")


if __name__ == "__main__":
    # Test visualization
    print("Testing visualization...\n")

    # Need to run simulation first
    from carrier_generator import CarrierGenerator
    from interferer_generator import InterfererGenerator
    from psd_simulator import PSDSimulator

    # Generate carriers
    print("Generating carriers...")
    carrier_gen = CarrierGenerator(seed=42)
    carrier_config = carrier_gen.generate_carriers(num_static_per_xpdr=(5, 8), num_dynamic=10)

    # Generate interferers
    print("\nGenerating interferers...")
    interferer_gen = InterfererGenerator(seed=43)
    interferers = interferer_gen.generate_interferers(
        carrier_configs=carrier_config.carriers,
        num_long_duration=2,
        num_short_duration=3
    )

    # Create simulator
    print("\nCreating simulator...")
    simulator = PSDSimulator(
        carrier_config=carrier_config,
        interferer_configs=interferers,
        rbw_hz=10e3,
        vbw_hz=1e3
    )

    # Run short simulation for testing (2 hours)
    print("\nRunning test simulation (2 hours)...")
    snapshots, metadata = simulator.run_simulation(
        duration_min=120,
        snapshot_interval_min=5
    )

    # Extract data for visualization
    time_arr = np.array([s.time_min for s in snapshots])
    freq_arr = snapshots[0].frequency_hz
    psd_arr = np.array([s.psd_dbm_hz for s in snapshots])

    # Create activity log
    activity_log = []
    for s in snapshots:
        activity_log.append({
            'time_min': s.time_min,
            'num_carriers': s.num_carriers,
            'num_interferers': s.num_interferers
        })

    # Create visualizer
    print("\nCreating visualizations...")
    viz = Visualizer(
        time_array=time_arr,
        frequency_array=freq_arr,
        psd_array=psd_arr,
        metadata=metadata,
        output_dir="test_plots"
    )

    # Generate all plots
    viz.create_all_plots(activity_log=activity_log)

    print(f"\nTest complete! Check 'test_plots' directory for output.")
