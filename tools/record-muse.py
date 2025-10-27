#!/usr/bin/env python3

import json
import csv
import sys
import time
import threading
from pathlib import Path
import subprocess
import asyncio
from datetime import datetime
import select
import tty
import termios

try:
    import pylsl as lsl
except Exception as e:
    print(f"Error importing pylsl: {e}\nInstall it with: pip install pylsl", file=sys.stderr)
    sys.exit(1)


def play_bell():
    try:
        subprocess.run(
            ["paplay", "/usr/share/sounds/freedesktop/stereo/bell.oga"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2
        )
    except Exception:
        try:
            subprocess.run(["beep", "-f", "800", "-l", "500"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
        except Exception:
            print("\a", flush=True)


def get_muse_stream_by_type(stream_type: str, timeout: float = 8.0, source_id: str | None = None):
    streams = lsl.resolve_byprop('type', stream_type, timeout=timeout)
    if not streams:
        return None
    muse_streams = [s for s in streams if 'muse' in (s.name() or '').lower()]
    if not muse_streams:
        return None
    if source_id is None:
        return muse_streams[0]
    for s in muse_streams:
        try:
            if s.source_id() and s.source_id() == source_id:
                return s
        except Exception:
            pass
    return muse_streams[0]


def default_labels(n: int) -> list[str]:
    return [f"ch{i+1}" for i in range(n)]

def extract_eeg_labels(info) -> list[str]:
    try:
        n = info.channel_count()
        desc = info.desc()
        chs = desc.child('channels').child('channel')
        labels = []
        while not chs.empty():
            lab = chs.child_value('label') or chs.child_value('name')
            labels.append(lab if lab else f'ch{len(labels)+1}')
            chs = chs.next_sibling()
        if labels and len(labels) == n:
            return labels
        muse_defaults = ['TP9', 'AF7', 'AF8', 'TP10', 'Right AUX']
        if n <= len(muse_defaults):
            return muse_defaults[:n]
        return default_labels(n)
    except Exception:
        n = info.channel_count()
        muse_defaults = ['TP9', 'AF7', 'AF8', 'TP10', 'Right AUX']
        if n <= len(muse_defaults):
            return muse_defaults[:n]
        return default_labels(n)

def discover_muse_mac(timeout: float = 10.0):
    try:
        from bleak import BleakScanner
    except Exception:
        print("Error: bleak is required for Bluetooth discovery. Install with: pip install bleak", file=sys.stderr)
        sys.exit(2)

    try:
        devices = asyncio.run(BleakScanner.discover(timeout=timeout))
    except Exception as e:
        print(f"Bluetooth scan failed: {e}", file=sys.stderr)
        sys.exit(2)

    for d in devices:
        try:
            name = (d.name or "")
            if name and 'muse' in name.lower():
                return d.address, name
        except Exception:
            continue
    return None, None


def get_next_session_number(dataset_dir, prefix):
    existing = list(dataset_dir.glob(f"{prefix}_*"))
    numbers = []
    for folder in existing:
        try:
            num_str = folder.name.split('_')[-1]
            numbers.append(int(num_str))
        except (ValueError, IndexError):
            continue
    return max(numbers) + 1 if numbers else 1


def check_channel_quality(eeg_info):
    from scipy import signal
    import numpy as np
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    n_channels = eeg_info.channel_count()
    labels = extract_eeg_labels(eeg_info)
    fs = eeg_info.nominal_srate()
    if fs <= 0:
        fs = 256

    display_channels = []
    for ch_idx in range(n_channels):
        ch_label = labels[ch_idx] if ch_idx < len(labels) else f"ch{ch_idx+1}"
        if 'aux' not in ch_label.lower():
            display_channels.append((ch_idx, ch_label))

    inlet = lsl.StreamInlet(eeg_info, max_buflen=10, processing_flags=0)

    stop_flag = threading.Event()

    def wait_for_enter():
        input()
        stop_flag.set()

    input_thread = threading.Thread(target=wait_for_enter, daemon=True)
    input_thread.start()

    print("\nChecking channel quality (Press Enter when ready)...\n")

    buffer_duration = 1.0
    buffer_size = int(fs * buffer_duration)
    plot_duration = 5.0
    plot_size = int(fs * plot_duration)
    channel_buffers = [[] for _ in range(n_channels)]
    plot_buffers = [[] for _ in range(n_channels)]

    n_display = len(display_channels)
    fig, axes = plt.subplots(n_display, 1, figsize=(10, 8), sharex=True)
    if n_display == 1:
        axes = [axes]

    lines = []
    for ax, (ch_idx, ch_label) in zip(axes, display_channels):
        line, = ax.plot([], [], linewidth=0.5)
        ax.set_ylabel(ch_label)
        ax.set_ylim(-500, 500)
        ax.grid(True, alpha=0.3)
        lines.append(line)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Live EEG Signal - Press Enter in terminal when ready')
    plt.tight_layout()
    plt.ion()
    plt.show()

    header_printed = False

    try:
        while not stop_flag.is_set():
            samples, timestamps = inlet.pull_chunk(timeout=0.1)

            if timestamps:
                for sample in samples:
                    for ch_idx in range(min(n_channels, len(sample))):
                        channel_buffers[ch_idx].append(sample[ch_idx])
                        plot_buffers[ch_idx].append(sample[ch_idx])
                        if len(channel_buffers[ch_idx]) > buffer_size:
                            channel_buffers[ch_idx].pop(0)
                        if len(plot_buffers[ch_idx]) > plot_size:
                            plot_buffers[ch_idx].pop(0)

            for idx, (ch_idx, ch_label) in enumerate(display_channels):
                if len(plot_buffers[ch_idx]) > 0:
                    t = np.arange(len(plot_buffers[ch_idx])) / fs
                    lines[idx].set_data(t, plot_buffers[ch_idx])
                    axes[idx].set_xlim(0, max(5.0, t[-1]))

            fig.canvas.draw()
            fig.canvas.flush_events()

            if not header_printed:
                print(f"{'Channel':<10} {'Noise Ratio':<15} {'Status':<10}")
                print("-" * 40)
                header_printed = True
            else:
                print(f"\033[{n_display}A", end='')

            for ch_idx, ch_label in display_channels:
                if len(channel_buffers[ch_idx]) < buffer_size // 2:
                    print(f"\033[K{ch_label:<10} {'Collecting...':<15} {'':<10}")
                    continue

                data = np.array(channel_buffers[ch_idx])

                try:
                    freqs, psd = signal.welch(data, fs=fs, nperseg=min(256, len(data)))

                    signal_mask = (freqs >= 0.5) & (freqs <= 50)
                    noise_mask = freqs > 50

                    signal_power = np.sum(psd[signal_mask])
                    noise_power = np.sum(psd[noise_mask])

                    if signal_power > 0:
                        ratio = noise_power / signal_power
                    else:
                        ratio = float('inf')

                    status = "Ready" if ratio < 1 else "Not Ready"

                    print(f"\033[K{ch_label:<10} {ratio:<15.3f} {status:<10}")
                except Exception:
                    print(f"\033[K{ch_label:<10} {'Error':<15} {'':<10}")

            time.sleep(0.1)

    finally:
        plt.close(fig)
        inlet.close_stream()

    print()


def main():
    duration = 1200.0

    dataset_dir = Path(__file__).resolve().parent
    session_num = get_next_session_number(dataset_dir, "baseline")
    session_name = f"baseline_{session_num}"

    print("Checking for existing Muse LSL stream...")
    eeg_info = get_muse_stream_by_type('EEG', timeout=2.0)
    muselsl_proc = None
    muse_mac = None
    muse_name = None

    if eeg_info is None:
        print("No existing LSL stream found. Scanning Bluetooth for Muse device...")
        muse_mac, muse_name = discover_muse_mac(timeout=12.0)
        if not muse_mac:
            print("No Muse device found over Bluetooth. Ensure the headset is on and in range.", file=sys.stderr)
            sys.exit(2)
        print(f"Found Muse device: {muse_name or 'Muse'} [{muse_mac}]")

        print("Starting muselsl stream (EEG+ACC+PPG) using discovered MAC...")
        muselsl_cmd = [
            sys.executable, "-m", "muselsl", "stream",
            "--address", muse_mac,
            "--acc", "--ppg"
        ]
        try:
            muselsl_proc = subprocess.Popen(
                muselsl_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"Failed to start muselsl stream: {e}", file=sys.stderr)
            sys.exit(2)

        print("Waiting for Muse EEG LSL stream...")
        eeg_info = None
        start_wait = time.time()
        while time.time() - start_wait < 25.0:
            eeg_info = get_muse_stream_by_type('EEG', timeout=1.5)
            if eeg_info is not None:
                break
        if eeg_info is None:
            print("Could not detect EEG LSL stream after starting muselsl.", file=sys.stderr)
            if muselsl_proc:
                try:
                    muselsl_proc.terminate()
                except Exception:
                    pass
            sys.exit(2)
    else:
        print("Found existing Muse LSL stream. Using it.")

    src_id = ''
    try:
        src_id = eeg_info.source_id() or ''
    except Exception:
        src_id = ''

    print("Waiting for Muse ACC and PPG streams...")
    acc_info = None
    ppg_info = None
    wait2_start = time.time()
    while time.time() - wait2_start < 20.0 and (acc_info is None or ppg_info is None):
        if acc_info is None:
            acc_info = get_muse_stream_by_type('ACC', timeout=1.0, source_id=src_id)
        if ppg_info is None:
            ppg_info = get_muse_stream_by_type('PPG', timeout=1.0, source_id=src_id)

    if acc_info is None or ppg_info is None:
        missing = []
        if acc_info is None:
            missing.append('ACC')
        if ppg_info is None:
            missing.append('PPG')
        print(f"Missing Muse streams: {', '.join(missing)}.", file=sys.stderr)
        try:
            muselsl_proc.terminate()
        except Exception:
            pass
        sys.exit(2)

    print("Found Muse EEG, ACC, and PPG streams.")
    try:
        print(f"EEG: {eeg_info.channel_count()} ch @ {eeg_info.nominal_srate()} Hz | ACC: {acc_info.channel_count()} ch @ {acc_info.nominal_srate()} Hz | PPG: {ppg_info.channel_count()} ch @ {ppg_info.nominal_srate()} Hz")
    except Exception:
        pass

    check_channel_quality(eeg_info)

    out_dir = dataset_dir / session_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{session_name}.csv"
    metadata_path = out_dir / f"{session_name}_metadata.json"

    input("Press Enter to start 20-minute baseline meditation recording...")

    session_start_time = datetime.now().isoformat()

    print("Playing start bell...")
    play_bell()
    time.sleep(1)

    eeg_inlet = lsl.StreamInlet(eeg_info, max_buflen=max(60, int(duration) + 5), processing_flags=0)
    acc_inlet = lsl.StreamInlet(acc_info, max_buflen=max(60, int(duration) + 5), processing_flags=0)
    ppg_inlet = lsl.StreamInlet(ppg_info, max_buflen=max(60, int(duration) + 5), processing_flags=0)

    n_eeg = eeg_info.channel_count()
    eeg_labels = extract_eeg_labels(eeg_info)
    n_acc = min(3, acc_info.channel_count())
    n_ppg = min(3, ppg_info.channel_count())

    header = ["timestamp", "stream"] + eeg_labels \
             + ["acc_x", "acc_y", "acc_z"][:n_acc] \
             + ["ppg1", "ppg2", "ppg3"][:n_ppg]

    rows = []
    eeg_ts, eeg_vals = [], [[] for _ in range(n_eeg)]
    awareness_marks = []

    print(f"Recording baseline meditation for {duration:.0f}s... (Ctrl+C to cancel)")
    print("Press SPACEBAR when you notice you were thinking and return to meditation")
    print("[Playing start bell]")
    play_bell()
    time.sleep(0.5)

    start = time.time()

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while (time.time() - start) < duration:
            if select.select([sys.stdin], [], [], 0)[0]:
                char = sys.stdin.read(1)
                if char == ' ':
                    elapsed = time.time() - start
                    awareness_marks.append(elapsed)
                    print(f"\n[Awareness mark at {elapsed:.1f}s]", flush=True)

            eeg_samples, eeg_timestamps = eeg_inlet.pull_chunk(timeout=0.0)
            if eeg_timestamps:
                for ts, sample in zip(eeg_timestamps, eeg_samples):
                    sample = list(sample)[:n_eeg]
                    row = [ts, 'EEG'] + sample + ([""] * (len(header) - 2 - len(sample)))
                    rows.append(row)
                    eeg_ts.append(ts)
                    for i, v in enumerate(sample):
                        eeg_vals[i].append(v)

            acc_samples, acc_timestamps = acc_inlet.pull_chunk(timeout=0.0)
            if acc_timestamps:
                for ts, sample in zip(acc_timestamps, acc_samples):
                    sample = list(sample)[:n_acc]
                    row = [ts, 'ACC'] + ([""] * n_eeg) + sample + ([""] * (len(header) - 2 - n_eeg - len(sample)))
                    rows.append(row)

            ppg_samples, ppg_timestamps = ppg_inlet.pull_chunk(timeout=0.0)
            if ppg_timestamps:
                for ts, sample in zip(ppg_timestamps, ppg_samples):
                    sample = list(sample)[:n_ppg]
                    row = [ts, 'PPG'] + ([""] * (n_eeg + n_acc)) + sample + ([""] * (len(header) - 2 - n_eeg - n_acc - len(sample)))
                    rows.append(row)

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nRecording interrupted by user.")
    except Exception as e:
        print(f"\nLSL error: {e}")
    else:
        print("\n[Recording complete - Playing end bell]")
        play_bell()
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        for inlet in (eeg_inlet, acc_inlet, ppg_inlet):
            try:
                inlet.close_stream()
            except Exception:
                pass
        if muselsl_proc:
            try:
                muselsl_proc.terminate()
                muselsl_proc.wait(timeout=5)
            except Exception:
                try:
                    muselsl_proc.kill()
                except Exception:
                    pass

    print("Playing end bell...")
    play_bell()

    if not rows:
        print("No samples captured. Is the Muse streaming?", file=sys.stderr)
        sys.exit(3)

    rows.sort(key=lambda r: r[0])

    try:
        with out_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in rows:
                writer.writerow(row)
    except Exception as e:
        print(f"Failed to write CSV: {e}", file=sys.stderr)
        sys.exit(4)

    print(f"Saved {len(rows)} rows to {out_path}")

    metadata = {
        "session_type": "baseline",
        "session_name": session_name,
        "start_time": session_start_time,
        "duration_seconds": duration,
        "samples_collected": len(rows),
        "awareness_marks": awareness_marks,
        "muse_device": muse_name or "Muse",
        "muse_mac": muse_mac
    }

    try:
        with metadata_path.open('w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")
    except Exception as e:
        print(f"Failed to write metadata: {e}", file=sys.stderr)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import signal
        from scipy.ndimage import gaussian_filter1d

        if not eeg_ts or not any(len(ch) for ch in eeg_vals):
            print("No EEG data for plotting")
        else:
            t0 = eeg_ts[0]
            t_eeg = [t - t0 for t in eeg_ts]

            raw_plot_path = out_dir / f"{session_name}_raw_eeg.png"
            nrows = max(1, n_eeg)
            height = max(4, 2.2 * nrows)
            fig_raw, axes_raw = plt.subplots(nrows, 1, sharex=True, figsize=(12, height), constrained_layout=True)
            if nrows == 1:
                axes_raw = [axes_raw]
            for i, series in enumerate(eeg_vals):
                ax = axes_raw[i]
                if series:
                    ax.plot(t_eeg, series, linewidth=0.8, color='tab:blue')

                    for mark_idx, mark_time in enumerate(awareness_marks):
                        ax.axvline(mark_time, color='purple', linestyle=':', linewidth=0.8, alpha=0.5,
                                  label='Awareness' if i == 0 and mark_idx == 0 else '')

                ch_name = eeg_labels[i] if i < len(eeg_labels) else f"ch{i+1}"
                ax.set_title(f"EEG {ch_name}")
                ax.set_ylabel("uV")
                if i == 0 and awareness_marks:
                    ax.legend(loc='upper right', fontsize=7)
            axes_raw[-1].set_xlabel("Time (s)")
            fig_raw.suptitle(f"Raw EEG: {session_name}")
            fig_raw.savefig(raw_plot_path, dpi=150)
            plt.close(fig_raw)
            print(f"Saved raw EEG plot to {raw_plot_path}")

            band_power_plot_path = out_dir / f"{session_name}_band_power.png"
            fig_bp, ax_bp = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)

            bands = {
                'Delta': (0.5, 4),
                'Theta': (4, 8),
                'Alpha': (8, 13),
                'Beta': (13, 30),
                'Gamma': (30, 50)
            }
            colors = ['purple', 'blue', 'green', 'orange', 'red']

            fs = len(eeg_ts) / (t_eeg[-1] - t_eeg[0]) if len(t_eeg) > 1 else 256
            window_sec = 1.0
            hop_sec = 0.125
            window_samples = int(fs * window_sec)
            hop_samples = int(fs * hop_sec)

            band_powers = {band: [] for band in bands.keys()}
            time_points = []

            if window_samples > 0 and len(eeg_ts) >= window_samples:
                for start_idx in range(0, len(eeg_ts) - window_samples + 1, hop_samples):
                    end_idx = start_idx + window_samples
                    window_center_time = t_eeg[start_idx + window_samples // 2]
                    time_points.append(window_center_time)

                    channel_band_powers = {band: [] for band in bands.keys()}

                    for series in eeg_vals:
                        if not series or len(series) < end_idx:
                            continue
                        window_data = np.array(series[start_idx:end_idx])

                        freqs, psd = signal.welch(window_data, fs=fs, nperseg=min(256, len(window_data)))

                        for band_name, (low, high) in bands.items():
                            mask = (freqs >= low) & (freqs <= high)
                            band_power = np.sum(psd[mask])
                            if band_power > 0:
                                channel_band_powers[band_name].append(band_power)

                    for band_name in bands.keys():
                        if channel_band_powers[band_name]:
                            avg_power = np.mean(channel_band_powers[band_name])
                            band_powers[band_name].append(avg_power)
                        else:
                            band_powers[band_name].append(np.nan)

                for band_name in bands.keys():
                    powers = np.array(band_powers[band_name])
                    if len(powers) > 11:
                        band_powers[band_name] = gaussian_filter1d(powers, sigma=3)

                for (band_name, _), color in zip(bands.items(), colors):
                    powers = band_powers[band_name]
                    if len(powers) > 0 and not all(np.isnan(powers)):
                        powers_db = [10 * np.log10(p) if p > 1e-12 else np.nan for p in powers]
                        ax_bp.plot(time_points, powers_db, color=color, linewidth=2.5, alpha=0.8, label=band_name)

            for mark_idx, mark_time in enumerate(awareness_marks):
                ax_bp.axvline(mark_time, color='purple', linestyle=':', linewidth=1.0, alpha=0.6,
                             label='Awareness' if mark_idx == 0 else '')

            ax_bp.set_xlabel('Time (s)')
            ax_bp.set_ylabel('Band Power (dB)')
            ax_bp.set_title(f'Frequency Band Power Over Time: {session_name}')
            ax_bp.legend(loc='upper right')
            ax_bp.grid(True, alpha=0.3)
            fig_bp.savefig(band_power_plot_path, dpi=150)
            plt.close(fig_bp)
            print(f"Saved band power plot to {band_power_plot_path}")

    except Exception as e:
        print(f"Plotting failed: {e}")

    print("Done.")


if __name__ == "__main__":
    main()