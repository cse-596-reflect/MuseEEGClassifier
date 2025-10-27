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
from collections import deque
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import re

try:
    import pylsl as lsl
except Exception as e:
    print(f"Error importing pylsl: {e}\nInstall it with: pip install pylsl", file=sys.stderr)
    sys.exit(1)

try:
    from sklearn.svm import SVC
    # joblib is now a separate package in newer sklearn versions
    try:
        from sklearn.externals import joblib
    except ImportError:
        import joblib
except Exception as e:
    print(f"Error importing sklearn: {e}\nInstall it with: pip install scikit-learn", file=sys.stderr)
    sys.exit(1)


class LibSVMModel:
    """Parser for libsvm model format."""
    
    def __init__(self):
        self.svm_type = None
        self.kernel_type = None
        self.degree = None
        self.gamma = None
        self.coef0 = None
        self.nr_class = None
        self.total_sv = None
        self.rho = None
        self.label = None
        self.probA = None
        self.probB = None
        self.nr_sv = None
        self.sv_coef = None
        self.SV = None
        
    def load_from_file(self, filepath):
        """Load libsvm model from file."""
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('svm_type'):
                self.svm_type = line.split()[1]
            elif line.startswith('kernel_type'):
                self.kernel_type = line.split()[1]
            elif line.startswith('degree'):
                self.degree = int(line.split()[1])
            elif line.startswith('gamma'):
                self.gamma = float(line.split()[1])
            elif line.startswith('coef0'):
                self.coef0 = float(line.split()[1])
            elif line.startswith('nr_class'):
                self.nr_class = int(line.split()[1])
            elif line.startswith('total_sv'):
                self.total_sv = int(line.split()[1])
            elif line.startswith('rho'):
                self.rho = [float(x) for x in line.split()[1:]]
            elif line.startswith('label'):
                self.label = [int(x) for x in line.split()[1:]]
            elif line.startswith('probA'):
                self.probA = [float(x) for x in line.split()[1:]]
            elif line.startswith('probB'):
                self.probB = [float(x) for x in line.split()[1:]]
            elif line.startswith('nr_sv'):
                self.nr_sv = [int(x) for x in line.split()[1:]]
            elif line.startswith('SV'):
                # Parse support vectors
                i += 1
                self.sv_coef = []
                self.SV = []
                
                while i < len(lines):
                    line = lines[i].strip()
                    if not line:
                        break
                    
                    # Parse support vector coefficients and values
                    parts = line.split()
                    coef = float(parts[0])
                    self.sv_coef.append(coef)
                    
                    sv_vector = {}
                    for part in parts[1:]:
                        if ':' in part:
                            idx, val = part.split(':')
                            sv_vector[int(idx)] = float(val)
                    
                    self.SV.append(sv_vector)
                    i += 1
                break
            i += 1
    
    def predict_probability(self, features):
        """Predict probability using the libsvm model."""
        if not self.SV or not self.sv_coef:
            return [0.5, 0.5]  # Default probabilities
        
        # Convert features to dictionary format
        feature_dict = {i: features[i] for i in range(len(features))}
        
        # Calculate kernel values and decision function
        decision_value = 0.0
        
        for i, sv in enumerate(self.SV):
            if self.kernel_type == 'rbf':
                # RBF kernel: K(x,y) = exp(-gamma * ||x-y||^2)
                kernel_val = self._rbf_kernel(feature_dict, sv)
            else:
                # Default to linear kernel for simplicity
                kernel_val = self._linear_kernel(feature_dict, sv)
            
            decision_value += self.sv_coef[i] * kernel_val
        
        # Add bias term
        if self.rho:
            decision_value -= self.rho[0]
        
        # Convert decision value to probability using sigmoid
        if self.probA and self.probB:
            prob = self._sigmoid_probability(decision_value, self.probA[0], self.probB[0])
        else:
            # Simple sigmoid
            prob = 1.0 / (1.0 + np.exp(-decision_value))
        
        return [prob, 1.0 - prob]
    
    def _rbf_kernel(self, x, y):
        """RBF kernel calculation."""
        if self.gamma is None:
            gamma = 1.0 / len(x)
        else:
            gamma = self.gamma
        
        # Calculate squared distance
        dist_sq = 0.0
        all_indices = set(x.keys()) | set(y.keys())
        
        for idx in all_indices:
            x_val = x.get(idx, 0.0)
            y_val = y.get(idx, 0.0)
            dist_sq += (x_val - y_val) ** 2
        
        return np.exp(-gamma * dist_sq)
    
    def _linear_kernel(self, x, y):
        """Linear kernel calculation."""
        dot_product = 0.0
        all_indices = set(x.keys()) | set(y.keys())
        
        for idx in all_indices:
            dot_product += x.get(idx, 0.0) * y.get(idx, 0.0)
        
        return dot_product
    
    def _sigmoid_probability(self, decision_value, probA, probB):
        """Convert decision value to probability using sigmoid."""
        f_apb = decision_value * probA + probB
        if f_apb >= 0:
            return np.exp(-f_apb) / (1.0 + np.exp(-f_apb))
        else:
            return 1.0 / (1.0 + np.exp(f_apb))


class MuseEEGClassifier:
    """
    Real-time EEG classification using SVM model from Android MuseEEGClassifier project.
    Implements the same preprocessing pipeline as the Android version.
    """
    
    def __init__(self, model_path=None):
        # Constants from Android SVMConstants.java
        self.SAMPLE_RATE = 256
        self.NUM_EEG_CH = 4
        self.WINDOW_LENGTH = 2.0  # seconds
        self.WINDOW_SIZE = int(self.SAMPLE_RATE * self.WINDOW_LENGTH)  # 512 samples
        self.OVER_LAP = 50  # percentage
        self.WINDOW_SHIFT = int(self.WINDOW_SIZE * (100 - self.OVER_LAP) / 100)  # 256 samples
        
        # Frequency bands (excluding delta as per Android implementation)
        self.BANDS = [1, 4, 8, 12, 18, 30, 45]  # Hz
        self.NUM_BAND = len(self.BANDS) - 1  # 6 bands
        self.N_START_BAND = 1  # Start from theta band (skip delta)
        
        # Pre-filter coefficients from Android implementation
        self.preFilterA = [1, -5.15782851817200, 11.5608198955593, -14.9658099132349, 
                          12.4693538123194, -6.90189476483231, 2.44985418058679, 
                          -0.502508975737991, 0.0480142849697873]
        self.preFilterB = [0.0302684886055911, 0, -0.121073954422364, 0, 0.181610931633547, 
                          0, -0.121073954422364, 0, 0.0302684886055911]
        
        # Artifact removal parameters
        self.kCompMat = [22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 48, 52]
        self.kCompMat2 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 26]
        self.nComp = len(self.kCompMat)
        
        # Bandpass filter coefficients (simplified - using scipy instead of exact Android coefficients)
        self.bandpass_filters = self._create_bandpass_filters()
        
        # Data buffers
        self.eeg_buffer = deque(maxlen=self.WINDOW_SIZE * 3)  # 3 windows as per Android
        self.raw_eeg_buffer = np.zeros((self.WINDOW_SIZE * 3, self.NUM_EEG_CH))
        self.meditation_prob_history = deque(maxlen=100)
        
        # SVM model
        self.svm_model = None
        self.load_model(model_path)
        
        # Smoothing for meditation probability
        self.meditation_smoother = SmoothingFilter(window_size=10)
        
    def _create_bandpass_filters(self):
        """Create bandpass filters for each frequency band."""
        filters = {}
        nyquist = self.SAMPLE_RATE / 2
        
        for i in range(self.N_START_BAND, len(self.BANDS) - 1):
            low = self.BANDS[i] / nyquist
            high = self.BANDS[i+1] / nyquist
            
            # Ensure frequencies are within valid range
            low = max(0.01, min(low, 0.99))
            high = max(0.01, min(high, 0.99))
            
            if low < high:
                b, a = signal.butter(4, [low, high], btype='band')
                filters[i] = (b, a)
        
        return filters
    
    def load_model(self, model_path=None):
        """Load SVM model from file."""
        if model_path is None:
            # Try to find the model file
            model_path = Path(__file__).parent.parent / "app" / "src" / "main" / "assets" / "final_svm_model.txt"
        
        if not Path(model_path).exists():
            print(f"Model file not found: {model_path}", file=sys.stderr)
            print("Please provide the correct path to final_svm_model.txt", file=sys.stderr)
            return
        
        try:
            # Load libsvm format model
            print(f"Loading libsvm model from: {model_path}")
            self.svm_model = LibSVMModel()
            self.svm_model.load_from_file(model_path)
            print("Model loaded successfully!")
            print(f"SVM Type: {self.svm_model.svm_type}")
            print(f"Kernel: {self.svm_model.kernel_type}")
            print(f"Gamma: {self.svm_model.gamma}")
            print(f"Number of classes: {self.svm_model.nr_class}")
            print(f"Total support vectors: {self.svm_model.total_sv}")
            
        except Exception as e:
            print(f"Error loading model: {e}", file=sys.stderr)
            self.svm_model = None
    
    def apply_prefilter(self, data):
        """Apply pre-filtering to remove artifacts."""
        filtered_data = data.copy()
        
        for ch in range(self.NUM_EEG_CH):
            # Apply IIR filter (simplified version of Android implementation)
            filtered_data[:, ch] = signal.filtfilt(self.preFilterB, self.preFilterA, data[:, ch])
        
        return filtered_data
    
    def artifact_removal(self, data):
        """Remove artifacts using moving average technique from Android implementation."""
        data_length = data.shape[0]
        f_ref_data = np.zeros_like(data)
        f_out_data = np.zeros_like(data)
        
        # Apply moving average artifact removal
        for j in range(self.nComp):
            k_comp = self.kCompMat[j]
            k_comp2 = self.kCompMat2[j]
            
            # Calculate reference data using moving average
            for k in range(data_length - k_comp):
                for ch in range(self.NUM_EEG_CH):
                    f_ref_data[k, ch] = np.mean(data[k:k+k_comp, ch])
            
            # Apply artifact removal
            index = k_comp2
            for i in range(data_length - k_comp2):
                for ch in range(self.NUM_EEG_CH):
                    f_out_data[i, ch] = data[index, ch] - f_ref_data[i, ch] + f_out_data[i, ch]
                index += 1
        
        # Normalize by number of components
        f_out_data = f_out_data / self.nComp
        
        return f_out_data
    
    def bandpass_filter(self, data):
        """Apply bandpass filtering to extract frequency bands."""
        filtered_data = np.zeros((data.shape[0], data.shape[1], self.NUM_BAND))
        
        for band_idx in range(self.N_START_BAND, self.NUM_BAND):
            if band_idx in self.bandpass_filters:
                b, a = self.bandpass_filters[band_idx]
                
                for ch in range(self.NUM_EEG_CH):
                    filtered_data[:, ch, band_idx] = signal.filtfilt(b, a, data[:, ch])
        
        return filtered_data
    
    def extract_features(self, data):
        """Extract features from EEG data matching Android implementation."""
        fs = self.SAMPLE_RATE
        win_len = self.WINDOW_LENGTH
        win_size = int(np.floor(win_len * fs))
        win_shift = int(np.floor(win_size * (100 - self.OVER_LAP) / 100))
        
        num_seg = int(np.floor((data.shape[0] - win_size) / win_shift)) + 1
        num_channel = data.shape[1]
        n_band = self.NUM_BAND
        
        # Apply bandpass filtering
        filtered_data = self.bandpass_filter(data)
        
        # Extract features for each segment
        features = np.zeros((num_seg, n_band * num_channel))
        
        for i_seg in range(num_seg):
            x_start = i_seg * win_shift
            x_end = x_start + win_size
            
            if x_end > data.shape[0]:
                break
            
            # Calculate relative power for each channel and band
            for i_ch in range(num_channel):
                for band in range(n_band):
                    # Calculate power in this band
                    band_power = np.sum(filtered_data[x_start:x_end, i_ch, band] ** 2)
                    features[i_seg, i_ch * n_band + band] = band_power
            
            # Normalize by total power for each channel (relative power)
            for i_ch in range(num_channel):
                total_power = np.sum(features[i_seg, i_ch * n_band:(i_ch + 1) * n_band])
                if total_power > 0:
                    features[i_seg, i_ch * n_band:(i_ch + 1) * n_band] /= total_power
        
        return features
    
    def process_eeg_data(self, eeg_data):
        """Process EEG data through the complete pipeline."""
        if self.svm_model is None:
            return None
        
        # Convert to numpy array if needed
        if isinstance(eeg_data, list):
            eeg_data = np.array(eeg_data)
        
        # Ensure we have the right shape
        if eeg_data.shape[1] != self.NUM_EEG_CH:
            print(f"Warning: Expected {self.NUM_EEG_CH} channels, got {eeg_data.shape[1]}")
            return None
        
        # Apply preprocessing pipeline
        # 1. Pre-filtering
        filtered_data = self.apply_prefilter(eeg_data)
        
        # 2. Artifact removal
        clean_data = self.artifact_removal(filtered_data)
        
        # 3. Feature extraction
        features = self.extract_features(clean_data)
        
        if len(features) == 0:
            return None
        
        # Use the middle segment for classification (as per Android implementation)
        middle_idx = len(features) // 2
        feature_vector = features[middle_idx]
        
        # Ensure we have the right number of features
        if len(feature_vector) != 24:
            print(f"Warning: Expected 24 features, got {len(feature_vector)}")
            return None
        
        # Classify
        try:
            # Get probability for meditation class (class 0)
            prob = self.svm_model.predict_probability(feature_vector)
            meditation_prob = prob[0]  # Assuming class 0 is meditation
            
            # Apply smoothing
            smoothed_prob = self.meditation_smoother.update(meditation_prob)
            
            return {
                'meditation_probability': smoothed_prob,
                'raw_probability': meditation_prob,
                'features': feature_vector,
                'timestamp': time.time()
            }
            
        except Exception as e:
            print(f"Classification error: {e}")
            return None
    
    def update_buffer(self, eeg_sample):
        """Update the EEG buffer with new sample."""
        self.eeg_buffer.append(eeg_sample)
        
        # Update raw buffer for processing
        if len(self.eeg_buffer) >= self.SAMPLE_RATE:
            # Fill the first segment
            for i in range(self.SAMPLE_RATE, self.WINDOW_SIZE):
                if len(self.eeg_buffer) > i:
                    sample = self.eeg_buffer[i]
                    for j in range(min(len(sample), self.NUM_EEG_CH)):
                        self.raw_eeg_buffer[i, j] = sample[j]
            
            # Process the middle window
            middle_start = self.WINDOW_SIZE
            middle_end = self.WINDOW_SIZE * 2
            
            if len(self.eeg_buffer) >= middle_end:
                middle_data = self.raw_eeg_buffer[middle_start:middle_end, :]
                result = self.process_eeg_data(middle_data)
                
                if result:
                    self.meditation_prob_history.append(result['meditation_probability'])
                    return result
            
            # Shift buffers
            self.raw_eeg_buffer[:self.WINDOW_SIZE*2] = self.raw_eeg_buffer[self.WINDOW_SHIFT:self.WINDOW_SIZE*2+self.WINDOW_SHIFT]
        
        return None


class SmoothingFilter:
    """Simple smoothing filter for meditation probability."""
    
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
    
    def update(self, value):
        self.buffer.append(value)
        return np.mean(self.buffer)


def get_muse_stream_by_type(stream_type: str, timeout: float = 8.0, source_id: str | None = None):
    """Get Muse LSL stream by type."""
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


def extract_eeg_labels(info):
    """Extract EEG channel labels."""
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
        return [f"ch{i+1}" for i in range(n)]
    except Exception:
        n = info.channel_count()
        muse_defaults = ['TP9', 'AF7', 'AF8', 'TP10', 'Right AUX']
        if n <= len(muse_defaults):
            return muse_defaults[:n]
        return [f"ch{i+1}" for i in range(n)]


def discover_muse_mac(timeout: float = 10.0):
    """Discover Muse device via Bluetooth."""
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


def play_bell():
    """Play notification bell."""
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


def main():
    """Main function for real-time EEG classification."""
    print("Muse EEG Real-time Classifier")
    print("=============================")
    
    # Initialize classifier
    classifier = MuseEEGClassifier()
    
    if classifier.svm_model is None:
        print("Failed to load SVM model. Exiting.")
        return
    
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

        print("Starting muselsl stream (EEG) using discovered MAC...")
        muselsl_cmd = [
            sys.executable, "-m", "muselsl", "stream",
            "--address", muse_mac
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

    print("Found Muse EEG stream.")
    try:
        print(f"EEG: {eeg_info.channel_count()} ch @ {eeg_info.nominal_srate()} Hz")
    except Exception:
        pass

    # Setup LSL inlet
    inlet = lsl.StreamInlet(eeg_info, max_buflen=60, processing_flags=0)
    n_eeg = min(eeg_info.channel_count(), classifier.NUM_EEG_CH)
    eeg_labels = extract_eeg_labels(eeg_info)

    print(f"\nStarting real-time classification...")
    print(f"Channels: {eeg_labels[:n_eeg]}")
    print("Press Ctrl+C to stop")
    print("=" * 50)

    # Real-time classification loop
    try:
        while True:
            samples, timestamps = inlet.pull_chunk(timeout=0.1)
            
            if timestamps:
                for sample in samples:
                    # Take only the first 4 channels (EEG channels)
                    eeg_sample = sample[:n_eeg]
                    
                    # Process the sample
                    result = classifier.update_buffer(eeg_sample)
                    
                    if result:
                        meditation_prob = result['meditation_probability']
                        raw_prob = result['raw_probability']
                        
                        # Display results
                        print(f"\rMeditation Level: {meditation_prob:.3f} ({meditation_prob*100:.1f}%) | Raw: {raw_prob:.3f}", end='', flush=True)
            
            time.sleep(0.01)  # Small delay to prevent excessive CPU usage
            
    except KeyboardInterrupt:
        print("\n\nStopping classification...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
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

    print("Classification stopped.")


if __name__ == "__main__":
    main()