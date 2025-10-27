#!/usr/bin/env python3

"""
Demo script for Muse EEG Classifier without external dependencies.
This demonstrates the core preprocessing pipeline using only numpy.
"""

import numpy as np
from pathlib import Path
import sys

# Add the current directory to path
sys.path.append(str(Path(__file__).parent))


class SimpleEEGClassifier:
    """Simplified EEG classifier for demonstration."""
    
    def __init__(self):
        # Constants from Android implementation
        self.SAMPLE_RATE = 256
        self.NUM_EEG_CH = 4
        self.WINDOW_LENGTH = 2.0
        self.WINDOW_SIZE = int(self.SAMPLE_RATE * self.WINDOW_LENGTH)
        self.OVER_LAP = 50
        self.WINDOW_SHIFT = int(self.WINDOW_SIZE * (100 - self.OVER_LAP) / 100)
        
        # Frequency bands
        self.BANDS = [1, 4, 8, 12, 18, 30, 45]
        self.NUM_BAND = len(self.BANDS) - 1
        self.N_START_BAND = 1
        
        print(f"Initialized classifier:")
        print(f"  Sample rate: {self.SAMPLE_RATE} Hz")
        print(f"  Window size: {self.WINDOW_SIZE} samples ({self.WINDOW_LENGTH}s)")
        print(f"  Window shift: {self.WINDOW_SHIFT} samples")
        print(f"  Number of bands: {self.NUM_BAND}")
        print(f"  Expected features: {self.NUM_EEG_CH * self.NUM_BAND}")
    
    def simple_bandpass_filter(self, data, low_freq, high_freq):
        """Simple bandpass filter using FFT."""
        # Convert to frequency domain
        fft_data = np.fft.fft(data, axis=0)
        freqs = np.fft.fftfreq(data.shape[0], 1/self.SAMPLE_RATE)
        
        # Create frequency mask
        mask = (np.abs(freqs) >= low_freq) & (np.abs(freqs) <= high_freq)
        
        # Apply filter
        fft_data[~mask] = 0
        
        # Convert back to time domain
        filtered_data = np.real(np.fft.ifft(fft_data, axis=0))
        
        return filtered_data
    
    def extract_features_simple(self, data):
        """Simplified feature extraction."""
        features = []
        
        for ch in range(self.NUM_EEG_CH):
            channel_data = data[:, ch]
            
            for band_idx in range(self.N_START_BAND, len(self.BANDS) - 1):
                low_freq = self.BANDS[band_idx]
                high_freq = self.BANDS[band_idx + 1]
                
                # Apply bandpass filter
                filtered_channel = self.simple_bandpass_filter(
                    channel_data.reshape(-1, 1), low_freq, high_freq
                ).flatten()
                
                # Calculate power
                power = np.mean(filtered_channel ** 2)
                features.append(power)
        
        # Normalize features (relative power)
        features = np.array(features)
        total_power = np.sum(features)
        if total_power > 0:
            features = features / total_power
        
        return features
    
    def process_eeg_data(self, eeg_data):
        """Process EEG data and extract features."""
        if eeg_data.shape[1] != self.NUM_EEG_CH:
            print(f"Warning: Expected {self.NUM_EEG_CH} channels, got {eeg_data.shape[1]}")
            return None
        
        if eeg_data.shape[0] < self.WINDOW_SIZE:
            print(f"Warning: Need at least {self.WINDOW_SIZE} samples, got {eeg_data.shape[0]}")
            return None
        
        # Extract features
        features = self.extract_features_simple(eeg_data)
        
        # Simulate classification (dummy model)
        # In real implementation, this would use the SVM model
        meditation_prob = self.simulate_classification(features)
        
        return {
            'meditation_probability': meditation_prob,
            'features': features,
            'feature_count': len(features)
        }
    
    def simulate_classification(self, features):
        """Simulate SVM classification with dummy logic."""
        # Simple heuristic: higher alpha power = more meditation
        # This is just for demonstration - real model would be more complex
        
        if len(features) < 24:
            return 0.5  # Default probability
        
        # Alpha band features are typically indices 1-2 for each channel
        alpha_powers = []
        for ch in range(self.NUM_EEG_CH):
            alpha_idx = ch * self.NUM_BAND + 1  # Alpha band index
            if alpha_idx < len(features):
                alpha_powers.append(features[alpha_idx])
        
        if alpha_powers:
            avg_alpha = np.mean(alpha_powers)
            # Convert to probability (simplified)
            meditation_prob = min(1.0, max(0.0, avg_alpha * 2))
        else:
            meditation_prob = 0.5
        
        return meditation_prob


def generate_test_eeg_data(duration_seconds=5):
    """Generate synthetic EEG data for testing."""
    n_samples = int(duration_seconds * 256)  # 256 Hz
    n_channels = 4
    t = np.linspace(0, duration_seconds, n_samples)
    
    eeg_data = np.zeros((n_samples, n_channels))
    
    # Channel 1: Strong alpha waves (meditation-like)
    eeg_data[:, 0] = 50 * np.sin(2 * np.pi * 10 * t) + 10 * np.random.randn(n_samples)
    
    # Channel 2: Beta waves (active state)
    eeg_data[:, 1] = 30 * np.sin(2 * np.pi * 20 * t) + 15 * np.random.randn(n_samples)
    
    # Channel 3: Theta waves (relaxed)
    eeg_data[:, 2] = 40 * np.sin(2 * np.pi * 6 * t) + 12 * np.random.randn(n_samples)
    
    # Channel 4: Mixed frequencies
    eeg_data[:, 3] = (20 * np.sin(2 * np.pi * 8 * t) + 
                      15 * np.sin(2 * np.pi * 15 * t) + 
                      8 * np.random.randn(n_samples))
    
    return eeg_data, t


def main():
    """Main demo function."""
    print("Muse EEG Classifier Demo")
    print("=" * 30)
    print("This demo shows the core preprocessing pipeline")
    print("without requiring external dependencies.\n")
    
    # Initialize classifier
    classifier = SimpleEEGClassifier()
    
    # Generate test data
    print("Generating synthetic EEG data...")
    eeg_data, t = generate_test_eeg_data(duration_seconds=3)
    print(f"Generated {eeg_data.shape[0]} samples across {eeg_data.shape[1]} channels")
    print(f"Duration: {t[-1]:.1f} seconds\n")
    
    # Process the data
    print("Processing EEG data...")
    result = classifier.process_eeg_data(eeg_data)
    
    if result:
        print(f"✓ Processing successful!")
        print(f"  Meditation probability: {result['meditation_probability']:.3f}")
        print(f"  Features extracted: {result['feature_count']}")
        print(f"  Feature values: {[f'{x:.3f}' for x in result['features'][:8]]}... (showing first 8)")
        
        # Show frequency band analysis
        print(f"\nFrequency band analysis:")
        band_names = ['Theta', 'Alpha', 'Beta', 'Gamma', 'High Beta', 'Gamma']
        for ch in range(min(2, classifier.NUM_EEG_CH)):  # Show first 2 channels
            print(f"  Channel {ch+1}:")
            for band in range(classifier.NUM_BAND):
                feature_idx = ch * classifier.NUM_BAND + band
                if feature_idx < len(result['features']):
                    power = result['features'][feature_idx]
                    print(f"    {band_names[band]}: {power:.3f}")
    else:
        print("✗ Processing failed!")
    
    print(f"\nDemo completed!")
    print(f"\nTo run the full classifier with real Muse data:")
    print(f"  1. Install dependencies: pip install -r requirements.txt")
    print(f"  2. Run: python realtime_eeg_classifier.py")


if __name__ == "__main__":
    main()
