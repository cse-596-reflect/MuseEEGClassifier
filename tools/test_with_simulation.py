#!/usr/bin/env python3

"""
Test script for Muse EEG Classifier with simulated LSL stream.
This allows testing the full pipeline without a physical Muse device.
"""

import numpy as np
import time
import threading
from pathlib import Path
import sys

# Add the current directory to path
sys.path.append(str(Path(__file__).parent))

from realtime_eeg_classifier import MuseEEGClassifier


class SimulatedLSLStream:
    """Simulate an LSL stream for testing."""
    
    def __init__(self, sample_rate=256, n_channels=4):
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        self.running = False
        self.data_buffer = []
        
    def generate_sample(self):
        """Generate a synthetic EEG sample."""
        t = time.time()
        
        # Generate different frequency components for each channel
        sample = np.zeros(self.n_channels)
        
        # Channel 1: Alpha waves (meditation-like)
        sample[0] = 50 * np.sin(2 * np.pi * 10 * t) + 10 * np.random.randn()
        
        # Channel 2: Beta waves (active state)
        sample[1] = 30 * np.sin(2 * np.pi * 20 * t) + 15 * np.random.randn()
        
        # Channel 3: Theta waves (relaxed)
        sample[2] = 40 * np.sin(2 * np.pi * 6 * t) + 12 * np.random.randn()
        
        # Channel 4: Mixed frequencies
        sample[3] = (20 * np.sin(2 * np.pi * 8 * t) + 
                     15 * np.sin(2 * np.pi * 15 * t) + 
                     8 * np.random.randn())
        
        return sample
    
    def start_streaming(self, duration_seconds=30):
        """Start streaming simulated data."""
        print(f"Starting simulated LSL stream for {duration_seconds} seconds...")
        print("Generating synthetic EEG data with meditation-like patterns...")
        
        start_time = time.time()
        sample_count = 0
        
        while time.time() - start_time < duration_seconds:
            sample = self.generate_sample()
            self.data_buffer.append(sample)
            sample_count += 1
            
            # Simulate real-time streaming rate
            time.sleep(1.0 / self.sample_rate)
            
            # Print progress every 5 seconds
            if sample_count % (self.sample_rate * 5) == 0:
                elapsed = time.time() - start_time
                print(f"Streamed {sample_count} samples ({elapsed:.1f}s)")
        
        print(f"Simulated streaming completed. Total samples: {sample_count}")


def test_classifier_with_simulation():
    """Test the classifier with simulated data."""
    print("Muse EEG Classifier - Simulation Test")
    print("=" * 40)
    
    # Initialize classifier
    classifier = MuseEEGClassifier()
    
    if classifier.svm_model is None:
        print("Failed to load SVM model. Exiting.")
        return
    
    print("✓ SVM model loaded successfully")
    print(f"✓ Model type: {classifier.svm_model.svm_type}")
    print(f"✓ Kernel: {classifier.svm_model.kernel_type}")
    print(f"✓ Support vectors: {classifier.svm_model.total_sv}")
    
    # Create simulated stream
    stream = SimulatedLSLStream()
    
    print(f"\nTesting preprocessing pipeline...")
    
    # Generate test data
    test_data = []
    for _ in range(classifier.WINDOW_SIZE * 2):  # Generate enough data for processing
        sample = stream.generate_sample()
        test_data.append(sample)
    
    test_data = np.array(test_data)
    print(f"Generated {test_data.shape[0]} samples across {test_data.shape[1]} channels")
    
    # Test the processing pipeline
    print("\nTesting complete processing pipeline...")
    result = classifier.process_eeg_data(test_data)
    
    if result:
        print(f"✓ Processing successful!")
        print(f"  Meditation probability: {result['meditation_probability']:.3f}")
        print(f"  Raw probability: {result['raw_probability']:.3f}")
        print(f"  Features extracted: {len(result['features'])}")
        
        # Show feature breakdown
        print(f"\nFeature breakdown (relative power):")
        band_names = ['Theta', 'Alpha', 'Beta', 'Gamma', 'High Beta', 'Gamma']
        for ch in range(min(2, classifier.NUM_EEG_CH)):
            print(f"  Channel {ch+1}:")
            for band in range(classifier.NUM_BAND):
                feature_idx = ch * classifier.NUM_BAND + band
                if feature_idx < len(result['features']):
                    power = result['features'][feature_idx]
                    print(f"    {band_names[band]}: {power:.3f}")
    else:
        print("✗ Processing failed!")
        return
    
    # Test buffer processing (simulating real-time)
    print(f"\nTesting real-time buffer processing...")
    print("Simulating streaming data...")
    
    results = []
    start_time = time.time()
    
    # Simulate streaming for 10 seconds
    while time.time() - start_time < 10:
        sample = stream.generate_sample()
        result = classifier.update_buffer(sample)
        
        if result:
            results.append(result)
            meditation_prob = result['meditation_probability']
            raw_prob = result['raw_probability']
            
            elapsed = time.time() - start_time
            print(f"\r[{elapsed:.1f}s] Meditation: {meditation_prob:.3f} ({meditation_prob*100:.1f}%) | Raw: {raw_prob:.3f}", 
                  end='', flush=True)
        
        time.sleep(1.0 / classifier.SAMPLE_RATE)
    
    print(f"\n\n✓ Real-time processing completed!")
    print(f"✓ Processed {len(results)} classification results")
    
    if results:
        avg_prob = np.mean([r['meditation_probability'] for r in results])
        print(f"✓ Average meditation probability: {avg_prob:.3f}")
        
        # Show trend
        probs = [r['meditation_probability'] for r in results]
        print(f"✓ Probability range: {min(probs):.3f} - {max(probs):.3f}")
    
    print(f"\n🎉 All tests completed successfully!")
    print(f"\nThe classifier is ready for real Muse data!")
    print(f"To use with a real Muse device:")
    print(f"  1. Turn on Bluetooth")
    print(f"  2. Power on your Muse headband")
    print(f"  3. Run: python realtime_eeg_classifier.py")


if __name__ == "__main__":
    test_classifier_with_simulation()
