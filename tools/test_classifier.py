#!/usr/bin/env python3

"""
Test script for the Muse EEG Classifier.
This script tests the preprocessing pipeline with synthetic data.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the current directory to path to import the classifier
sys.path.append(str(Path(__file__).parent))

from realtime_eeg_classifier import MuseEEGClassifier


def generate_synthetic_eeg_data(duration_seconds=10, sample_rate=256):
    """Generate synthetic EEG data for testing."""
    n_samples = int(duration_seconds * sample_rate)
    n_channels = 4
    
    # Generate time vector
    t = np.linspace(0, duration_seconds, n_samples)
    
    # Generate synthetic EEG data with different frequency components
    eeg_data = np.zeros((n_samples, n_channels))
    
    # Channel 1: Alpha waves (8-12 Hz) - meditation-like
    alpha_freq = 10
    eeg_data[:, 0] = 50 * np.sin(2 * np.pi * alpha_freq * t) + 10 * np.random.randn(n_samples)
    
    # Channel 2: Beta waves (13-30 Hz) - active state
    beta_freq = 20
    eeg_data[:, 1] = 30 * np.sin(2 * np.pi * beta_freq * t) + 15 * np.random.randn(n_samples)
    
    # Channel 3: Theta waves (4-8 Hz) - relaxed state
    theta_freq = 6
    eeg_data[:, 2] = 40 * np.sin(2 * np.pi * theta_freq * t) + 12 * np.random.randn(n_samples)
    
    # Channel 4: Mixed frequencies
    eeg_data[:, 3] = (20 * np.sin(2 * np.pi * 8 * t) + 
                      15 * np.sin(2 * np.pi * 15 * t) + 
                      10 * np.sin(2 * np.pi * 25 * t) + 
                      8 * np.random.randn(n_samples))
    
    return eeg_data, t


def test_preprocessing_pipeline():
    """Test the preprocessing pipeline with synthetic data."""
    print("Testing Muse EEG Classifier Preprocessing Pipeline")
    print("=" * 50)
    
    # Initialize classifier
    classifier = MuseEEGClassifier()
    
    if classifier.svm_model is None:
        print("Warning: SVM model not loaded. Testing preprocessing only.")
    
    # Generate synthetic data
    print("Generating synthetic EEG data...")
    eeg_data, t = generate_synthetic_eeg_data(duration_seconds=5)
    
    print(f"Generated {eeg_data.shape[0]} samples across {eeg_data.shape[1]} channels")
    print(f"Sample rate: {classifier.SAMPLE_RATE} Hz")
    print(f"Duration: {t[-1]:.1f} seconds")
    
    # Test preprocessing steps
    print("\nTesting preprocessing steps...")
    
    # 1. Pre-filtering
    print("1. Applying pre-filtering...")
    filtered_data = classifier.apply_prefilter(eeg_data)
    print(f"   Pre-filtered data shape: {filtered_data.shape}")
    
    # 2. Artifact removal
    print("2. Applying artifact removal...")
    clean_data = classifier.artifact_removal(filtered_data)
    print(f"   Clean data shape: {clean_data.shape}")
    
    # 3. Feature extraction
    print("3. Extracting features...")
    features = classifier.extract_features(clean_data)
    print(f"   Features shape: {features.shape}")
    print(f"   Number of segments: {features.shape[0]}")
    print(f"   Features per segment: {features.shape[1]}")
    
    # Test classification
    if classifier.svm_model is not None:
        print("\n4. Testing classification...")
        result = classifier.process_eeg_data(clean_data)
        if result:
            print(f"   Meditation probability: {result['meditation_probability']:.3f}")
            print(f"   Raw probability: {result['raw_probability']:.3f}")
        else:
            print("   Classification failed")
    
    # Plot results
    print("\nGenerating plots...")
    plot_results(eeg_data, filtered_data, clean_data, features, t)
    
    print("\nTest completed successfully!")


def plot_results(raw_data, filtered_data, clean_data, features, t):
    """Plot the preprocessing results."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    # Plot raw data
    for ch in range(min(4, raw_data.shape[1])):
        axes[0].plot(t, raw_data[:, ch], label=f'Ch{ch+1}', alpha=0.7)
    axes[0].set_title('Raw EEG Data')
    axes[0].set_ylabel('Amplitude (μV)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot filtered data
    for ch in range(min(4, filtered_data.shape[1])):
        axes[1].plot(t, filtered_data[:, ch], label=f'Ch{ch+1}', alpha=0.7)
    axes[1].set_title('Pre-filtered EEG Data')
    axes[1].set_ylabel('Amplitude (μV)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot clean data
    for ch in range(min(4, clean_data.shape[1])):
        axes[2].plot(t, clean_data[:, ch], label=f'Ch{ch+1}', alpha=0.7)
    axes[2].set_title('Artifact-removed EEG Data')
    axes[2].set_ylabel('Amplitude (μV)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot features
    if len(features) > 0:
        feature_time = np.linspace(0, t[-1], len(features))
        im = axes[3].imshow(features.T, aspect='auto', cmap='viridis', 
                           extent=[0, t[-1], 0, features.shape[1]])
        axes[3].set_title('Extracted Features (Relative Power)')
        axes[3].set_xlabel('Time (s)')
        axes[3].set_ylabel('Feature Index')
        plt.colorbar(im, ax=axes[3], label='Relative Power')
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(__file__).parent / "test_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Show plot
    plt.show()


def test_buffer_processing():
    """Test the buffer processing with streaming-like data."""
    print("\nTesting buffer processing...")
    
    classifier = MuseEEGClassifier()
    
    # Generate data in chunks to simulate streaming
    chunk_size = 64  # samples per chunk
    total_samples = 1000
    n_chunks = total_samples // chunk_size
    
    print(f"Processing {total_samples} samples in {n_chunks} chunks of {chunk_size} samples each")
    
    results = []
    
    for chunk_idx in range(n_chunks):
        # Generate chunk of data
        chunk_data, _ = generate_synthetic_eeg_data(
            duration_seconds=chunk_size/256, 
            sample_rate=256
        )
        
        # Process each sample in the chunk
        for sample in chunk_data:
            result = classifier.update_buffer(sample)
            if result:
                results.append(result)
                print(f"Chunk {chunk_idx+1}: Meditation prob = {result['meditation_probability']:.3f}")
    
    print(f"\nProcessed {len(results)} classification results")
    if results:
        avg_prob = np.mean([r['meditation_probability'] for r in results])
        print(f"Average meditation probability: {avg_prob:.3f}")


if __name__ == "__main__":
    test_preprocessing_pipeline()
    test_buffer_processing()
