# Muse EEG Real-time Classifier

This Python implementation replicates the Android MuseEEGClassifier functionality for real-time EEG classification using a Muse headband. It implements the same preprocessing pipeline and SVM model as the original Android application.

## Features

- **Real-time EEG Classification**: Classifies meditation vs active/stress states in real-time
- **LSL Integration**: Works with Muse LSL streams for live data processing
- **Android-compatible Pipeline**: Implements the exact same preprocessing as the Android version
- **SVM Model Support**: Loads and uses the libsvm format model from the Android app
- **Bluetooth Discovery**: Automatically discovers and connects to Muse devices

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- `pylsl` - Lab Streaming Layer for real-time data streaming
- `scikit-learn` - Machine learning library for SVM support
- `numpy` - Numerical computing
- `scipy` - Scientific computing (signal processing)
- `matplotlib` - Plotting and visualization
- `bleak` - Bluetooth Low Energy communication
- `muselsl` - Muse LSL streaming

## Usage

### Real-time Classification

Run the main classifier script:

```bash
python realtime_eeg_classifier.py
```

The script will:
1. Look for existing Muse LSL streams
2. If none found, scan for Muse devices via Bluetooth
3. Start muselsl streaming
4. Begin real-time classification
5. Display meditation probability scores

### Testing the Pipeline

Test the preprocessing pipeline with synthetic data:

```bash
python test_classifier.py
```

This will:
1. Generate synthetic EEG data
2. Test all preprocessing steps
3. Generate visualization plots
4. Test buffer processing

## Model File

The classifier expects the SVM model file at:
```
../app/src/main/assets/final_svm_model.txt
```

You can specify a different path by modifying the `model_path` parameter in the `MuseEEGClassifier` constructor.

## Preprocessing Pipeline

The implementation follows the exact same preprocessing pipeline as the Android version:

1. **Pre-filtering**: IIR filtering to remove artifacts
2. **Artifact Removal**: Moving average technique to clean the signal
3. **Bandpass Filtering**: Extract 6 frequency bands (theta, alpha, beta, etc.)
4. **Feature Extraction**: Calculate relative power in each band for each channel
5. **Classification**: Use SVM model to predict meditation probability

### Technical Details

- **Sampling Rate**: 256 Hz
- **Window Size**: 2 seconds (512 samples)
- **Overlap**: 50% between windows
- **Channels**: 4 EEG channels (TP9, AF7, AF8, TP10)
- **Frequency Bands**: 6 bands (excluding delta)
- **Features**: 24 features (4 channels × 6 bands)

## Output

The classifier outputs:
- **Meditation Probability**: Smoothed probability score (0-1)
- **Raw Probability**: Unsmoothed probability score
- **Real-time Display**: Continuous updates showing current meditation level

## Troubleshooting

### No Muse Device Found
- Ensure Muse headband is powered on
- Check Bluetooth connectivity
- Try running `muselsl list` to see available devices

### Model Loading Issues
- Verify the model file path is correct
- Check that `final_svm_model.txt` exists in the assets folder
- Ensure the model file is not corrupted

### LSL Stream Issues
- Install muselsl: `pip install muselsl`
- Check if other applications are using the Muse
- Restart the Muse headband if needed

## Comparison with Android Version

This Python implementation maintains compatibility with the Android version:

- **Same Constants**: Uses identical parameters from `SVMConstants.java`
- **Same Preprocessing**: Implements the exact artifact removal and feature extraction
- **Same Model**: Loads the identical libsvm model file
- **Same Features**: Produces 24-feature vectors matching the Android implementation

## File Structure

```
tools/
├── realtime_eeg_classifier.py  # Main classifier script
├── test_classifier.py          # Test script with synthetic data
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## License

This implementation follows the same license as the original Android MuseEEGClassifier project.
