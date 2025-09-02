
# SEED-DV EEG-VP Scripts
This repository provides a lightweight PyTorch implementation for decoding gaze **vision** (V) and **position** (P) information from multi-channel EEG signals, including EEG models and training code.

## Quick Start
First change the paths of the dataset, output directory and the result directory to your own.

To train and test on raw signals(Segmented during the preprocessing), run
```bash
python Raw_EEG_VP_train_test.py
```

To train and test on DE or PSD features(1 second), run
```bash
python DE&PSD_EEG_VP_train_test.py
```
