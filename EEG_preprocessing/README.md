# SEED-DV EEG Pre-processing Scripts

This folder contains all scripts required to pre-process and extract features from the **SEED-DV dataset**.  

## Script Overview
| File | Purpose | Input | Output |
|------|---------|-------|--------|
| `DE_PSD.py` | Implementation of the DE_PSD function | — | — |
| `segment_raw_signals_200Hz.py` | Resample the EEG signals to 200 Hz, segments by trial and stores as `.npy` | `SEED-DV/EEG` | `SEED-DV/Segmented_Rawf_200Hz_2s` |
| `extract_DE_PSD_features_1per1s.py` | Loads 200 Hz segments, slides 1-second windows (no overlap), computes DE/PSD | `SEED-DV/Segmented_Rawf_200Hz_2s` | `SEED-DV/DE_1per1s` and `SEED-DV/PSD_1per1s` |
| `extract_DE_PSD_features_1per2s.py` | Same as above but uses 2-second windows | `SEED-DV/Segmented_Rawf_200Hz_2s` | `SEED-DV/DE_1per2s` and `SEED-DV/PSD_1per2s` |