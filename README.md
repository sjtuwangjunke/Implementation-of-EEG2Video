# Implementation-of-EEG2Video

This repository is a **reorganized and extended version** of the official implementation of the NeurIPS 2024 paper:

**[EEG2Video: Towards Decoding Dynamic Visual Perception from EEG Signals](https://nips.cc/virtual/2024/poster/95156)**  
*Xuan-Hao Liu, Yan-Kai Liu, Yansen Wang, Kan Ren, Hanwen Shi, Zilong Wang, Dongsheng Li, Bao-Liang Lu, Wei-Long Zheng*

---

## 🔧 Modifications Made
- Explained how to use the files in README.md files
- Clarified and supplemented previously undocumented or ambiguous project files.
---
## Installation

1. Fill out the SEED-DV's [License file](https://cloud.bcmi.sjtu.edu.cn/sharing/o64PBIsIc) and [Apply](https://bcmi.sjtu.edu.cn/ApplicationForm/apply_form/) the dataset.

2. Download this repository: ``git clone https://github.com/sjtuwangjunke/Implementation-of-EEG2Video.git``

3. Create a conda environment and install the packages necessary to run the code. (GTX4060 and python3.9 on my own device)

```bash
conda create -n eegvideo python=3.9 -y
conda activate eegvideo
pip install -r requirements.txt
```
4. Follow the README.md in each of the subdirectory

## 📚 Citation
If you use this code or the original EEG2Video model, please cite:

```bibtex
@inproceedings{liu2024eegvideo,
  title={{EEG}2Video: Towards Decoding Dynamic Visual Perception from {EEG} Signals},
  author={Liu, Xuan-Hao and Liu, Yan-Kai and Wang, Yansen and Ren, Kan and Shi, Hanwen and Wang, Zilong and Li, Dongsheng and Lu, Bao-Liang and Zheng, Wei-Long},
  booktitle={NeurIPS},
  year={2024}

## Credits
- `EEG2Video_New/Seq2Seq/train_latents_generator.py` and `EEG2Video_New/Seq2Seq/train_latents_generator.py` are based on a code snippet posted by GitHub user [@gaspachoo](https://github.com/gaspachoo)  
  in [this comment](https://github.com/XuanhaoLiu/EEG2Video/issues/27#issuecomment-2921221772).  
