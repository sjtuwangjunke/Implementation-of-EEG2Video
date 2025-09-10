# EEG2Video_New Folder Map

| Folder | Paper Module | What It Does |
|--------|--------------|--------------|
| `Semantic/` | Semantic Predictor | Trains & infers **EEG → 77×768 text embeddings**; includes scripts to generate CLIP text features from BLIP captions. |
| `Seq2Seq/` | Seq2Seq Transformer | Trains & infers **EEG → 6×4×36×64 video latents**; contains autoregressive Transformer and utilities to produce train/test latents. |
| `DANA/` | Dynamic-Aware Noise Adding | Implements fast/slow-conditioned noise scheduling (β=0.2/0.3) for Stable Diffusion. |
| `Generation/` | Inflated Stable Diffusion | Fine-tunes Tune-A-Video weights; pipelines for EEG→video inference; one-click 40-class evaluation scripts. |
| `extract_gif.py` | Data Pre-processing | Batch-extracts frames from original GIFs. |

---

## Quick Start

**Please change all the paths to your own to keep consistency and avoid errors**

Firstly, to get the gifs ready, run
```
python extract_gif.py
```
To train your own Seq2Seq model, run 
```
cd Seq2Seq
python train_latents_generator.py
python test_latents_generator.py
python my_autoregressive_transformer.py
cd ..
```
To train your own semantic predictor, run
```
cd Semantic
python text_embeddings_generator.py #generate text embeddings with CLIP
python eeg2text.py
python model_test.py
cd ..
```
To add noise for stable diffusion, run
```
cd DANA
python add_noise.py
cd ..
```
To generate delicate video outputs, train videodiffusion model with
```
cd Generation
python train_finetune_videodiffusion.py
cd ..
```
Finally, run
```
cd Generation
python inference_eeg2video.py
python 40_class_run_metrics.py
cd ..
```
Tips: to generate clear gifs, try
```
woSeq2Seq = True
woDANA = False
``` 
