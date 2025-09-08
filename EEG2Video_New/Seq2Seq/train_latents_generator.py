import numpy as np
import torch
import imageio
from tqdm import tqdm
from torchvision import transforms
from diffusers.models import AutoencoderKL
import os
# === Config ===
video_root = "SEED-DV/Video_Gif/"  # root folder containing Block1, Block2, ..., Block6
out_path   = "train_latents.npy"               # output file for stored latents

# load pre-trained VAE (fine-tuned MSE version) and set to evaluation mode
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").cuda()
vae.eval()

# convert PIL/ndarray images to [0, 1] tensors
transform = transforms.ToTensor()

all_latents = []

# Blocks 1–6 for training
for block in tqdm(range(1, 7),desc="Generating Latents"): 
    video_dir = os.path.join(video_root, f"Block{block}")
    # 40 concepts per block
    for concept in tqdm(range(40), desc=f"Block {block}"):
         # 5 repetitions per concept
        for rep in range(5):
            gif_index = concept * 5 + rep # global GIF index 0~199
            gif_path = os.path.join(video_dir, f"{gif_index}.gif")
            if not os.path.exists(gif_path):
                print(f"❌ Missing video: {gif_path}")
                continue
            
            # read all frames from GIF (expected 6 frames)
            frames = imageio.mimread(gif_path)
            if len(frames) != 6:
                print(f"⚠️ GIF {gif_path} has {len(frames)} frames, expected 6.")
                continue
            
            # convert frames to tensors and stack into single batch (6, 3, 288, 512)
            frames = [transform(f) for f in frames]
            frames = torch.stack(frames).cuda()
            
            # encode to VAE latent space (shape: 6, 4, 36, 64)
            with torch.no_grad():
                z_latents = vae.encode(frames).latent_dist.mean

            all_latents.append(z_latents.cpu().numpy())

# === Final assembly ===
# stack all latent chunks into one array and reorder dimensions
# final shape: (1200, 4, 6, 36, 64)  ->  1200 GIFs, 4 latent channels, 6 frames, 36x64 spatial
latents_np = np.stack(all_latents, axis=0).transpose(0,2,1,3,4)

np.save(out_path, latents_np)
print(f"✅ Saved latents to {out_path} with shape {latents_np.shape}")
