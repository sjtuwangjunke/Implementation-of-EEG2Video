import numpy as np
import torch
import imageio
from tqdm import tqdm
from torchvision import transforms
from diffusers.models import AutoencoderKL
import os
# === Config ===
video_root = "/home/drink/SEED-DV/Video_Gif/"
out_path = "train_latents.npy"

vae = AutoencoderKL.from_pretrained("/home/drink/huggingface/sd-vae-ft-mse").cuda()
vae.eval()
transform = transforms.ToTensor()

all_latents = []
#huggingface-cli download stabilityai/sd-vae-ft-mse --local-dir /home/drink/huggingface
for block in tqdm(range(1, 7),desc="Generating Latents"):  # Blocks 1–6 for training
    video_dir = os.path.join(video_root, f"Block{block}")
    for concept in tqdm(range(40), desc=f"Block {block}"):
        for rep in range(5):
            gif_index = concept * 5 + rep + 1
            gif_path = os.path.join(video_dir, f"{gif_index}.gif")
            if not os.path.exists(gif_path):
                print(f"❌ Missing video: {gif_path}")
                continue

            frames = imageio.mimread(gif_path)
            if len(frames) != 6:
                print(f"⚠️ GIF {gif_path} has {len(frames)} frames, expected 6.")
                continue

            frames = [transform(f) for f in frames]
            frames = torch.stack(frames).cuda()  # shape: (6, 3, 288, 512)

            with torch.no_grad():
                z_latents = vae.encode(frames).latent_dist.mean  # (6, 4, 36, 64)

            all_latents.append(z_latents.cpu().numpy())

# === Final assembly ===
latents_np = np.stack(all_latents, axis=0).transpose(0,2,1,3,4)  # (1200, 4, 6, 36, 64)

np.save(out_path, latents_np)
print(f"✅ Saved latents to {out_path} with shape {latents_np.shape}")