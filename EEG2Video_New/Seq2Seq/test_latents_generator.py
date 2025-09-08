import torch
import imageio
from tqdm import tqdm
from torchvision import transforms
from diffusers.models import AutoencoderKL
import os

# === Config ===
video_dir = "SEED-DV/Video_Gif/Block7"  # folder containing 200 GIFs for test block 7
out_path  = "test_latents.pt"                       # output file to save latent tensors

# load pre-trained VAE and send to GPU
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").cuda()
transform = transforms.ToTensor()  # convert PIL/ndarray → [0,1] tensor

all_latents = []

# 40 concepts
for concept in tqdm(range(40), desc="Block7 concepts"):
    # 5 repetitions per concept
    for rep in range(5):
        gif_index = concept * 5 + rep # global index 0~199
        gif_path = os.path.join(video_dir, f"{gif_index}.gif")

        if not os.path.exists(gif_path):
            print(f"❌ Missing GIF: {gif_path}")
            continue

        frames = imageio.mimread(gif_path)
        if len(frames) != 6:
            print(f"⚠️ GIF {gif_path} has {len(frames)} frames (expected 6)")
            continue

        # convert frames to tensors and stack → (6, 3, H, W) then send to GPU
        frames = torch.stack([transform(f) for f in frames]).cuda()

        # encode to VAE latent space → (6, 4, 36, 64)
        with torch.no_grad():
            z = vae.encode(frames).latent_dist.mean

        # reorder dimensions: (6, 4, 36, 64) → (4, 6, 36, 64) and move to CPU
        z = z.permute(1, 0, 2, 3).cpu()
        all_latents.append(z)

# === Final stack ===
# stack list into single tensor: (200, 4, 6, 36, 64)
final_tensor = torch.stack(all_latents) 
torch.save(final_tensor, out_path)

print(f"✅ Saved latents to {out_path} with shape {final_tensor.shape}")
