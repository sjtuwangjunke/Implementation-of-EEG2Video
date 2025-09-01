import torch
import imageio
from tqdm import tqdm
from torchvision import transforms
from diffusers.models import AutoencoderKL
import os
# === Config ===
video_dir = "/home/drink/SEED-DV/Video_Gif/Block7"
out_path = "test_latents.pt"

vae = AutoencoderKL.from_pretrained("/home/drink/huggingface/sd-vae-ft-mse").cuda()
transform = transforms.ToTensor()

# === Génération ===
all_latents = []  # liste de 200 tensors (c, f, h, w)

for concept in tqdm(range(40), desc="Block7 concepts"):
    for rep in range(5):
        gif_index = concept * 5 + rep + 1
        gif_path = os.path.join(video_dir, f"{gif_index}.gif")

        if not os.path.exists(gif_path):
            print(f"❌ Missing GIF: {gif_path}")
            continue

        frames = imageio.mimread(gif_path)
        if len(frames) != 6:
            print(f"⚠️ GIF {gif_path} has {len(frames)} frames (expected 6)")
            continue

        frames = torch.stack([transform(f) for f in frames]).cuda()  # (6, 3, H, W)

        with torch.no_grad():
            z = vae.encode(frames).latent_dist.mean  # (6, 4, 36, 64)

        z = z.permute(1, 0, 2, 3).cpu()  # → (4, 6, 36, 64)
        all_latents.append(z)

# === Final stack ===
final_tensor = torch.stack(all_latents)  # → (200, 4, 6, 36, 64)
torch.save(final_tensor, out_path)

print(f"✅ Saved latents to {out_path} with shape {final_tensor.shape}")