#!/usr/bin/env python3
# save as: clip_line_by_line.py
import os
import torch
from transformers import CLIPModel, CLIPTokenizer

def encode_line_by_line(model_path, txt_path, out_path):
    # 1. 加载模型
    model = CLIPModel.from_pretrained(model_path).eval()
    tokenizer = CLIPTokenizer.from_pretrained(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # 2. 读 200 行
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 200, f'{txt_path} 行数≠200'

    # 3. 逐行编码
    all_emb = []
    with torch.no_grad():
        for text in lines:
            inputs = tokenizer(
                text,
                padding='max_length',      # 固定 77
                truncation=True,
                max_length=77,
                return_tensors='pt'
            ).to(device)
            # 取最后一层所有 token： (1, 77, 768)
            emb = model.text_model(**inputs).last_hidden_state
            all_emb.append(emb.squeeze(0).cpu())  # (77, 768)
    # 4. 拼接成 200×77×768
    all_emb = torch.stack(all_emb, dim=0)  # (200, 77, 768)
    torch.save(all_emb, out_path)
    print(f'✅ saved {out_path}  shape {list(all_emb.shape)}')


# ========== 主程序 ==========
if __name__ == "__main__":
    model_path = "/home/drink/huggingface/clip-vit-large-patch14"
    text_dir   = "/home/drink/SEED-DV/Video/BLIP-caption"
    out_dir    = "text_embeddings"
    os.makedirs(out_dir, exist_ok=True)

    for block_id in range(1, 8):          # 1.txt ~ 7.txt
        txt_path = os.path.join(text_dir, f"{block_id}.txt")
        out_path = os.path.join(out_dir, f"block{block_id}.pt")
        encode_line_by_line(model_path, txt_path, out_path)