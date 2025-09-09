import os
import torch
from transformers import CLIPModel, CLIPTokenizer

def encode_line_by_line(model_path, txt_path, out_path):
    """
    Encode 200 lines of text into CLIP token-wise embeddings (200×77×768).
    Each line is padded/truncated to 77 tokens, and the last_hidden_state
    (before pooling) is saved.
    """
    # 1. load CLIP model & tokenizer
    model = CLIPModel.from_pretrained(model_path).eval()
    tokenizer = CLIPTokenizer.from_pretrained(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # 2. read exactly 200 non-empty lines
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 200, f'{txt_path} 行数≠200'

    # 3. encode line-by-line
    all_emb = []
    with torch.no_grad():
        for text in lines:
            inputs = tokenizer(
                text,
                padding='max_length',      # fix 77 tokens
                truncation=True,
                max_length=77,
                return_tensors='pt'
            ).to(device)
            # extract token-level features: (1, 77, 768)
            emb = model.text_model(**inputs).last_hidden_state
            all_emb.append(emb.squeeze(0).cpu())  # (77, 768)
    # 4. stack & save
    all_emb = torch.stack(all_emb, dim=0)  # (200, 77, 768)
    torch.save(all_emb, out_path)
    print(f'✅ saved {out_path}  shape {list(all_emb.shape)}')

def ordinal(n: int) -> str:
    """return 1st, 2nd, 3rd, 4th …"""
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

# ========== main ==========
if __name__ == "__main__":
    model_path = "openai/clip-vit-large-patch14"
    text_dir   = "SEED-DV/Video/BLIP-caption"
    out_dir    = "text_embeddings"
    os.makedirs(out_dir, exist_ok=True)

    for block_id in range(1, 8):
        txt_path = os.path.join(text_dir, f"{ordinal(block_id)}_10min.txt")
        out_path = os.path.join(out_dir, f"Block{block_id}.pt")
        encode_line_by_line(model_path, txt_path, out_path)
