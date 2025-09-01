# save as: concat_blocks_7x200.py
import torch
import numpy as np
import os

emb_dir = '.'          # 7 个 .pt 所在文件夹
out_npy = 'text_embeddings.npy'

# 按 block1.pt … block7.pt 的顺序读
tensors = []
for i in range(1, 8):
    pt_path = os.path.join(emb_dir, f'block{i}.pt')
    t = torch.load(pt_path)          # (200,77,768)
    tensors.append(t.unsqueeze(0))   # 变成 (1,200,77,768)

# 在 PyTorch 里拼接：dim=0 → (7,200,77,768)
big_tensor = torch.cat(tensors, dim=0)
np.save(out_npy, big_tensor.numpy())

print('✅ 已生成：', out_npy, 'shape =', big_tensor.shape)