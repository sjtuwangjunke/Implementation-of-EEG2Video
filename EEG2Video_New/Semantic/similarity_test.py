import torch
import torch.nn.functional as F
from einops import rearrange
# ---------------------------
# 1. 加载两份文件
# ---------------------------
a = torch.load('sub1_session7_embeddings.pt')     # [200, 77 * 768]
b = torch.load('/home/drink/EEG2Video/EEG2Video_New/Semantic/text_embeddings/block7.pt')     # [200, 77, 768]
b = rearrange(b, 'a b c -> a (b c)')
assert a.shape == b.shape, "两份文件形状不一致！"

# ---------------------------
# 2. 计算余弦相似度
#    把 77*768 展平成 59136 维向量 → 再算余弦
# ---------------------------
a_flat = a.view(a.size(0), -1)          # [200, 59136]
b_flat = b.view(b.size(0), -1)          # [200, 59136]

cos_sim = F.cosine_similarity(a_flat, b_flat, dim=1)  # [200]

# ---------------------------
# 3. 打印结果
# ---------------------------
print("逐样本余弦相似度：\n", cos_sim.numpy())
print(f"平均相似度: {cos_sim.mean().item():.4f}")
print(f"最小相似度: {cos_sim.min().item():.4f}")
print(f"最大相似度: {cos_sim.max().item():.4f}")