import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce


"""
Different EEG encoders for comparison
shallownet, deepnet, eegnet, conformer, tsconv
"""
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        self.deepnet = nn.Sequential(
            nn.Conv2d(1, 25, (1, 10), (1, 1)),
            nn.Conv2d(25, 25, (63, 1), (1, 1)),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(25, 50, (1, 10), (1, 1)),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(50, 100, (1, 10), (1, 1)),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(100, 200, (1, 10), (1, 1)),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),

        )

        self.eegnet = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), (1, 1)),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (63, 1), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(16, 16, (1, 16), (1, 1)),
            nn.BatchNorm2d(16), 
            nn.ELU(),
            # nn.AvgPool2d((1, 2), (1, 2)),
            nn.Dropout2d(0.5)
        )

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (63, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # 5 is better than 1
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.tsconv(x)
        return x

class shallownet(nn.Module):
    """
    Shallow ConvNet: two-layer CNN with aggressive pooling.
    """
    def __init__(self, out_dim, C, T):
        super(shallownet, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (C, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.Dropout(0.5),
        )
        self.out = nn.Linear(1040*(T//200), out_dim)
    
    def forward(self, x):               #input:(batch,1,C,T)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x
    
class deepnet(nn.Module):
    """
    Deep ConvNet: four blocks of conv->norm->activation->pool->drop.
    """
    def __init__(self, out_dim, C, T):
        super(deepnet, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(1, 25, (1, 10), (1, 1)),
            nn.Conv2d(25, 25, (C, 1), (1, 1)),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(25, 50, (1, 10), (1, 1)),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(50, 100, (1, 10), (1, 1)),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(100, 200, (1, 10), (1, 1)),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
        )
        self.out = nn.Linear(800*(T//200), out_dim)
    
    def forward(self, x):               #input:(batch,1,C,T)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x
    
class eegnet(nn.Module):
    """
    EEGNet: depth-wise separable convolutions.
    """
    def __init__(self, out_dim, C, T):
        super(eegnet, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), (1, 1)),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (C, 1), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(16, 16, (1, 16), (1, 1)),
            nn.BatchNorm2d(16), 
            nn.ELU(),
            nn.AvgPool2d((1, 2), (1, 2)),
            nn.Dropout2d(0.5)
        )
        self.out = nn.Linear(416*(T//200), out_dim)
    
    def forward(self, x):               #input:(batch,1,C,T)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x

class tsconv(nn.Module):
    """
    Temporal-Spatial ConvNet: temporal conv followed by spatial conv.
    """
    def __init__(self, out_dim, C, T):
        super(tsconv, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (C, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        self.out = nn.Linear(1040*(T//200), out_dim)
    
    def forward(self, x):               #input:(batch,1,C,T)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x

# ----------- Transformer components -----------
# Convolution module, use conv to capture local features, instead of postion embedding.
class PatchEmbedding(nn.Module):
    """
    Patch embedding for transformer-based models.
    """
    def __init__(self, emb_size=40):
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (62, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )


    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention.
    """
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    """
    Global average pooling followed by a linear classifier.
    """
    def __init__(self, emb_size, out_dim):
        super().__init__()
        
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, out_dim)
        )
        self.fc = nn.Sequential(
            nn.Linear(280, out_dim),
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out

class conformer(nn.Sequential):
    """
    Conformer: CNN patch embedding + transformer encoder + classifier.
    """
    def __init__(self, emb_size=40, depth=3, out_dim=4, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            # nn.Linear(280, out_dim)
            ClassificationHead(emb_size, out_dim)
        )

# ----------- Global–Local Fusion models -----------
class glfnet(nn.Module):
    """
    Global-Local Fusion network for EEG.
    """
    def __init__(self, out_dim, emb_dim, C, T):
        super(glfnet, self).__init__()
        
        self.globalnet = shallownet(emb_dim, C, T)
        
        self.occipital_index = list(range(50, 62))
        self.occipital_localnet = shallownet(emb_dim, 12, T)
        
        self.out = nn.Linear(emb_dim*2, out_dim)
        
    
    def forward(self, x):               #input:(batch,1,C,T)
        global_feature = self.globalnet(x)
        global_feature = global_feature.view(x.size(0), -1)
        # global_feature = self.out(global_feature)
        occipital_x = x[:, :, self.occipital_index, :]
        # print("occipital_x.shape = ", occipital_x.shape)
        occipital_feature = self.occipital_localnet(occipital_x)
        # print("occipital_feature.shape = ", occipital_feature.shape)
        out = self.out(torch.cat((global_feature, occipital_feature), 1))
        return out

class mlpnet(nn.Module):
    """
    Simple MLP baseline.
    """
    def __init__(self, out_dim, input_dim):
        super(mlpnet, self).__init__()
        
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, out_dim)
        )
        
    def forward(self, x):               #input:(batch,C,5)
        out = self.net(x)
        return out

class GLMNet(nn.Module): #input:(batch,channels_conv,channels_eeg,data_num)  output:(batch,num_classes)
    """
    Global-Local MLP fusion for multi-channel feature maps.
    """
    def __init__(self, out_dim, emb_dim, input_dim):
        super(GLMNet, self).__init__()
        
        self.globalnet = mlpnet(emb_dim, input_dim)
        
        self.occipital_index = list(range(50, 62))
        self.occipital_localnet = mlpnet(emb_dim, 12*5)
        
        self.out = nn.Linear(emb_dim*2, out_dim)
        
    
    def forward(self, x):               #input:(batch,C,5)
        global_feature = self.globalnet(x)
        occipital_x = x[:, self.occipital_index, :]
        occipital_feature = self.occipital_localnet(occipital_x)
        out = self.out(torch.cat((global_feature, occipital_feature), 1))
        return out
    
class mlpnet_raw(nn.Module):
    """
    Simple MLP baseline.
    """
    def __init__(self, out_dim, input_dim):
        super(mlpnet_raw, self).__init__()
        
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, out_dim)
        )
        
    def forward(self, x):               #input:(batch,C,T)
        out = self.net(x)
        return out

class GLMNet_raw(nn.Module): #input:(batch,channels_conv,channels_eeg,data_num)  output:(batch,num_classes)
    """
    Global-Local MLP fusion for multi-channel feature maps.
    """
    def __init__(self, out_dim, emb_dim, input_dim, time_step):
        super(GLMNet_raw, self).__init__()
        
        self.globalnet = mlpnet_raw(emb_dim, input_dim)
        
        self.occipital_index = list(range(50, 62))
        self.occipital_localnet = mlpnet(emb_dim, 12*time_step)
        
        self.out = nn.Linear(emb_dim*2, out_dim)
        
    
    def forward(self, x):               #input:(batch,C,T)
        global_feature = self.globalnet(x)
        occipital_x = x[:, self.occipital_index, :]
        occipital_feature = self.occipital_localnet(occipital_x)
        out = self.out(torch.cat((global_feature, occipital_feature), 1))
        return out
