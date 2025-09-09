import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn import preprocessing
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange
import os
# ---------- Model ----------
class CLIP(nn.Module):
    def __init__(self):
        super(CLIP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(310, 10000),
            nn.ReLU(),
            nn.Linear(10000, 10000),
            nn.ReLU(),
            nn.Linear(10000, 10000),
            nn.ReLU(),
            nn.Linear(10000, 10000),
            nn.ReLU(),
            nn.Linear(10000, 77 * 768)
        )

    def forward(self, eeg):
        eeg_embeddings = self.mlp(eeg) # (B, 77*768)
        return eeg_embeddings


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
# ---------- Dataset ----------
class Dataset():
    def __init__(self, eeg, text):
        self.eeg = eeg
        self.text = text
        self.len = eeg.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.eeg[item], self.text[item]

GT_label = np.array([[23, 22, 9, 6, 18,       14, 5, 36, 25, 19,      28, 35, 3, 16, 24,      40, 15, 27, 38, 33,
             34, 4, 39, 17, 1,       26, 20, 29, 13, 32,     37, 2, 11, 12, 30,      31, 8, 21, 7, 10, ],
            [27, 33, 22, 28, 31,     12, 38, 4, 18, 17,      35, 39, 40, 5, 24,      32, 15, 13, 2, 16,
 	         34, 25, 19, 30, 23,     3, 8, 29, 7, 20,        11, 14, 37, 6, 21,      1, 10, 36, 26, 9, ],
            [15, 36, 31, 1, 34,      3, 37, 12, 4, 5,        21, 24, 14, 16, 39,     20, 28, 29, 18, 32,
             2, 27, 8, 19, 13,       10, 30, 40, 17, 26,     11, 9, 33, 25, 35,      7, 38, 22, 23, 6,],
            [16, 28, 23, 1, 39,      10, 35, 14, 19, 27,     37, 31, 5, 18, 11,      25, 29, 13, 20, 24,
            7, 34, 26, 4, 40 ,       12, 8, 22, 21, 30,      17, 2, 38, 9,  3 ,      36, 33, 6, 32, 15,],
            [18, 29, 7, 35, 22  ,    19, 12, 36, 8, 15,      28, 1, 34, 23, 20 ,     13, 37, 9, 16, 30  ,
             2, 33, 27, 21, 14 ,     38, 10, 17, 31, 3,      24, 39, 11, 32, 4,      25, 40, 5, 26, 6 ,],
            [29, 16, 1, 22, 34,      39, 24, 10, 8, 35,      27, 31, 23, 17, 2,      15, 25, 40, 3, 36,
             26, 6, 14, 37, 9,       12, 19, 30, 5, 28,      32, 4, 13, 18, 21,      20, 7, 11, 33, 38],
            [38, 34, 40, 10, 28,     7, 1, 37, 22, 9,        16, 5, 12, 36, 20,      30, 6, 15, 35, 2,
             31, 26, 18, 24, 8,      3, 23, 19, 14, 13,      21, 4, 25, 11, 32,      17, 39, 29, 33, 27]
            ])
chosed_label = [i for i in range(1,41)]
labels = np.zeros((40, 5, 62, 5))
for i in range(40):
    labels[i]=i

import random
def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        ## needs to be False to use conv3D
        print('Note: not using cudnn.deterministic')
seed_everything(114514)
device='cuda:0'

if __name__ == '__main__':
    
    # 1. load EEG  (7 blocks, 40 vids, 5 trials, 62 ch, 400 t) → (1200, 310)
    eeg_data = np.load('SEED-DV/DE_1per2s/sub1.npy') #[7,40,5,62,400]
    eeg_label = []
    eeg = []
    for i in range(6):
        indices = [list(GT_label[i]).index(element) for element in chosed_label]
        chosed_eeg = eeg_data[i][indices,:]
        eeg.append(chosed_eeg)
        eeg_label.append(labels)
    eeg_label = np.stack(eeg_label, axis=0) #[6,40,5,62,5]
    eeg = np.stack(eeg, axis=0) #[6,40,5,62,5]
    eeg_label = torch.from_numpy(eeg_label)
    eeg = torch.from_numpy(eeg)
    eeg_label = rearrange(eeg_label, 'a b c e f -> (a b c) (e f)') #[1200,310]
    eeg = rearrange(eeg, 'a b c e f -> (a b c) (e f)') #[1200,310]
    normalize = preprocessing.StandardScaler()
    normalize.fit(eeg)
    eeg = normalize.transform(eeg)
    eeg_label = normalize.transform(eeg_label) 

    # 2. load CLIP text embeddings  (7×200×77×768) → (1200, 77*768)
    Text = []
    for i in range(1, 7):
        text_embedding = torch.load(f'text_embeddings/block{i}.pt') #[200,77,768]
        text = rearrange(text_embedding,'(a b) c d -> a b c d',a=40)
        indices = [list(GT_label[i-1]).index(element) for element in chosed_label]
        text = text[indices,:]
        Text.append(text)
    Text = torch.cat(Text,dim=0) #[240,5,77,768]
    Text = torch.reshape(Text, (-1, Text.shape[2]*Text.shape[3])) #[1200,77*768]
    
    model = CLIP()
    dataset = Dataset(eeg, Text) 
        
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    epoch_size = 200
    model=model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_size * len(dataloader))

    for epoch in tqdm(range(epoch_size)):
        model.train()
        epoch_loss = 0
        write_offset = 0
        torch.cuda.empty_cache()
        for i, batch in enumerate(dataloader):
            eeg, text = batch
            eeg = eeg.float().to(device)
            text_embeddings = text.float().to(device)
            optimizer.zero_grad()
            eeg_embeddings = model(eeg)
        
            loss = F.mse_loss(eeg_embeddings, text_embeddings)

            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        print(epoch_loss)

    model_dict = model.state_dict()

    torch.save({'state_dict': model_dict}, f'../checkpoints/Semantic/eeg2text.pt')
