import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn import preprocessing
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange


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
            # nn.BatchNorm1d(50000),
            nn.ReLU(),
            # nn.Linear(10000, 10000),
            # nn.ReLU(),
            nn.Linear(10000, 77 * 768)
        )

    def forward(self, eeg):
        eeg_embeddings = self.mlp(eeg)
          # shape: (batch_size)
        return eeg_embeddings


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

class Dataset():
    def __init__(self, eeg, text):

        scaler = preprocessing.StandardScaler().fit(eeg)
        eeg = scaler.transform(eeg)

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

# chosed_label = [1, 10, 12, 16, 19, 23, 25, 31, 34, 39]
chosed_label = [i for i in range(1, 41)]              # set subset for training semantic predictor
if __name__ == '__main__':
    eeg_data_path = "sub1.npy"                        # your own data path for eeg data
    text_embedding_path = "text_embedding.npy"        # your own data path for text embedding
    eegdata = np.load(eeg_data_path)
    text_embedding = np.load(text_embedding_path)     
    print(eegdata.shape)
    EEG = []
    for i in range(6):
        indices = [list(GT_label[i]).index(element) for element in chosed_label]
        chosed_eeg = eegdata[i][indices,:]
        EEG.append(chosed_eeg)
    EEG = np.stack(EEG, axis=0)

    EEG = torch.from_numpy(EEG)
    print("EEG.shape = ", EEG.shape)
    EEG = rearrange(EEG, 'a b c d e f -> (a b c) d (e f)')


    print("after arrange EEG.shape = ", EEG.shape)
        
    print(EEG)
    print("text_embedding.shape = ", text_embedding.shape)

    Text = []
    for i in range(6):
        # Text.append(text_embedding[:30,...])
        Text.append(text_embedding[:150,...])
    Text = np.concatenate(Text)

    print("Text.shape = ", Text.shape)

    Text = torch.from_numpy(Text)
    Text = torch.reshape(Text, (Text.shape[0], Text.shape[1]*Text.shape[2]))
    EEG = torch.mean(EEG, dim=1).resize(EEG.shape[0], 310)
    print(EEG)
    print(EEG.shape)
    print(Text.shape)


    model = CLIP()
    model_file = '/home/EEG2Video/Tune-A-Video/tuneavideo/models/semantic_predictor.pt'
    model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage)['state_dict'])
    model.cuda()

    dataset = Dataset(EEG, Text)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200 * len(dataloader))

    for epoch in tqdm(range(200)):
        model.train()
        epoch_loss = 0
        for i, batch in enumerate(dataloader):
            eeg, text = batch
            eeg = eeg.float().cuda()
            text_embeddings = text.float().cuda()
            optimizer.zero_grad()
            eeg_embeddings = model(eeg)

            loss = F.mse_loss(eeg_embeddings, text_embeddings)

            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        print(epoch_loss)

    model_dict = model.state_dict()
    torch.save({'state_dict': model_dict}, 'semantic_predictor.pt')