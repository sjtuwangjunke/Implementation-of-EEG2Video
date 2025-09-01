import math
import random
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from copy import deepcopy
max_length = 16


class MyEEGNet_embedding(nn.Module):
    def __init__(self, d_model=128, C=62, T=200, F1=16, D=4, F2=16, cross_subject=False):
        super(MyEEGNet_embedding, self).__init__()
        if (cross_subject == True):
            self.drop_out = 0.25
        else:
            self.drop_out = 0.5

        self.block_1 = nn.Sequential(
            # Pads the input tensor boundaries with zero
            # left, right, up, bottom
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(
                in_channels=1,  # input shape (1, C, T)
                out_channels=F1,  # num_filters
                kernel_size=(1, 64),  # filter size
                bias=False
            ),  # output shape (F1, C, T)
            nn.BatchNorm2d(F1)  # output shape (F1, C, T)
        )

        # block 2 and 3 are implementations of Depthwise Convolution and Separable Convolution
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=F1,  # input shape (F1, C, T)
                out_channels=F1 * D,  # num_filters
                kernel_size=(C, 1),  # filter size
                groups=F1,
                bias=False
            ),  # output shape (F1 * D, 1, T)
            nn.BatchNorm2d(F1 * D),  # output shape (F1 * D, 1, T)
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  # output shape (F1 * D, 1, T//4)
            nn.Dropout(self.drop_out)  # output shape (F1 * D, 1, T//4)
        )

        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            # The Separable Convolution can be diveded into two steps
            # The first conv dosen't change the channels, only layers to layers respectively
            nn.Conv2d(
                in_channels=F1 * D,  # input shape (F1 * D, 1, T//4)
                out_channels=F1 * D,
                kernel_size=(1, 16),  # filter size
                groups=F1 * D,
                bias=False
            ),  # output shape (F1 * D, 1, T//4)
            # The second conv changes the channels, use 1x1 conv to combine channels' information
            nn.Conv2d(
                in_channels=F1 * D,  # input shape (F1 * D, 1, T//4)
                out_channels=F2,  # num_filters
                kernel_size=(1, 1),  # filter size
                bias=False
            ),  # output shape (F2, 1, T//4)
            nn.BatchNorm2d(F2),  # output shape (F2, 1, T//4)
            nn.ELU(),
            nn.AvgPool2d((1, 8)),  # output shape (F2, 1, T//32)
            nn.Dropout(self.drop_out)
        )

        self.embedding = nn.Linear(48, d_model)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)

        # print(x.shape)
        x = x.view(x.shape[0], -1)
        x = self.embedding(x)
        return x


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class myTransformer(nn.Module):

    def __init__(self, d_model=512):
        super(myTransformer, self).__init__()

        # 定义词向量，词典数为10。我们不预测两位小数。
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=d_model)
        self.img_embedding = nn.Linear(4 * 36 * 64, d_model)
        self.eeg_embedding = MyEEGNet_embedding(d_model=d_model, C=62, T=100, F1=16, D=4, F2=16, cross_subject=False)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True),
            num_layers=2
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=4, batch_first=True),
            num_layers=4
        )

        # 定义位置编码器
        self.positional_encoding = PositionalEncoding(d_model, dropout=0)

        self.txtpredictor = nn.Linear(512, 13)

        # 定义最后的线性层，这里并没有用Softmax，因为没必要。
        # 因为后面的CrossEntropyLoss中自带了
        self.predictor = nn.Linear(512, 4 * 36 * 64)

    def forward(self, src, tgt):
        # 对src和tgt进行编码
        # x = torch.rand(size=(32, 10, 62, 200))
        src = self.eeg_embedding(src.reshape(src.shape[0] * src.shape[1], 1, 62, 100)).reshape(src.shape[0], 7, -1)
        # print("src.shape = ", src.shape)

        tgt = tgt.reshape(tgt.shape[0], tgt.shape[1], tgt.shape[2] * tgt.shape[3] * tgt.shape[4])
        tgt = self.img_embedding(tgt)
        # print("tgt.shape = ", tgt.shape)
        # 给src和tgt的token增加位置信息
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # 生成mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-2]).to(tgt.device)
        # print("mask:", tgt_mask)
        # src_key_padding_mask = myTransformer.get_key_padding_mask(src)
        # tgt_key_padding_mask = myTransformer.get_key_padding_mask(tgt)

        # 使用 Transformer Encoder 对输入序列进行编码
        encoder_output = self.transformer_encoder(src)
        #print("en.shape = ", encoder_output.shape)

        # 使用 Transformer Decoder 对目标序列进行解码
        #print(tgt.shape)
        new_tgt = torch.zeros((tgt.shape[0], 1, tgt.shape[2])).cuda()
        for i in range(6):
            decoder_output = self.transformer_decoder(new_tgt, encoder_output, tgt_mask=tgt_mask[:i+1, :i+1])

            # print(new_tgt.shape)
            new_tgt = torch.cat((new_tgt, decoder_output[:, -1:, :]), dim=1)

        #decoder_output = self.transformer_decoder(tgt, encoder_output, tgt_mask=tgt_mask)

        #print("new_tgt.shape = ", new_tgt.shape)

        encoder_output = torch.mean(encoder_output, dim=1)
        # print(encoder_output.shape)

        return self.txtpredictor(encoder_output), self.predictor(new_tgt).reshape(new_tgt.shape[0],
                                                                                         new_tgt.shape[1], 4, 36,
                                                                                         64)


# criteria = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

def evaluate_accuracy_auto(net, data_iter, device):
    loss = nn.MSELoss()
    total_loss = 0
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    for X, y in data_iter:
        if isinstance(X, list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        y_auto = torch.zeros(y.shape[0], 1, 768).to(device)
        y_auto_hat = net(X, y_auto)
        for i in range(9):
            y_auto = torch.cat((y_auto, y_auto_hat[:, -1, :].reshape(y.shape[0], 1, 768)))
            y_auto_hat = net(X, y_auto)

        y_auto = torch.cat((y_auto, y_auto_hat[:, -1, :].reshape(y.shape[0], 1, 768)))
        y_auto = y_auto[:, 1:, :]
        print(y_auto.shape)
        test_loss = loss(y_auto, y)
        total_loss += test_loss.item()
    return total_loss / len(data_iter)

class Dataset():
    def __init__(self, eeg, video):
        self.eeg = eeg
        self.video = video
        self.len = eeg.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.eeg[item], self.video[item]


    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.eeg[item], self.video[item]


def loss(true, pred):
    l = nn.MSELoss()
    return l(true, pred)


def normalizetion(data):
    mean = torch.mean(data, dim=(0, 2, 3, 4), dtype=torch.float64)
    std = torch.std(data, dim=(0, 2, 3, 4))

    print("mean:", mean)
    print("std:", std)
    normalized_data = (data - mean.reshape(1, 4, 1, 1, 1)) / std.reshape(1, 4, 1, 1, 1)

    return normalized_data



chosed_label = [i for i in range(1,41)]
GT_label = np.array([[23, 22, 9, 6, 18, 14, 5, 36, 25, 19, 28, 35, 3, 16, 24, 40, 15, 27, 38, 33,
                      34, 4, 39, 17, 1, 26, 20, 29, 13, 32, 37, 2, 11, 12, 30, 31, 8, 21, 7, 10, ],
                     [27, 33, 22, 28, 31, 12, 38, 4, 18, 17, 35, 39, 40, 5, 24, 32, 15, 13, 2, 16,
                      34, 25, 19, 30, 23, 3, 8, 29, 7, 20, 11, 14, 37, 6, 21, 1, 10, 36, 26, 9, ],
                     [15, 36, 31, 1, 34, 3, 37, 12, 4, 5, 21, 24, 14, 16, 39, 20, 28, 29, 18, 32,
                      2, 27, 8, 19, 13, 10, 30, 40, 17, 26, 11, 9, 33, 25, 35, 7, 38, 22, 23, 6, ],
                     [16, 28, 23, 1, 39, 10, 35, 14, 19, 27, 37, 31, 5, 18, 11, 25, 29, 13, 20, 24,
                      7, 34, 26, 4, 40, 12, 8, 22, 21, 30, 17, 2, 38, 9, 3, 36, 33, 6, 32, 15, ],
                     [18, 29, 7, 35, 22, 19, 12, 36, 8, 15, 28, 1, 34, 23, 20, 13, 37, 9, 16, 30,
                      2, 33, 27, 21, 14, 38, 10, 17, 31, 3, 24, 39, 11, 32, 4, 25, 40, 5, 26, 6, ],
                     [29, 16, 1, 22, 34, 39, 24, 10, 8, 35, 27, 31, 23, 17, 2, 15, 25, 40, 3, 36,
                      26, 6, 14, 37, 9, 12, 19, 30, 5, 28, 32, 4, 13, 18, 21, 20, 7, 11, 33, 38],
                     [38, 34, 40, 10, 28, 7, 1, 37, 22, 9, 16, 5, 12, 36, 20, 30, 6, 15, 35, 2,
                      31, 26, 18, 24, 8, 3, 23, 19, 14, 13, 21, 4, 25, 11, 32, 17, 39, 29, 33, 27]
                     ])

if __name__ == "__main__":
    eegdata = np.load('/home/drink/SEED-DV/Segmented_Rawf_200Hz_2s/sub1.npy') #(7, 40, 5, 62, 400)
    chosed_index = []
    for i in range(7):
        index = [list(GT_label[i]).index(element) for element in chosed_label]
        chosed_index.append(index)
    new_eeg = np.zeros((7, 40, 5, 62, 400))
    for i in range(7):
        new_eeg[i] = eegdata[i][chosed_index[i], :, :, :]
    new_eeg = torch.from_numpy(new_eeg) #(7, 40, 5, 62, 400)
    
    # use VAE model to get the latent
    latent_data = np.load('train_latents.npy') 
    latent_data = torch.from_numpy(latent_data) #[1200, 4, 6, 36, 64]
    test_latent = torch.load('test_latents.pt') #[200, 4, 6, 36, 64]
    latent_data = rearrange(latent_data, "(g p d) c f h w -> g p d c f h w", g=6, p=40, d=5) #[6, 40, 5, 4, 6, 36, 64]
    new_latent = np.zeros((6, 40, 5, 4, 6, 36, 64))
    for i in range(6):
        new_latent[i] = latent_data[i][chosed_index[i], :, :, :, :, :]
    new_latent = rearrange(new_latent, "g p d c f h w -> (g p d) c f h w")
    new_latent = torch.from_numpy(new_latent) #(1200, 4, 6, 36, 64)
    
    window_size = 100
    overlap = 50
    EEG = []
    for i in range(0, new_eeg.shape[-1] - window_size + 1, window_size - overlap):
        EEG.append(new_eeg[..., i:i + window_size])
    EEG = torch.stack(EEG, dim=-1)
    test_eeg = EEG[6,:]
    EEG = EEG[0:6, :]
    EEG = torch.reshape(EEG, (EEG.shape[0] * EEG.shape[1] * EEG.shape[2], EEG.shape[3], EEG.shape[4], EEG.shape[5]))
    test_eeg = torch.reshape(test_eeg,(test_eeg.shape[0] * test_eeg.shape[1] , test_eeg.shape[2], test_eeg.shape[3], test_eeg.shape[4]))

    b,c,l,f = EEG.shape
    EEG = EEG.flatten(1)
    test_eeg = test_eeg.flatten(1)
    normalize = StandardScaler()
    normalize.fit(EEG)
    EEG = normalize.transform(EEG)
    test_eeg = normalize.transform(test_eeg)
    EEG = rearrange(EEG,'b (c l f) -> b c l f',c=c,l=l,f=f)
    test_eeg = rearrange(test_eeg,'b (c l f) -> b c l f',c=c,l=l,f=f)
    EEG = rearrange(EEG, "b c l f -> b f c l")
    test_eeg = rearrange(test_eeg, "b c l f -> b f c l")
    EEG = torch.from_numpy(EEG) #[1200, 7, 62, 100]
    test_eeg = torch.from_numpy(test_eeg) #[200, 7, 62, 100]
    latent_data = rearrange(new_latent, "b c f h w -> b f c h w") #[1200, 6, 4, 36, 64]
    test_latent = rearrange(test_latent, "b c f h w -> b f c h w") #[200, 6, 4, 36, 64]

    dataset = Dataset(EEG, latent_data)
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model_file='../checkpoints/seq2seqmodel.pt'
    model = myTransformer()
    model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage)['state_dict'])
    model =model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200 * len(train_dataloader))
    latent_out = None
    for epoch in tqdm(range(25)):
        model.train()
        epoch_loss = 0
        for i, batch in enumerate(train_dataloader):
            eeg, video = batch
            eeg = eeg.float().cuda()

            b, _, c, w, h = video.shape
            padded_video = torch.zeros((b, 1, c, w, h))
            full_video = torch.cat((padded_video, video), dim=1).float().cuda()
            optimizer.zero_grad()

            txt_label, out = model(eeg, full_video)
            # print(out[:,:-1,:].shape)
            # if epoch == 199:
            #     latent_out.append(out[:, :-1, :].cpu().detach().numpy())
            video = video.float().cuda()
            l = loss(video, out[:, :-1, :])
            l.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += l.item()
        print(epoch_loss)
    
    # inference
    model.eval()
    test_latent = test_latent.float().cuda()
    test_eeg = test_eeg.float().cuda()
    b, _, c, w, h = test_latent.shape
    padded_video = torch.zeros((b, 1, c, w, h)).float().cuda()
    full_video = torch.cat((padded_video, test_latent), dim=1).float().cuda()
    txt_label, out = model(test_eeg, full_video)
    latent_out = out[:, :-1, :].cpu().detach().numpy()
    latent_out = np.array(latent_out)
    print(latent_out.shape)
    np.save('latent_out_block7_40_classes.npy', latent_out)
    model_dict = model.state_dict()
    torch.save({'state_dict': model_dict}, f'../checkpoints/seq2seqmodel.pt')


