from tuneavideo.pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline
from tuneavideo.models.unet import UNet3DConditionModel
from tuneavideo.util import save_videos_grid
import torch
from tuneavideo.models.eeg_text import CLIP
import numpy as np
from einops import rearrange
from sklearn import preprocessing

pretrained_eeg_encoder_path = '/home/drink/EEG2Video/EEG2Video_New/checkpoints/Semantic/eeg2text_40_classes.pt'
model = CLIP()
model.load_state_dict(torch.load(pretrained_eeg_encoder_path, map_location=lambda storage, loc: storage)['state_dict'])
model.to(torch.device('cuda'))
model.eval()

eeg_data_path = ""                           # your own data path for eeg data
EEG_dim = 62*200                             # the dimension of an EEG segment
eegdata = np.load(eeg_data_path)
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
chosed_label = [i for i in range(1, 41)]

EEG = []
for i in range(6):
    indices = [list(GT_label[i]).index(element) for element in chosed_label]
    chosed_eeg = eegdata[i][indices,:]
    EEG.append(chosed_eeg)
EEG = np.stack(EEG, axis=0)

test_indices = [list(GT_label[6]).index(element) for element in chosed_label]
eeg_test = eegdata[6][test_indices, :]
eeg_test = torch.from_numpy(eeg_test)
eeg_test = rearrange(eeg_test, 'a b c d e -> (a b) c (d e)')
eeg_test = torch.mean(eeg_test, dim=1).resize(eeg_test.shape[0], EEG_dim)

EEG = torch.from_numpy(EEG)
print(EEG.shape)
# id = 1
# for i in range(40):
#     EEG[:,i,...] = id
#     id += 1
EEG = rearrange(EEG, 'a b c d e f -> (a b c) d (e f)')
print(EEG.shape)
EEG = torch.mean(EEG, dim=1).resize(EEG.shape[0], EEG_dim)

scaler = preprocessing.StandardScaler().fit(EEG)
EEG = scaler.transform(EEG)
EEG = torch.from_numpy(EEG).float().cuda()
eeg_test = scaler.transform(eeg_test)
eeg_test = torch.from_numpy(eeg_test).float().cuda()

pretrained_model_path = "/home/drink/huggingface/stable-diffusion-v1-4"
my_model_path = "/home/drink/EEG2Video/EEG2Video_New/Generation/outputs/40_classes_video_200_epoch"
unet = UNet3DConditionModel.from_pretrained(my_model_path, subfolder='unet', torch_dtype=torch.float16).to('cuda')
pipe = TuneAVideoPipeline.from_pretrained(pretrained_model_path, unet=unet, torch_dtype=torch.float16).to("cuda")
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_slicing()

# this are latents with DANA, these latents are pre-prepared by Seq2Seq model
latents_add_noise = torch.load('/home/drink/EEG2Video/EEG2Video_New/DANA/40_classes_latent_add_noise.pt')
#latents_add_noise = torch.from_numpy(latents_add_noise).half()
latents_add_noise = rearrange(latents_add_noise, 'a b c d e -> a c b d e')

# this are latents w/o DANA, these latents are pre-prepared by Seq2Seq model
latents = torch.load('/home/drink/EEG2Video/EEG2Video_New/checkpoints/Seq2Seq/latent_out_block7_40_classes.pt')
#latents = torch.from_numpy(latents).half()
latents = rearrange(latents, 'a b c d e -> a c b d e')
print(latents_add_noise.shape)
print(eeg_test.shape)

# Ablation, inference w/o Seq2Seq and w/o DANA
woSeq2Seq = True
woDANA = True

for i in range(0,200):
    if woSeq2Seq:
        video = pipe(model, eeg_test[i:i+1,...], latents=None, video_length=6, height=288, width=512, num_inference_steps=100, guidance_scale=12.5).videos
        savename = '40_Classes_woSeq2Seq'
    elif woDANA:
        video = pipe(model, eeg_test[i:i+1,...], latents=latents[i:i+1,...], video_length=6, height=288, width=512, num_inference_steps=100, guidance_scale=12.5).videos
        savename = '40_Classes_woDANA'
    else:
        video = pipe(model, eeg_test[i:i+1,...], latents=latents_add_noise[i:i+1,...], video_length=6, height=288, width=512, num_inference_steps=100, guidance_scale=12.5).videos
        savename = '40_Classes_Fullmodel'
    save_videos_grid(video, f"./{savename}/{i}.gif")
