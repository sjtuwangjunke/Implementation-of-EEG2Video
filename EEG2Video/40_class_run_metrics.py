import numpy as np
from transformers import pipeline
from typing import Callable, List, Optional, Union
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from PIL import Image
import torch
from einops import rearrange
from torchmetrics.functional import accuracy
import torch.nn.functional as F
from transformers import logging
from skimage.metrics import structural_similarity as ssim
import imageio
from PIL import Image


logging.set_verbosity_error()

class clip_score:
    predefined_classes = [
        'an image of people',
        'an image of a bird',
        'an image of a mammal',
        'an image of an aquatic animal',
        'an image of a reptile',
        'an image of buildings',
        'an image of a vehicle',
        'an image of a food',
        'an image of a plant',
        'an image of a natural landscape',
        'an image of a cityscape',
    ]
    
    def __init__(self, 
                device: Optional[str] = 'cuda',
                cache_dir: str = '.cache'
                ):
        self.device = device
        self.clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32", 
                                                                        cache_dir=cache_dir).to(device, torch.float16)
        self.clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=cache_dir)
        self.clip_model.eval()

    @torch.no_grad()
    def __call__(self, img1, img2):
        # img1, img2: w, h, 3
        # all in pixel values: 0 ~ 255
        # return clip similarity score
        img1 = self.clip_processor(images=img1, return_tensors="pt")['pixel_values'].to(self.device, torch.float16)
        img2 = self.clip_processor(images=img2, return_tensors="pt")['pixel_values'].to(self.device, torch.float16)

        img1_features = self.clip_model(img1).image_embeds.float()
        img2_features = self.clip_model(img2).image_embeds.float()
        return F.cosine_similarity(img1_features, img2_features, dim=-1).item()

def n_way_top_k_acc(pred, class_id, n_way, num_trials=40, top_k=1):
    if isinstance(class_id, int):
        class_id = [class_id]
    pick_range =[i for i in np.arange(len(pred)) if i not in class_id]
    corrects = 0
    for t in range(num_trials):
        idxs_picked = np.random.choice(pick_range, n_way-1, replace=False)
        for gt_id in class_id:
            pred_picked = torch.cat([pred[gt_id].unsqueeze(0), pred[idxs_picked]])
            pred_picked = pred_picked.argsort(descending=False)[-top_k:]
            if 0 in pred_picked:
                corrects += 1
                break
    return corrects / num_trials, np.sqrt(corrects / num_trials * (1 - corrects / num_trials) / num_trials)

@torch.no_grad()
def img_classify_metric(
                        pred_videos: np.array, 
                        gt_videos: np.array,
                        n_way: int = 50,
                        num_trials: int = 100,
                        top_k: int = 1,
                        cache_dir: str = '.cache',
                        device: Optional[str] = 'cuda',
                        return_std: bool = False
                        ):
    # pred_videos: n, 256, 256, 3 in pixel values: 0 ~ 255
    # gt_videos: n, 256, 256, 3 in pixel values: 0 ~ 255
    assert n_way > top_k
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224', 
                                                  cache_dir=cache_dir)
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', 
                                                      cache_dir=cache_dir).to(device, torch.float16)
    model.eval()
    
    acc_list = []
    std_list = []
    for pred, gt in zip(pred_videos, gt_videos):
        pred = processor(images=pred.astype(np.uint8), return_tensors='pt')
        gt = processor(images=gt.astype(np.uint8), return_tensors='pt')
        gt_class_id = model(**gt.to(device, torch.float16)).logits.argsort(-1,descending=False).detach().flatten()[-3:]
        pred_out = model(**pred.to(device, torch.float16)).logits.softmax(-1).detach().flatten()

        acc, std = n_way_top_k_acc(pred_out, gt_class_id, n_way, num_trials, top_k)
        acc_list.append(acc)
        std_list.append(std)
    if return_std:
        return acc_list, std_list
    return acc_list

@torch.no_grad()
def video_classify_metric(
                        pred_videos: np.array, 
                        gt_videos: np.array,
                        n_way: int = 50,
                        num_trials: int = 100,
                        top_k: int = 1,
                        num_frames: int = 6,
                        cache_dir: str = '.cache',
                        device: Optional[str] = 'cuda',
                        return_std: bool = False
                        ):
    # pred_videos: n, 6, 256, 256, 3 in pixel values: 0 ~ 255
    # gt_videos: n, 6, 256, 256, 3 in pixel values: 0 ~ 255
    assert n_way > top_k
    processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base-finetuned-kinetics', 
                                                         cache_dir=cache_dir)
    model = VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-base-finetuned-kinetics', num_frames=num_frames,
                                                           cache_dir=cache_dir).to(device, torch.float16)
    model.eval()

    acc_list = []
    std_list = []
 
    for pred, gt in zip(pred_videos, gt_videos):
        pred = processor(list(pred), return_tensors='pt')
        gt = processor(list(gt), return_tensors='pt')
        gt_class_id = model(**gt.to(device, torch.float16)).logits.argsort(-1,descending=False).detach().flatten()[-3:]
        pred_out = model(**pred.to(device, torch.float16)).logits.softmax(-1).detach().flatten()

        acc, std = n_way_top_k_acc(pred_out, gt_class_id, n_way, num_trials, top_k)
        acc_list.append(acc)
        std_list.append(std)
    if return_std:
        return acc_list, std_list
    return acc_list
    

def n_way_scores(
                pred_videos: np.array, 
                gt_videos: np.array,
                n_way: int = 50,
                top_k: int = 1,
                num_trials: int = 10,
                cache_dir: str = '.cache',
                device: Optional[str] = 'cuda',):
    # pred_videos: n, 256, 256, 3 in pixel values: 0 ~ 255
    # gt_videos: n, 256, 256, 3 in pixel values: 0 ~ 255
    assert n_way > top_k
    clip_calculator = clip_score(device, cache_dir)

    corrects = []
    for idx, pred in enumerate(pred_videos):
        gt = gt_videos[idx]
        gt_score = clip_calculator(pred, gt)
        rest = np.stack([img for i, img in enumerate(gt_videos) if i != idx])
        correct_count = 0
     
        for _ in range(num_trials):
            n_imgs_idx = np.random.choice(len(rest), n_way-1, replace=False)
            n_imgs = rest[n_imgs_idx]
            score_list = [gt_score]
            for comp in n_imgs:
                comp_score = clip_calculator(pred, comp)
                score_list.append(comp_score)
            correct_count += 1 if 0 in np.argsort(score_list)[-top_k:] else 0
        corrects.append(correct_count / num_trials)
    return corrects

def clip_score_only(
                pred_videos: np.array, 
                gt_videos: np.array,
                cache_dir: str = '.cache',
                device: Optional[str] = 'cuda',
                ):
    # pred_videos: n, 256, 256, 3 in pixel values: 0 ~ 255
    # gt_videos: n, 256, 256, 3 in pixel values: 0 ~ 255
    clip_calculator = clip_score(device, cache_dir)
    scores = []
    for pred, gt in zip(pred_videos, gt_videos):
        scores.append(clip_calculator(pred, gt))
    return np.mean(scores)

def channel_last(img):
    if img.shape[-1] == 3:
        return img
    if len(img.shape) == 3:
        img = rearrange(img, 'c h w -> h w c')
    elif len(img.shape) == 4:
        img = rearrange(img, 'f c h w -> f h w c')
    else:
        raise ValueError(f'img shape should be 3 or 4, but got {len(img.shape)}')
    return img

def mse_score_only(
                pred_videos: np.array, 
                gt_videos: np.array,
                **kwargs
                ):
    # pred_videos: n, 256, 256, 3 in pixel values: 0 ~ 255
    # gt_videos: n, 256, 256, 3 in pixel values: 0 ~ 255
    scores = []
    for pred, gt in zip(pred_videos, gt_videos):
        scores.append(mse_metric(pred, gt))
    return np.mean(scores), np.std(scores)

def ssim_score_only(
                pred_videos: np.array, 
                gt_videos: np.array,
                **kwargs
                ):
    # pred_videos: n, 256, 256, 3 in pixel values: 0 ~ 255
    # gt_videos: n, 256, 256, 3 in pixel values: 0 ~ 255
    pred_videos = channel_last(pred_videos)
    gt_videos = channel_last(gt_videos)
    scores = []
    for pred, gt in zip(pred_videos, gt_videos):
        scores.append(ssim_metric(pred, gt))
    return np.mean(scores), np.std(scores)

import torch.nn.functional as F

def mse_metric(img1, img2):
    return F.mse_loss(torch.FloatTensor(img1/255.0), torch.FloatTensor(img2/255.0), reduction='mean').item()

def ssim_metric(img1, img2):
    return ssim(img1, img2, data_range=255, channel_axis=-1)



def remove_overlap(
                pred_videos: np.array,
                gt_videos: np.array,
                scene_seg_list: List,
                get_scene_seg: bool=False,
                ):
    # pred_videos: 5 * 240, 6, 256, 256, 3 in pixel values: 0 ~ 255
    # gt_videos: 5 * 240, 6, 256, 256, 3 in pixel values: 0 ~ 255
    # scene_seg_list: 5 * 240
    pred_list = []
    gt_list = []
    seg_dict = {}
    for pred, gt, seg in zip(pred_videos, gt_videos, scene_seg_list):
        if '-' not in seg:
            if get_scene_seg:
                if seg not in seg_dict.keys():
                    seg_dict[seg] = seg
                    pred_list.append(pred)
                    gt_list.append(gt)
            else:
                pred_list.append(pred)
                gt_list.append(gt)
    return np.stack(pred_list), np.stack(gt_list) 

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
indices = [list(GT_label[6]).index(element) for element in range(1,41)]
print(indices)

video_2way_acc = []
video_40way_acc = []
image_2way_acc = []
image_40way_acc = []
image_ssim = []
for i in range(200):
    class_id = i // 5
    video_clip_id = i % 5
    gt_video_id = indices[class_id] * 5 + video_clip_id
    print("gt_video_id = ", gt_video_id)
    gt_video = imageio.mimread('../data/small_video_3fps/test_video_gif/'+str(gt_video_id+1)+'.gif')
    pred_video = imageio.mimread('./final_200_results/40_classes_eeg_'+str(i)+'.gif')
    gt_video = np.concatenate(gt_video).reshape(6, 288, 512, 3)
    pred_video = np.concatenate(pred_video).reshape(6, 288, 512, 3)

    print("gt_video.shape = ", gt_video.shape)

    mean, std = ssim_score_only(gt_video, pred_video)
    print(mean, std)
    image_ssim.append(mean)

    print("gt_video.shape = ", gt_video.shape)

    acc = img_classify_metric(
                            pred_videos=pred_video,
                            gt_videos=gt_video,
                            n_way = 40,
                            num_trials = 100,
                            top_k = 1,
                            cache_dir = '.cache',
                            device = 'cuda',
                            return_std = False
                            )
    print("image_40way_acc = ", acc)
    for acci in acc:
        image_40way_acc.append(acc)
    
    acc = img_classify_metric(
                            pred_videos=pred_video,
                            gt_videos=gt_video,
                            n_way = 2,
                            num_trials = 100,
                            top_k = 1,
                            cache_dir = '.cache',
                            device = 'cuda',
                            return_std = False
                            )
    print("image_2way_acc = ", acc)
    for acci in acc:
        image_2way_acc.append(acc)

    gt_video = gt_video.reshape((1,) + gt_video.shape)
    pred_video = pred_video.reshape((1,) + pred_video.shape)
    acc = video_classify_metric(
                            pred_videos=pred_video,
                            gt_videos=gt_video,
                            n_way = 40,
                            num_trials = 100,
                            top_k = 1,
                            cache_dir = '.cache',
                            device = 'cuda',
                            return_std = False
                            )

    print("video_40way_acc = ", acc)
    for acci in acc:
        video_40way_acc.append(acc)
    
    acc = video_classify_metric(
                            pred_videos=pred_video,
                            gt_videos=gt_video,
                            n_way = 2,
                            num_trials = 100,
                            top_k = 1,
                            cache_dir = '.cache',
                            device = 'cuda',
                            return_std = False
                            )

    print("video_2way_acc = ", acc)
    for acci in acc:
        video_2way_acc.append(acc)



print("video_2way_acc = ", np.mean(np.array(video_2way_acc)), np.std(np.array(video_2way_acc)))
print("video_40way_acc = ", np.mean(np.array(video_40way_acc)), np.std(np.array(video_40way_acc)))
print("image_2way_acc = ", np.mean(np.array(image_2way_acc)), np.std(np.array(image_2way_acc)))
print("image_40way_acc = ", np.mean(np.array(image_40way_acc)), np.std(np.array(image_40way_acc)))
print("image_ssim = ", np.mean(np.array(image_ssim)), np.std(np.array(image_ssim)))

np.save("./final_200_results/video_2way_acc.npy", np.array(video_2way_acc))
np.save("./final_200_results/video_40way_acc.npy", np.array(video_40way_acc))
np.save("./final_200_results/image_2way_acc.npy", np.array(image_2way_acc))
np.save("./final_200_results/image_40way_acc.npy", np.array(image_40way_acc))
np.save("./final_200_results/image_ssim.npy", np.array(image_ssim))
