import numpy as np
import matplotlib.pyplot as plt
import time

import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils import data
from torch.nn.parameter import Parameter
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
# import scikitplot as skplt
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from einops import rearrange
import models
import time
###################################### Hyperparameters ###################################
batch_size = 256
num_epochs = 500
lr = 0.001   # learning rate
C = 62       # the number of channels
T = 5        # the time samples of EEG signals
output_dir = './output_dir/'
network_name = "GLMNet_mlp"
saved_model_path = output_dir + network_name + '_40c.pth'

run_device = "cuda"

##########################################################################################
starttimer = time.time()
def my_normalize(data_array):
    data_array = data_array.reshape(data_array.shape[0], C*T)
    normalize = StandardScaler()
    normalize.fit(data_array)
    return (normalize.transform(data_array)).reshape(data_array.shape[0], C, T)

def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    print('%s Accuracy:%.3f' % (tip, np.mean(acc)))

def Get_subject(f):
    if(f[1] == '_'):
        return int(f[0])
    return int(f[0])*10 + int(f[1])

def Get_Dataloader(datat, labelt, istrain, batch_size):
    features = torch.tensor(datat, dtype=torch.float32)
    labels = torch.tensor(labelt, dtype=torch.long)
    return data.DataLoader(data.TensorDataset(features, labels), batch_size, shuffle=istrain)

class Accumulator:  #@save
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Timer:
    def __init__(self):
        self.times = []
        self.start()
    def start(self):
        self.tik = time.time()
    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    def avg(self):
        return sum(self.times) / len(self.times)
    def sum(self):
        return sum(self.times)
    def cumsum(self):
        return np.array(self.times).cumsum().tolist()

def cal_accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, axis=1)
    cmp = (y_hat == y)
    return torch.sum(cmp, dim=0)

def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)
    for X, y in data_iter:
        if isinstance(X, list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(cal_accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def topk_accuracy(output, target, topk=(1, )):       
    # output.shape (bs, num_classes), target.shape (bs, )
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size).item())
        return res

#训练函数
def train(net, train_iter, val_iter, test_iter, num_epochs, lr, device, save_path):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0)
    loss = nn.CrossEntropyLoss()
    
    timer, num_bathces = Timer(), len(train_iter)
    
    best_test_acc = -1
    
    best_model = net
    
    best_val_acc, best_test_acc = 0, 0
    
    for epoch in range(num_epochs):
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            metric.add(l * X.shape[0], cal_accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            # if(i + 1) % (num_bathces // 5) == 0 or i == num_bathces:
            #     print(f'{epoch + (i+1) / num_bathces} : , train_l:{train_l}, train_acc:{train_acc}')
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        
        val_acc = evaluate_accuracy_gpu(net, val_iter)
        if(val_acc > best_val_acc):
            best_val_acc = val_acc
            torch.save(net, save_path)
        
        if(epoch % 3 == 0):        
            print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, val acc {val_acc:.3f},'f'test acc {test_acc:.3f}')
            print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec 'f'on {str(device)}')
            # print('Saving model!')
            # # Save model to specified path
            # torch.save(net.state_dict(), './NeuralNetworkLvBaoliang/CNN_hw3/' + 'Alexnet.pth')
    # test_pred = np.array([])
    # for X, y in test_iter:
    #     X, y = X.to(device), y.to(device)
    #     y_hat = net(X)
    #     y_hat = torch.argmax(y_hat, axis=1)
    #     y_hat = y_hat.cpu().numpy()
    #     y_hat = y_hat.reshape(y_hat.shape[0])
    #     test_pred = np.concatenate((test_pred,y_hat))    
    
    # torch.save(net, save_path)
    return best_val_acc   


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
GT_label = GT_label - 1
All_label = np.empty((0, 400))
for block_id in range(7):
    All_label = np.concatenate((All_label, GT_label[block_id].repeat(10).reshape(1, 400)))

def get_files_names_in_directory(directory):
    files_names = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            files_names.append(filename)
    return files_names

sub_list = get_files_names_in_directory("/home/drink/SEED-DV/DE_1per1s/")

All_sub_top1 = []
All_sub_top5 = []

for subname in sub_list:
    load_npy = np.load("/home/drink/SEED-DV/DE_1per1s/" + subname)

    print(load_npy.shape)

    all_test_label = np.array([])
    all_test_pred = np.array([])

    print(load_npy.shape)

    print("shape = ", load_npy.shape)

    All_train = rearrange(load_npy, 'a b c d e f -> a (b c d) e f')
    print(All_train.shape)

    Top_1 = []
    Top_K = []

    for test_set_id in range(7):
        val_set_id = test_set_id - 1
        if(val_set_id < 0):
            val_set_id = 6
        train_data = np.empty((0, 62, 5))
        train_label = np.empty((0))
        for i in range(7):
            if(i == test_set_id):
                continue
            train_data = np.concatenate((train_data, All_train[i].reshape(400, 62, 5)))
            train_label = np.concatenate((train_label, All_label[i]))
        test_data = All_train[test_set_id]
        test_label = All_label[test_set_id]
        val_data = All_train[val_set_id]
        val_label = All_label[val_set_id]

        train_data = train_data.reshape(train_data.shape[0], 62*5)
        test_data = test_data.reshape(test_data.shape[0], 62*5)
        val_data = val_data.reshape(val_data.shape[0], 62*5)

        #对训练数据和测试数据归一化
        normalize = StandardScaler()
        normalize.fit(train_data)
        train_data = normalize.transform(train_data)  #分别进行归一化
        normalize = StandardScaler()
        normalize.fit(test_data)
        test_data = normalize.transform(test_data)
        normalize = StandardScaler()
        normalize.fit(val_data)
        val_data = normalize.transform(val_data)
            
        modelnet = models.glfnet_mlp(out_dim=40, emb_dim=64, input_dim=310)
        # backdoor_net = nn.Sequential(nn.Flatten(), nn.Linear(200, 256), nn.ReLU(),
        #                              nn.Linear(256, 256), nn.ReLU(),
        #                              nn.Linear(256, 256), nn.ReLU(),
        #                              nn.Linear(256, 5))

        # norm_backdoor_train_data = my_normalize(backdoor_data).reshape(backdoor_data.shape[0], 1, C, T)
        # norm_test_data = my_normalize(test_data).reshape(test_data.shape[0], 1, C, T)

        norm_train_data = train_data.reshape(train_data.shape[0], C, T)
        norm_test_data = test_data.reshape(test_data.shape[0], C, T)
        norm_val_data = val_data.reshape(val_data.shape[0], C, T)
        train_iter = Get_Dataloader(norm_train_data, train_label, istrain=True, batch_size=batch_size)
        test_iter = Get_Dataloader(norm_test_data, test_label, istrain=False, batch_size=batch_size)
        val_iter = Get_Dataloader(norm_val_data, val_label, istrain=False, batch_size=batch_size)

        now_pred = np.array([])

        accu = train(modelnet, train_iter, val_iter, test_iter, num_epochs, lr, run_device, save_path=saved_model_path)
        
        print("acc : =", accu)

        loaded_model = torch.load(saved_model_path, weights_only=False)
        loaded_model.to(run_device)

        block_top_1 = []
        block_top_k = []
        for X, y in test_iter:
            X, y = X.to(run_device), y.to(run_device)
            y_hat = loaded_model(X)

            top_K_results = topk_accuracy(y_hat, y, topk=(1,5))
            block_top_1.append(top_K_results[0])
            block_top_k.append(top_K_results[1])

            y_hat = torch.argmax(y_hat, axis=1)
            y_hat = y_hat.cpu().numpy()
            y_hat = y_hat.reshape(y_hat.shape[0])
            all_test_pred = np.concatenate((all_test_pred,y_hat))    
        all_test_label = np.concatenate((all_test_label, test_label))

        top_1_acc = np.mean(np.array(block_top_1))
        top_k_acc = np.mean(np.array(block_top_k))
        print("top1_acc = ", top_1_acc)
        print("top5_acc = ", top_k_acc)
        Top_1.append(top_1_acc)
        Top_K.append(top_k_acc)
        
        # break

    print(metrics.classification_report(all_test_label, all_test_pred))
    np.seterr(divide='ignore', invalid='ignore')

    proper = [0] * 40
    num = [0] * 40
    conf = np.zeros(shape=(40, 40))
    for i in range(all_test_label.shape[0]):
        l = int(all_test_label[i])
        p = int(all_test_pred[i])
        num[l] = num[l] + 1
        if(l == p):
            proper[l] = proper[l] + 1
        conf[l][p] = conf[l][p] + 1

    print('confusion_matrix=eev======================================')
    print(conf)
    print(proper)
    print(num)

    print("test_accu = ", np.sum(np.array(proper)) / np.sum(np.array(num)))
    print("test_Top_1_accu = ", np.mean(np.array(Top_1)))
    print("test_Top_5_accu = ", np.mean(np.array(Top_K)))
    All_sub_top1.append(np.mean(np.array(Top_1)))
    All_sub_top5.append(np.mean(np.array(Top_K)))

    save_results = np.concatenate((all_test_pred.reshape(1, all_test_label.shape[0]), all_test_label.reshape(1, all_test_label.shape[0])))
    print(save_results.shape)

    np.save('./ClassificationResults/40c_top1/'+network_name+'_Predict_Label_' + subname, save_results)

    # break

print(All_sub_top1)
print(All_sub_top5)

print("\nTOP1: ", np.mean(np.array(All_sub_top1)), np.std(np.array(All_sub_top1)))
print("TOP5: ", np.mean(np.array(All_sub_top5)), np.std(np.array(All_sub_top5)))
print(time.time() - starttimer)
np.save('./ClassificationResults/40c_top1/'+network_name+'_All_subject_acc.npy', np.array(All_sub_top1))
np.save('./ClassificationResults/40c_top5/'+network_name+'_All_subject_acc.npy', np.array(All_sub_top5))