import torchvision
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import time
import numpy as np
import torch.nn.functional as F

from models.siamese_utils import Normalize, LFW_Pairs_Dataset, DDFA_Pairs_Dataset
from utils.verification import calculate_roc, calculate_accuracy, compute_roc, generate_roc_curve
from models.resfcn256 import ResFCN256

import scipy.io as sio
import os.path as osp
import os
import matplotlib.pylab as plt

class sia_net(nn.Module):
    def __init__(self , model):
        super(sia_net, self).__init__()
        self.fc1 = nn.Sequential(nn.Sequential(*list(model.children())[:-2]), nn.AdaptiveAvgPool2d(1))

        self.fc1_0 = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.Linear(1024, 512))

        self.fc1_1 = nn.Sequential(nn.Linear(2048, 62))
        
    def forward_once(self, x):
        x = self.fc1(x)

        x = x.view(x.size()[0], -1) 

        feature = self.fc1_0(x)     #feature

        param = self.fc1_1(x)

        return feature, param
    
    def forward(self, input_l, input_r):
        feature_l, param_l = self.forward_once(input_l)
        feature_r, param_r = self.forward_once(input_r)

        return feature_l, feature_r, param_l, param_r

def load_SPRNET():
	prnet = torchvision.models.resnet50()
	model = sia_net(prnet)
	
	return model

#region TOOLKIT
def transform_for_infer(image_shape):
    return transforms.Compose(
        [transforms.Resize(image_shape),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )

def sia_extract_feature(checkpoint_fp, root, pairs_txt, log_dir, device_ids = [0], batch_size = 32, num_workers = 8):
    map_location = {f'cuda:{i}': 'cuda:0' for i in range(8)}
    checkpoint = torch.load(checkpoint_fp, map_location=map_location)['state_dict']
    torch.cuda.set_device(device_ids[0])
    model = load_SPRNET()
    model = nn.DataParallel(model, device_ids=device_ids).cuda()
    model.load_state_dict(checkpoint)
    dataset = DDFA_Pairs_Dataset(root, pairs_txt, transform=transforms.Compose([transforms.ToTensor(), Normalize(mean=127.5, std=128)]))
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    cudnn.benchmark = True
    model.eval()

    # 6000 x 512
    embeddings_l = []
    embeddings_r = []
    pairs_match = []
    with torch.no_grad():
        for i, (inputs_l, inputs_r, matches) in enumerate(data_loader):
            inputs_l = inputs_l.cuda()
            inputs_r = inputs_r.cuda()
            feature_l, feature_r, param_l, param_r = model(inputs_l, inputs_r)
            param_l = param_l[:,12:52]
            param_r = param_r[:,12:52]

            param_l = param_l.div(torch.norm(param_l, p=2, dim=1, keepdim=True).expand_as(param_l))
            param_r = param_r.div(torch.norm(param_r, p=2, dim=1, keepdim=True).expand_as(param_r))

            for j in range(feature_l.shape[0]):
                feature_l_np = feature_l[j].cpu().numpy().flatten()
                feature_r_np = feature_r[j].cpu().numpy().flatten()
                param_l_np = param_l[j].cpu().numpy().flatten()
                param_r_np = param_r[j].cpu().numpy().flatten()
                matches_np = matches[j].cpu().numpy().flatten()

                embeddings_l.append(feature_l_np)
                embeddings_r.append(feature_r_np)
                pairs_match.append(matches_np)

    embeddings_l    = np.array(embeddings_l)
    embeddings_r    = np.array(embeddings_r)
    pairs_match     = np.array(pairs_match)

    pairs_match = pairs_match.reshape(6000)
    thresholds = np.arange(0, 4, 0.01)

    tpr, fpr, accuracy, best_thresholds = calculate_roc(thresholds, embeddings_l, embeddings_r, pairs_match, nrof_folds = 10, pca = 0)

    generate_roc_curve(fpr, tpr, log_dir)
    diff = np.subtract(embeddings_l, embeddings_r)
    dist = np.sum(np.square(diff), 1)

    return tpr, fpr, accuracy, best_thresholds

def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold', bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()
#endregion

if __name__ == '__main__':
    path            = osp.dirname(osp.abspath(__file__))
    checkpoint_fp   = osp.join(path, "training_debug","logs", "")
    root_ddfa       = osp.join(path, "data", "train_aug_120x120")
    pairs_txt       = osp.join(path, "test", "pairs_ddfa.txt")
    log_dir         = osp.join(path, "test", "cc.png")

    tpr, fpr, accuracy, best_thresholds = sia_extract_feature(checkpoint_fp, root_ddfa, pairs_txt, log_dir)
    mean_acc = np.mean(accuracy)
    print(mean_acc)