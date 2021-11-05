# coding=utf-8

# Modified by Chunyuan Li (chunyl@microsoft.com)
#
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import argparse
from six.moves import cPickle

import pandas as pd
import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms

from torchvision import models as torchvision_models
torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

import json
from pathlib import Path
import utils
import models.vision_transformer as vits
from models.vision_transformer import DINOHead
from models import build_model

from config import config
from config import update_config
from config import save_config
    
import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import os
import my_transforms

class MPDataset(Dataset):
    """
    创建自己的数据集
    Note: 当 Dataset 创建好后并没有将数据生产出来，我们只是定义了数据及标签生产的流水线，只有在真正使用时，如手动调用 next(iter(train_dataset))，或被 DataLoader调用，才会触发数据集内部的 __getitem__() 函数来读取数据
    实现的逻辑: 读取本地图片、进行数据增强
    """
    def __init__(self, image_dir, transform = None):
        super(MPDataset, self).__init__()
        self.image_path = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
        self.transform = transform

    def __getitem__(self, idx):
        """
        获取对应idx的图像，并进行数据增强
        """
        image = Image.open(self.image_path[idx]) # 用PIL读取本地图片，取值范围[0, 255]，形状[W, H]
        if self.transform is not None:
            image_transform = self.transform(image) # 对图片进行数据增强
        else:
            image_transform = transforms.ToTensor()(image)  # 默认是转成tensor，取值范围变到[0，1.0]，形状[C, H, W]
        return image_transform, idx  # img, cls, idx, path

    def __len__(self):
        return len(self.image_path)

# transform = pth_transforms.Compose([
#         pth_transforms.Resize(256, interpolation=3),
#         pth_transforms.CenterCrop(224),
#         pth_transforms.ToTensor(),
#         pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#     ])
# dataset = MPDataset('./data/cifar10/train/airplane', transform=transform)
# print(len(dataset))
# for data in dataset:
#     print(data)
#     exit()


def extract_feature_pipeline(args):
    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = MPDataset(args.data_path, transform=transform)

    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} imgs.")

    # ============ building network ... ============
    # if the network is a 4-stage vision transformer (i.e. swin)
    if 'swin' in args.arch :
        update_config(config, args)
        model = build_model(config, is_teacher=True)

    # if the network is a 4-stage vision transformer (i.e. longformer)
    elif 'vil' in args.arch :
        update_config(config, args)
        model = build_model(config, is_teacher=True)

    # if the network is a 4-stage vision transformer (i.e. CvT)
    elif 'cvt' in args.arch :
        update_config(config, args)
        model = build_model(config, is_teacher=True)

    # if the network is a vision transformer (i.e. deit_tiny, deit_small, vit_base)
    elif args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)

    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()

    # ============ extract features ... ============
    print("Extracting features for image set...")
    train_features = extract_features(model, data_loader_train)

    if utils.get_rank() == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)

    train_paths = dataset_train.image_path
    
    print("train_paths len:", len(train_paths))
    print("train_features shape:", train_features.shape)
    
    # save features and labels
    if args.dump_features and dist.get_rank() == 0:
        if not os.path.exists(args.dump_features):
            os.makedirs(args.dump_features)
        train_features = train_features.cpu().float().numpy()
        
        # df = pd.DataFrame(data = train_features)
        # df.insert(0, 'name', train_paths)
        # df.to_csv(os.path.join(args.dump_features,'embeddings.txt'), sep=' ', header=None, index=False)
    return train_features, train_paths


@torch.no_grad() #不需要计算梯度（更快），也不会进行反向传播
def extract_features(model, data_loader):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    for samples, index in metric_logger.log_every(data_loader, 10): # samples, index是data_loader的元素，是img和它对应的索引idx，由metric_logger.log_every yield出的
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        feats = model(samples).clone()

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1]) # DataLoader有dataset属性，len(dataset)是dataset的样本数。对比之下，len(dataloader)是一个epoch里的batch个数，也即一轮的迭代次数
            if args.use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device) #torch.empty()返回填充有未初始化数据的张量，这里的shape=(1, batch)
        y_l = list(y_all.unbind(0)) #ubind移除指定维后，返回一个元组，包含了沿着指定维切片后的各个切片。也即，变成了[device1的batch idx tuple, device2的batch idx tuple, ...]
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True) # 把各个device里的tensor index集中到各个device都有的tensor list y_l中
        y_all_reduce.wait() # 对进程上锁，等待通信结束。在.wait() 执行之后，我们可以保证通信已经结束，所有index已经集中到y_l里了。
        index_all = torch.cat(y_l) # 得到所有index

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0)) 
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True) #将各个device里的feats，集中到每个device都有的output_l这个list
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if args.use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l)) # 在第0维，将torch.cat(output_l)[i]放在features的第index_all[i]位置上
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu()) # cpu
    return features              
      

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        type=str)

    parser.add_argument('--arch', default='deit_small', type=str,
        choices=['cvt_tiny', 'swin_tiny','swin_small', 'swin_base', 'swin_large', 'swin', 'vil', 'vil_1281', 'vil_2262', 'deit_tiny', 'deit_small', 'vit_base'] + torchvision_archs,
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using deit_tiny or deit_small.""")

    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,
        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.07, type=float,
        help='Temperature used in the voting coefficient')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--dump_features', default=None,
        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load_features', default=None, help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str)

    # Dataset
    parser.add_argument('--zip_mode', type=utils.bool_flag, default=False, help="""Whether or not
        to use zip file.""")


    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)    

    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True


    train_features, train_paths = extract_feature_pipeline(args)