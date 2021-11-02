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


def extract_feature_pipeline(args):
    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

 
    if args.zip_mode:
        from datasets.zipdata import ZipData

        datapath_train = os.path.join(args.data_path, 'train.zip')
        data_map_train = os.path.join(args.data_path, 'train_map.txt')

        dataset_train = ZipData(
            datapath_train, data_map_train,
            transform
        )

        datapath_val = os.path.join(args.data_path, 'val.zip')
        data_map_val = os.path.join(args.data_path, 'val_map.txt')

        dataset_val = ZipData(
            datapath_val, data_map_val,
            transform
        )

    else:
        dataset_train = ReturnIndexDataset(os.path.join(args.data_path, "train"), transform=transform)
        dataset_val = ReturnIndexDataset(os.path.join(args.data_path, "val"), transform=transform)

    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

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
    print("Extracting features for train set...")
    train_features = extract_features(model, data_loader_train)
    print("Extracting features for val set...")
    test_features = extract_features(model, data_loader_val)

    if utils.get_rank() == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

    train_labels = torch.tensor([s[-1] for s in dataset_train.samples]).long() # samples是DatasetFolder的属性，List of (sample path, class_index) tuples
    test_labels = torch.tensor([s[-1] for s in dataset_val.samples]).long() # torch定义了7种CPU类型、8种GPU类型的tensor，见torch.Tensor，tensor转为某CPU类型直接.long()，转为某GPU类型直接.cuda().long()
    # save features and labels
    if args.dump_features and dist.get_rank() == 0:
        if not os.path.exists(args.dump_features):
            os.makedirs(args.dump_features)
        torch.save(train_features.cpu(), os.path.join(args.dump_features, "trainfeat.pth")) # torch.save不光可以保存模型，也可以保存tensor，其实就是采用pickle将保存对象序列化了
        torch.save(test_features.cpu(), os.path.join(args.dump_features, "testfeat.pth"))
        torch.save(train_labels.cpu(), os.path.join(args.dump_features, "trainlabels.pth"))
        torch.save(test_labels.cpu(), os.path.join(args.dump_features, "testlabels.pth"))
    return train_features, test_features, train_labels, test_labels


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
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=1000):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t() # (num_feat, batch)
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).cuda()
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ] # (batch, num_feat)
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features) # (batch, batch) test feat到train feat的相似度。torch.mm是矩阵a和b矩阵相乘，比如a的维度是(1, 2)，b的维度是(2, 3)，返回的就是(1, 3)的矩阵
        distances, indices = similarity.topk(k, largest=True, sorted=True) # test feat到相似度最高的k个train feat的相似度、train_idx，(batch, k), (batch, k)
        candidates = train_labels.view(1, -1).expand(batch_size, -1) # train_labels (N, ) -> view (1, N) -> expand (batch, N)。expand的-1表示那维不变。
        retrieved_neighbors = torch.gather(candidates, 1, indices) # topk的train label, (batch, k)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_() # topk train label的one-hot，(batch * k, num_classes)
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1) # 把retrieved_neighbors.view(-1, 1) (batch*k, 1)当成index，在retrieval_one_hot的dim=1上，去放置value=1
        distances_transform = distances.clone().div_(T).exp_() # exp(相似度/temperature), (batch, k)
        probs = torch.sum(
            torch.mul( #矩阵a和b对应位相乘，a和b的维度必须相等，比如a的维度是(1, 2)，b的维度是(1, 2)，返回的仍是(1, 2)的矩阵
                retrieval_one_hot.view(batch_size, -1, num_classes), # (batch, k, num_classes)
                distances_transform.view(batch_size, -1, 1), # (batch, k, 1)
            ),
            1,
        ) #（batch, num_classes)
        _, predictions = probs.sort(1, True) # return (sorted_tensor, sorted_indices), (batch, num_classes)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1)) # tensor.data相当于tensor.detach(), .eq之后(batch, num_classes)，其中num_classes里与targets一致的是1其他是0
        top1 = top1 + correct.narrow(1, 0, 1).sum().item() # narrow相当于切片，在dim=1维上，[0:1）的范围，(batch, 1) -> sum (1,) 如果sum不设置dim，就是整个全部求和
        top5 = top5 + correct.narrow(1, 0, 5).sum().item() # narrow相当于切片，在dim=1维上，[0:5）的范围，(batch, 5) -> sum (1,) 5个数里只有一个可能是1，也可能全0
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx # 将返回值从img,label变成了img,idx；label信息从dataset.samples里去捞了


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
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)

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

    if args.load_features:
        train_features = torch.load(os.path.join(args.load_features, "trainfeat.pth"))
        test_features = torch.load(os.path.join(args.load_features, "testfeat.pth"))
        train_labels = torch.load(os.path.join(args.load_features, "trainlabels.pth"))
        test_labels = torch.load(os.path.join(args.load_features, "testlabels.pth"))
    else:
        # need to extract features !
        train_features, test_features, train_labels, test_labels = extract_feature_pipeline(args)

    if utils.get_rank() == 0:
        if args.use_cuda:
            train_features = train_features.cuda()
            test_features = test_features.cuda()
            train_labels = train_labels.cuda()
            test_labels = test_labels.cuda()

        print("Features are ready!\nStart the k-NN classification.")

        log_stats = {}
        for k in args.nb_knn:
            top1, top5 = knn_classifier(train_features, train_labels,
                test_features, test_labels, k, args.temperature)
            print(f"{k}-NN classifier result: Top1: {top1}, Top5: {top5}")
            log_stats[k] = [top1, top5] 

        if utils.is_main_process():
            with (Path(args.dump_features) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    dist.barrier() #同步所有的进程, 直到整组(也就是所有节点的所有GPU)到达这个函数的时候, 才会执行后面的代码
