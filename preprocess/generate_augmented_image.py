import argparse
import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import os
import random
import shutil
import my_transforms


def select_image_query(args):
    # 先mv 1000张
    image_paths = os.listdir(args.image_dir)  # 取图片的原始路径
    file_num = len(image_paths)
    rate = args.query_num / file_num  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    pick_num = int(file_num * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(image_paths, pick_num)  # 随机选取picknumber数量的样本图片
    print("will select %d images" % len(sample))
    if not os.path.exists(args.image_query_dir):
        os.makedirs(args.image_query_dir)
    for name in sample:
        shutil.move(os.path.join(args.image_dir, name), os.path.join(args.image_query_dir, name))
    
    # 再copy 1000张
    image_paths = os.listdir(args.image_dir)  # 取图片的原始路径
    file_num = len(image_paths)
    rate = args.query_num / file_num  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    pick_num = int(file_num * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(image_paths, pick_num)  # 随机选取picknumber数量的样本图片
    print("will select %d images" % len(sample))
    if not os.path.exists(args.image_query_dir):
        os.makedirs(args.image_query_dir)
    for name in sample:
        shutil.copy(os.path.join(args.image_dir, name), os.path.join(args.image_query_dir, name))
        
    # check
    print("select done, there are %d images in %s, %d images in %s" % 
            (
              len(os.listdir(args.image_query_dir)), 
               args.image_query_dir, 
               len(os.listdir(args.image_dir)), 
               args.image_dir
            )
         )
    # select done, there are 2000 images in ../data/coverimage/image_query/, 196864 images in ../data/coverimage/image/


class MPDataset(Dataset):
    """
    创建自己的数据集
    Note: 当 Dataset 创建好后并没有将数据生产出来，我们只是定义了数据及标签生产的流水线，只有在真正使用时，如手动调用 next(iter(train_dataset))，或被 DataLoader调用，才会触发数据集内部的 __getitem__() 函数来读取数据
    实现的逻辑: 读取本地图片、进行数据增强
    """
    def __init__(self, image_dir, transform = None):
        super(MPDataset, self).__init__()
        self.image_dir = image_dir
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
        return transforms.ToTensor()(image), image_transform, self.image_path[idx] # 返回原图、增广后的图、原图文件名

    def __len__(self):
        return len(self.image_path)

def augment_method():
    """
    定义各数据增强方案
    """
    # 尺寸缩放
    resize = transforms.Resize(400)
    # 中心裁剪
    center_crop = [transforms.Resize(400), transforms.CenterCrop([300, 300])]
    # 随机裁剪
    random_crop = [transforms.Resize(400), transforms.RandomCrop([300, 300])]
    # 随机长宽比裁剪
    random_resized_crop = [transforms.Resize(300), transforms.RandomResizedCrop(400,
                                                                                scale=(0.8, 1.0),
                                                                                ratio=(0.8, 1.0),
                                                                                interpolation=2)]
    # 依概率p水平翻转
    h_flip = transforms.RandomHorizontalFlip(1.0)
    # 依概率p垂直翻转
    v_flip = transforms.RandomVerticalFlip(1.0)
    # 随机旋转
    random_rotation = transforms.RandomRotation(10)

    # 图像填充，上下黑边框
    # 例如这种手机截图：http://mmbiz.qpic.cn/mmbiz/03I0wy5Q9xgFgGyyOhR7z3BpORibnqmlE9qicicdVPy4VbVbHBd8WWmpCVE5lzib8Vme06lUMIVQpahA4sF0UCJSibw/0?wx_fmt=jpeg
    pad = transforms.Pad(padding=(0,80,0,80), fill=0, padding_mode='constant')

    # 调整亮度、对比度和饱和度
    color_jitter = transforms.ColorJitter(brightness=0.9,
                                        contrast=0.8,
                                        saturation=0.1,
                                        hue=0.1)
    # 转成灰度图
    gray = transforms.Grayscale(num_output_channels = 3)

    # 仿射变换
    random_affine = transforms.RandomAffine(degrees=0, translate=(0.05, 0.1), scale=(1, 1), shear=3)

    # 透视变换
    random_perspective = transforms.RandomPerspective(distortion_scale=0.3, p=1)

    # 高斯平滑
    gaussian_blur = transforms.GaussianBlur(3, sigma=(0.1, 2.0))

    # 加椒盐噪声
    add_salt_pepper_noise = my_transforms.AddSaltPepperNoise(0.005)

    # 加高斯噪声
    add_gaussian_noise = my_transforms.AddGaussianNoise(mean=0, variance=1, amplitude=3)

    ####是针对tensor的
    random_erasing = transforms.RandomErasing(p=1, scale=(0.02, 0.02), ratio=(1, 1), value='random', inplace=False)

    augment_method_dict = {
        'resize': resize,
        'center_crop': center_crop,
        'random_crop': random_crop,
        'random_resized_crop': random_resized_crop,
        'h_flip': h_flip,
        'v_flip': v_flip,
        'random_rotation': random_rotation,
        'pad': pad,
        'color_jitter': color_jitter,
        'gray': gray,
        'random_affine': random_affine,
        'random_perspective': random_perspective,
        'gaussian_blur': gaussian_blur,
        'random_erasing': random_erasing,
        'add_salt_pepper_noise': add_salt_pepper_noise,
        'add_gaussian_noise': add_gaussian_noise
    }
    return augment_method_dict


def image_save(image_tensor_1, image_tensor_2, path, args):
    PILImage_1 = transforms.ToPILImage()(image_tensor_1)
    # plt.imshow(PILImage_1)
    # plt.show()
    PILImage_2 = transforms.ToPILImage()(image_tensor_2)
    # plt.imshow(PILImage_2)
    # plt.show()
    image_path = path.replace(args.augment_image_dir_change_pos, '/'+args.augment_method+'/')
    # print("image_path:", image_path)
    dir_path = os.path.split(image_path)[0]
    # print("dir_path:", dir_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    PILImage_2.save(image_path)

def generate_inputfile_for_C(args):
    image_names = os.listdir(args.image_query_dir)
    image_names = [os.path.join(args.image_query_dir, x)+'\n' for x in image_names]
    with open(args.image_query_dir + '.txt', mode='w') as f:
        f.writelines(image_names)

def arg_parser():
    parser = argparse.ArgumentParser('generate_augmented_image')
    
    parser.add_argument('--func', type=str, default='generate_image_augment',
                        help='choice: select_image_query, generate_image_augment, generate_inputfile_for_C')
    parser.add_argument('--query_num', type=int, default=1000,
                        help='how many query images choose from image_dir to image_query_dir')
    parser.add_argument('--image_dir', type=str, default='../data/coverimage/image/',
                        help='image dir, 200000 images')
    parser.add_argument('--image_query_dir', type=str, default='../data/coverimage/image_query/', 
                        help='image query dir, 2000 images')
    parser.add_argument('--augment_image_dir_change_pos', type=str, default='/image_query/',
                        help='to generate augment image dir, which part need to be changed')

    args = parser.parse_args()
    return args

def main():
    """
    生成各数据增强的图片，存到各个对应名字的文件夹里
    """
    args = arg_parser()
    
    if args.func == 'select_image_query':
        select_image_query(args)
    
    elif args.func == 'generate_image_augment':
        augment_method_dict = augment_method()
        
        # 对于各数据增强方法分别执行
        for name, t in augment_method_dict.items():
            print("augment method: ", name)
            args.augment_method = name
            
            # 生成dataset
            if name != 'random_erasing':
                if isinstance(t, list):
                    dataset = MPDataset(args.image_query_dir,
                                        transform=transforms.Compose([*t,
                                                                    transforms.ToTensor(),  # ToTensor会归一化到[0,1]
                                                                    ]))
                else:
                    dataset = MPDataset(args.image_query_dir,
                                        transform=transforms.Compose([t,
                                                                    transforms.ToTensor(),  # ToTensor会归一化到[0,1]
                                                                    ]))
                
            else:
                dataset = MPDataset(args.image_query_dir,
                                    transform=transforms.Compose([transforms.ToTensor(),
                                                                  t, # random erasing只对tensor有效
                                                                ]))
            # print(next(iter(dataset)))
            
            # 通过loader加载数据
            loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0) #在Win下num_workers只能设置为0，否则会报错
            
            # 逐张保存图片
            for i, img_data in enumerate(loader, 0): #enumerate中的0表示可迭代对象索引从0开始，如果是1那么从1开始
                image, image_aug, path = img_data
                # print(image.size())
                # print(image_aug.size())
                # print(path)
                image_save(image[0], image_aug[0], path[0], args) #因为loader的batch_size=1，所以是对每一张图片进行保存
                # print('batch{}:image shape info-->{}, image_aug shape info-->{}, path {}'.format(i, image.size(), image_aug.size(), path[0]))
                #batch0:image shape info-->torch.Size([1, 350, 233, 3]), image_aug shape info-->torch.Size([1, 3, 350, 233])
                # break
    elif args.func == 'generate_inputfile_for_C':
        generate_inputfile_for_C(args)
        
    else:
        print("!!!args.func %s not exist!!!" % args.func)
        
if __name__ == "__main__":
    main()


"""
docker ps -a (esvit:v1)
docker attach container_id
cd /src/data/esvit/preprocess/
python3 generate_augmented_image.py --func select_image_query
python3 generate_augmented_image.py --func generate_image_augment

"""