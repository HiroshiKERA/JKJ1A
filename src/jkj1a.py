from pathlib import Path

from PIL import Image

import os 
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import random_split, DataLoader, Subset, Dataset
import numpy as np

def load_data(batch_size=4, metric='beauty'):  # metric is 'beauty' or 'pleasure'
    img_size = 32

    # 画像への前処理
    transform = transforms.Compose(
            [transforms.Resize(img_size), # original size is 256 x 256
             transforms.ToTensor(),
             ] 
            )
    
    # ラベルに対する前処理（各自で適宜修正）
    target_transform = transforms.Compose(
        [
         transforms.Lambda(torch.tensor),
         transforms.Lambda(lambda x: x / x.norm())  # xは7次元ベクトル（ヒストグラム）．それの正規化している．
        ]
    )

    dataset = JKJ1Adataset(
                transform=transform, 
                target_transform=target_transform,
                metric=metric)

    n_test = 50
    n_train = len(dataset) - n_test
    trainset, testset = random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(42))

    trainloader = DataLoader(trainset, batch_size=batch_size)
    testloader = DataLoader(testset, batch_size=batch_size)

    return trainloader, testloader, metric


class JKJ1Adataset(Dataset):
    def __init__(self, 
                 img_dir='data/JKJ1A-dataset/images', 
                 label_dir='data/JKJ1A-dataset/labels',
                 transform=None,
                 target_transform=None,
                 metric='beauty'):
        
        self.img_dir = img_dir
        self.label_dir = os.path.join(label_dir, metric)

        # 画像ファイルのパス一覧を取得する。
        self.images = self.load_images(self.img_dir)
        self.labels = np.load(os.path.join(self.label_dir, 'histogram.npy'))

        self.transform = transform
        self.target_transform = target_transform
        # self.transform = transforms.Compose( 
        #     [transforms.Resize(size), 
        #      transforms.ToTensor(),
        #      ] # original size is 256 x 256
        #     )

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None: 
            label = self.target_transform(label)

        return img, label

    def _get_img_paths(self, img_dir, IMG_EXTENSIONS=[".jpg", ".jpeg", ".png", ".bmp"]):
        """指定したディレクトリ内の画像ファイルのパス一覧を取得する。
        """
        img_dir = Path(img_dir)
        img_paths = [
            p for p in img_dir.iterdir() if p.suffix in IMG_EXTENSIONS
        ]

        return img_paths

    def load_images(self, img_dir):
        img_paths = sorted(self._get_img_paths(img_dir))
        images = [Image.open(path) for path in img_paths]
        return images

    def _get_label_paths(self, label_dir):
        """指定したディレクトリ内の画像ファイルのパス一覧を取得する。
        """
        label_dir = Path(label_dir)
        label_paths = [
            p for p in label_dir.iterdir()
        ]

        return label_paths

    def __len__(self):
        """ディレクトリ内の画像ファイルの数を返す。
        """
        return len(self.labels)
