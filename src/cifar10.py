import torch.nn as nn 
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split, DataLoader

def load_data(batch_size=128, n_train=15000, n_test=2500, use_all=False):
    '''
    batch_size: バッチサイズ
    n_train   : 訓練用のデータ数
    n_test    : テスト用のデータ数
    use_all   : 全てのデータを使う場合はTrueを与える
    '''
    # クラスのラベル名
    classes = ('airplane', 'automobile', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    ## 前処理関数の準備
    transform = transforms.Compose([transforms.ToTensor()])

    # CIFAR10の準備（ローカルにデータがない場合はdataディレクトリにダウンロードされる）
    # 訓練用データセット
    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    # 評価用データセット
    testset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    if not use_all:
        trainset, _ = random_split(trainset, [n_train, len(trainset) - n_train])  # trainsetの内，n_train個だけ選ぶ
        testset, _ = random_split(testset, [n_test, len(testset) - n_test])       # testsetの内，n_test個だけ選ぶ

    # ミニバッチに小分けしておく．これを後で使う
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    # ミニバッチに小分けしておく．これを後で使う
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return (trainloader, testloader, classes)