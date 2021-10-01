# モジュールのインポート
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# `lenet.py`と`cifar10.py`にあるものを読み込んでいる（アンコメントせよ（シャープを外せ））
# from lenet import Net 
# from cifar10 import load_data

import warnings
warnings.filterwarnings("ignore")  # warningを表示しない

def train(net, trainloader, optimizer, criterion, nepochs):
    net.train()  # ネットワークを「訓練モード」にする（おまじない）．

    for epoch in range(nepochs):  
        # --- ここを埋める ---------------
        











        # ------------------------------
    print('Training completed')
    
def test(net, dataloader):
    net.eval()  # ネットワークを「評価モード」にする（おまじない）．

    correct = 0  # 正解数
    total = 0    # 画像総数

    for data in dataloader:
        # --- ここを埋める ---------------
        







        # ------------------------------

    acc = correct / total

    return acc 

# これはおまじない
def set_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--nepochs', type=int, default=14)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--save_model_name', type=str, default='')
    args = parser.parse_args()
    return args 

def main():
    # --- 変更不要 --------------------------------
    # GPU or CPUなのか表示
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('using device:', device)

    # `--nepochs 2` などで与えたパラメタの読み込み
    # `args.nepochs` のようにして使える
    args = set_args()

    
    # --- ここを埋める ---------------
    # --- データをロード ---------------------------
    



    
    
    
    # --- ネットワークの初期化と学習------------------
    
    






    # ------------------------------

    # --- ネットワークを評価して正解率を表示 ---------- 
    train_acc = test(net, trainloader)
    test_acc = test(net, testloader)
    print(f'train acc = {train_acc:.3f}')  # ':.3f'とつけると小数点以下3桁までの表示になる
    print(f' test acc = {test_acc:.3f}')  




    # --- 変更不要 --------------------------------
    if args.save_model_name:  # 保存先が与えられている場合保存
        PATH = args.save_model_name
        torch.save(net.state_dict(), PATH)

    #state_dict = torch.load(args.save_model_name)        # 保存したパラメタをロード
    #net.load_state_dict(state_dict)      # ネットワークにパラメタをセット
    #train_acc = test(net, trainloader)
    #test_acc = test(net, testloader)
    #print(f'train acc = {train_acc:.3f}')  # ':.3f'とつけると小数点以下3桁までの表示になる
    #print(f' test acc = {test_acc:.3f}')  


# おまじない（変更しなくて良い）
if __name__ == '__main__':
    main()  # main()が実行される