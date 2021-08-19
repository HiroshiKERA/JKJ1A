# モジュールのインポート
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# `lenet.py`と`cifar10.py`の中のものを読み込みしてる（アンコメントせよ）
# from lenet import Net 
# from cifar10 import load_data

def train(...):
    ''
    
def test(...):
    ''
    return acc 

# これはおまじない（変更しなくて良い）
def set_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--nepochs', type=int, default=14)
    parser.add_argument('--lr', type=float, default=0.001)
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

    
    # --- データをロード ---------------------------
    
    
    
    
    # --- ネットワークの初期化と学習------------------
    net = Net()




    # --- ネットワークを評価して正解率を表示 ----------





    # --- 変更不要 --------------------------------
    if args.save_model_name:  # 保存先が与えられている場合保存
        PATH = args.save_model_name
        torch.save(net.state_dict(), PATH)



# おまじない（変更しなくて良い）
if __name__ == '__main__':
    main()  # main()が実行される