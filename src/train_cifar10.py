# モジュールのインポート
import argparse
from utils import get_device
from cifar10 import load_data
from trainer import Trainer

import warnings
warnings.filterwarnings("ignore")  # warningを表示しない

def set_args():
    '''
    !python src/train_cifar10.py --nepochs 2 --batch_size 128 --lr 0.01 --env_name 'dryrun'
    のように，--NAME_OF_HYPTER_PARAMETERS を用いて学習性っていを調整できるようにするためのもの．
    '''
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--nepochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--env_name', type=str, default='')
    args = parser.parse_args()
    return vars(args)

def main():
    # --- 変更不要 --------------------------------
    # GPU or CPUなのか表示
    device = get_device()
    print('using device:', device)

    # `--nepochs 2` などで与えたパラメタの読み込み
    # `args.nepochs` のようにして使える
    args = set_args()

    
    # --- 学習（ここを埋める） ---------------
    trainer = Trainer(args)
    
    

# おまじない（変更しなくて良い）
if __name__ == '__main__':
    main()  # main()が実行される