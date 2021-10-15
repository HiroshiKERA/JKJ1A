'''
上がってきたラベルの形式が汚かったのでその処理．
'''


import numpy as np
import os 
from pathlib import Path

def _base_label_process(path):
    # with open('data/JKJ1A-dataset/beauty_csv/beauty_image8.csv') as f:
    arr_ = np.zeros(7)
    with open(path) as f:
        arr = []
        while True: 
            a = f.readline()
            if 'Name' in a: break 
            arr.append(list(map(int, a.split(','))))
        
        
        for k, v in arr:
            arr_[k-1] = v

        # print(arr_)
    return arr_


def base_label_process(root_path='data/JKJ1A-dataset/labels', metric='beauty', save=False):
    root_path = os.path.join(root_path, metric)
    hists = []
    for path in sorted(_get_label_paths(root_path)):
        hist = _base_label_process(path)
        hists.append(hist)

    hists = np.asarray(hists, dtype='float')

    if save:
        np.save(os.path.join(root_path, 'histogram.npy'), hists)

    return labels

def _get_label_paths(label_dir):
    """指定したディレクトリ内の画像ファイルのパス一覧を取得する。
    """
    label_dir = Path(label_dir)
    label_paths = [
        p for p in label_dir.iterdir()
    ]

    return label_paths
