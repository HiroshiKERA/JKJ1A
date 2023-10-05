import torch # 適宜importせよ


'''
Day1-2をもとに，load_data関数を完成させよ
'''

def load_data(batch_size=128, n_train=15000, n_test=2500, use_all=False):
    '''
    batch_size: バッチサイズ
    n_train   : 訓練用のデータ数
    n_test    : テスト用のデータ数
    use_all   : 全てのデータを使う場合はTrueを与える
    '''