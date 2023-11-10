import torch
import torch.nn as nn
import torch.nn.functional as F


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
    
class MLP(nn.Module):
    def __init__(self, depth=1, width=100):   
        super().__init__()
        self.depth = depth
        self.width = width
        
        self.hidden_layers = nn.ModuleList()
        
        self.input_layer = nn.Linear(28*28, self.width)  # 入力28*28次元, 出力128次元
        for i in range(depth):
            layer = nn.Linear(self.width, self.width)
            self.hidden_layers.append(layer)
        self.output_layer = nn.Linear(self.width, 10)
 
    def forward(self, x):
        x = x.view(-1, 28*28)  # 28 x 28の画像を 28*28のベクトルにする
        x = F.relu(self.input_layer(x))
        for hlayer in self.hidden_layers:
            x = F.rele(hlayer(x))
        x = self.output_layer(x)
        return x
    
class LeNet(nn.Module):
    # 初期化の時にレイヤーが準備される部分
    def __init__(self):   
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)          # 畳み込み層
        self.pool = nn.MaxPool2d(2, 2)           # プーリング層
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)    # 全結合層   
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # ネットワークにデータを通す時に機能する
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))     # 畳み込み -> 活性化 -> プーリング
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)               # テンソルの形を成形
        x = F.relu(self.fc1(x))                  # 全結合 -> 活性化
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
 