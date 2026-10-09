"""Image classification utilities. Model inputs are RGB tensors in [0, 1]."""
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

CIFAR_CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck')


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class CIFARClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            Normalize((.5, .5, .5), (.5, .5, .5)),
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128), nn.ReLU(), nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.network(x)


def load_cifar10(*, batch_size=128, train_size=None, test_size=None, seed=0):
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor
    # Notebook setup changes the working directory to the project root.
    # Use the same data folder locally and on Colab (Google Drive).
    root = 'data'
    datasets = [CIFAR10(root, train=train, download=True, transform=ToTensor())
                for train in (True, False)]
    loaders = []
    for dataset, size, shuffle in zip(datasets, (train_size, test_size), (True, False)):
        if size is not None:
            if not 1 <= size <= len(dataset):
                raise ValueError(f'size must be between 1 and {len(dataset)}')
            indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:size]
            dataset = Subset(dataset, indices.tolist())
        loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                  generator=torch.Generator().manual_seed(seed), num_workers=0))
    return tuple(loaders)


def evaluate(model, loader, *, device=None):
    # 評価モードに切り替える
    model.eval()

    # 評価では勾配を計算しない
    with torch.no_grad():
        # 分類精度を計算し、返す処理を書く
        pass


def save_classifier(model, path='models/day2-cifar10.pt'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)
    print(f'保存先: {path.resolve()}')


def load_classifier(path='models/day2-cifar10.pt', *, device=None):
    if not Path(path).is_file():
        raise FileNotFoundError(f'{path} がありません。01でモデルを学習・保存してください。')
    model = CIFARClassifier()
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    return model.to(device or get_device()).eval()


def show_predictions(model, x, labels, *, y=None, n=8):
    import matplotlib.pyplot as plt
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    with torch.no_grad():
        probabilities = model(x[:n].to(device)).softmax(1).cpu()
    model.train(was_training)
    n = min(n, len(x))
    fig, axes = plt.subplots(1, n, figsize=(2.4*n, 2.8), squeeze=False, constrained_layout=True)
    for i, ax in enumerate(axes[0]):
        score, label = probabilities[i].max(0)
        ax.imshow(x[i].detach().cpu().permute(1, 2, 0).clamp(0, 1))
        title = f'{labels[label.item()]} ({score:.2f})'
        if y is not None:
            title += f'\ntrue: {labels[int(y[i])]}'
        ax.set_title(title); ax.axis('off')
    plt.show(); plt.close(fig)


def load_imagenet_classifier(name='resnet18.a1_in1k', *, device=None):
    import timm
    from timm.data import create_transform, resolve_model_data_config, ImageNetInfo
    from torchvision.transforms import Compose, Normalize as TVNormalize
    backbone = timm.create_model(name, pretrained=True).eval()
    config = resolve_model_data_config(backbone)
    transform = create_transform(**config, is_training=False)
    # Spatial preprocessing is applied once, before optimizing the input pixels.
    transform = Compose([t for t in transform.transforms if not isinstance(t, TVNormalize)])
    model = nn.Sequential(Normalize(config['mean'], config['std']), backbone)
    labels = ImageNetInfo().label_descriptions()
    return model.to(device or get_device()).eval(), transform, labels
