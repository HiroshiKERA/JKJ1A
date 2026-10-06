"""Pixel-space attacks and evaluation for Day 2."""
import torch


def evaluate_attack(model, loader, *, attack, eps=8/255, alpha=2/255,
                    n_steps=10, max_batches=8):
    """同じ画像群で、攻撃前後の分類精度と非標的型攻撃の成功率を計算する。

    Args:
        model: (N, C, H, W)の画像から(N, クラス数)のスコアを返す分類器。
        loader: 画素値が[0, 1]の画像と、正解ラベル(N,)を返すデータローダ。
        attack: 攻撃関数。model, x, yとキーワード引数eps, alpha, n_stepsを
            受け取り、xと同じ形状・デバイスの攻撃画像を返す。
        eps: 各画素に許す元画像からの最大変化。attackへ渡す。
        alpha: 1回の更新幅。attackへ渡す。
        n_steps: 更新回数。attackへ渡す。
        max_batches: 評価する先頭バッチ数。Noneなら全バッチ。

    Returns:
        辞書。nは評価画像数、clean_accuracyとadversarial_accuracyは
        攻撃前後の正解率（0〜1）、initially_correctは攻撃前の正解数。
        success_rateは攻撃前に正解した画像のうち攻撃後に誤った割合。
        攻撃前の正解数が0ならsuccess_rateはNone。loaderは空でないこと。

    Notes:
        評価中はモデルをevalモードにし、終了時に元のモードへ戻す。
        モデルの重みを変更しない攻撃関数を渡すこと。
    """
    if max_batches is not None and max_batches < 1:
        raise ValueError('max_batches must be positive or None')
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    total = clean_correct = adv_correct = successful = 0
    try:
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                clean_ok = model(x).argmax(1) == y
            adv = attack(model, x, y, eps=eps, alpha=alpha, n_steps=n_steps)
            with torch.no_grad():
                adv_ok = model(adv).argmax(1) == y
            total += len(y)
            clean_correct += clean_ok.sum().item()
            adv_correct += adv_ok.sum().item()
            successful += (clean_ok & ~adv_ok).sum().item()
    finally:
        model.train(was_training)
    return {'n': total, 'clean_accuracy': clean_correct/total,
            'adversarial_accuracy': adv_correct/total,
            'initially_correct': clean_correct,
            'success_rate': successful/clean_correct if clean_correct else None}


def show_attack(model, x, adv, labels, *, index=0):
    """元画像・強調した摂動・攻撃画像を横に並べ、分類結果を表示する。

    Args:
        model: 画像バッチからクラスごとのスコアを返す分類器。
        x: 元画像。形状(N, 3, H, W)、画素値[0, 1]のテンソル。
        adv: 攻撃画像。xと同じ形状・デバイスで、画素値は[0, 1]。
        labels: クラス番号順のクラス名の配列。
        index: バッチ内で表示する画像の番号。

    Returns:
        None。図を表示し、元画像と攻撃画像の予測クラスと確信度を添える。

    Notes:
        摂動は見やすいように拡大表示するため、実際の変化量とは異なる。
        予測時はevalモードと勾配計算なしで実行し、元のモードに戻す。
        xとadvはモデルと同じデバイスに置く。
    """
    import matplotlib.pyplot as plt
    was_training = model.training
    model.eval()
    with torch.no_grad():
        scores = model(torch.stack((x[index], adv[index]))).softmax(1).cpu()
    model.train(was_training)
    before, after = x[index].detach().cpu(), adv[index].detach().cpu()
    delta = after - before
    scale = max(delta.abs().max().item(), 1e-12)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), constrained_layout=True)
    for ax, picture in zip(axes, (before, .5 + delta/(2*scale), after)):
        ax.imshow(picture.permute(1, 2, 0).clamp(0, 1)); ax.axis('off')
    for ax, score, title in zip((axes[0], axes[2]), scores, ('Original', 'Attacked')):
        value, label = score.max(0)
        ax.set_title(f'{title}\n{labels[label.item()]} ({value:.3f})')
    axes[1].set_title(f'Perturbation (rescaled)\nmax |delta| = {delta.abs().max():.5f}')
    plt.show(); plt.close(fig)
